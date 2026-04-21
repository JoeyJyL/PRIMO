#!/usr/bin/env python3
"""
vLLM evaluation and comparison utility (single-GPU, greedy decoding, 1568 ctx/out).

Features
--------
1. (Optional) Assemble a VERL FSDP checkpoint (e.g. .../global_step_111) into a
   Hugging Face directory that vLLM can load (offline-friendly, local only).
2. Evaluate pretrained and fine-tuned models on specified datasets (Parquet),
   compute reward via ORRewardCalculator, and summarize results.
3. Support dynamic LoRA loading at inference time (no need to merge weights).
4. For each sample, let LLM answer 10 times. Use a solver (see ORRewardCalculator) to check success, and apply majority vote on successfully solved answers.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Counter

import numpy as np
import pandas as pd
from collections import Counter as PyCounter

# Add the parent directory to PYTHONPATH so that python_interpreter can be imported.
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from python_interpreter import ORRewardCalculator

DEFAULT_DATA_PATH = "./data/data_cleaned"
DEFAULT_DATASETS = ["LowAltitudeOR"]
# DEFAULT_DATASETS = [
#     "industryor",
#     "logior",
#     "mamo_complex",
#     "mamo_easy",
#     "nl4opt",
#     "nlp4lp",
#     "optibench",
# ]

MAJORITY_VOTE_REPEAT = 10


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return super().default(obj)


try:
    from vllm import LLM, SamplingParams
    try:
        from vllm.lora.request import LoRARequest  # type: ignore
        VLLM_LORA_AVAILABLE = True
    except Exception:
        LoRARequest = None  # type: ignore
        VLLM_LORA_AVAILABLE = False
    VLLM_AVAILABLE = True
    print("✓ vLLM available")
except Exception:
    VLLM_AVAILABLE = False
    VLLM_LORA_AVAILABLE = False
    print("✗ vLLM unavailable; install via `pip install vllm`.")


def assemble_checkpoint_to_hf(
    checkpoint_root: str,
    base_model: str,
    target_dir: Optional[Path] = None,
) -> str:
    # (No change to model assembly code)
    import torch
    from transformers import AutoModelForCausalLM

    ckpt_path = Path(checkpoint_root).expanduser().resolve()
    actor_dir = ckpt_path / "actor"
    if not actor_dir.exists():
        raise FileNotFoundError(f"Actor directory not found: {actor_dir}")

    # Collect any plausible model shard files
    shard_paths = sorted(list(actor_dir.glob("**/*.pt")))
    if not shard_paths:
        raise FileNotFoundError(f"No model shards (.pt) found under {actor_dir}")

    print(f"Found {len(shard_paths)} shard(s) in {actor_dir}")

    print(f"Loading base model (local only): {base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        local_files_only=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    all_shards: List[Dict[str, torch.Tensor]] = []
    model_shard_paths = [p for p in shard_paths if "model_" in p.name and "optim_" not in p.name and "extra_state" not in p.name]
    if not model_shard_paths:
        model_shard_paths = shard_paths
    print(f"Loading {len(model_shard_paths)} model shard(s)...")
    for p in model_shard_paths:
        try:
            payload = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as exc:
            print(f"Warning: failed to load shard {p}: {exc}")
            continue
        if isinstance(payload, dict):
            for key in ("state_dict", "module", "model", "model_state_dict", "model_state", "state"):
                if key in payload and isinstance(payload[key], dict):
                    payload = payload[key]
                    break
        if not isinstance(payload, dict):
            continue
        shard_dict: Dict[str, torch.Tensor] = {}
        dtensor_count = 0
        for k, v in payload.items():
            if hasattr(v, "detach"):
                if hasattr(v, "to_local"):
                    try:
                        v = v.to_local()
                        dtensor_count += 1
                    except Exception:
                        pass
                shard_dict[k] = v.detach().cpu()
        if dtensor_count > 0:
            print(f"Converted {dtensor_count} DTensor(s) to local tensors in {p.name}")
        all_shards.append(shard_dict)

    merged_state: Dict[str, torch.Tensor] = {}
    base_state = base.state_dict()

    def get_rank_from_path(p: Path) -> int:
        import re
        match = re.search(r'rank_(\d+)', p.name)
        return int(match.group(1)) if match else 0

    shard_data = [(get_rank_from_path(p), shard) for p, shard in zip(model_shard_paths, all_shards)]
    shard_data.sort(key=lambda x: x[0])

    key_tensors: Dict[str, List[torch.Tensor]] = {}
    for rank, shard in shard_data:
        for k, v in shard.items():
            if k not in key_tensors:
                key_tensors[k] = []
            key_tensors[k].append(v)

    print(f"Total unique keys found: {len(key_tensors)}")
    sample_keys = list(key_tensors.keys())[:3]
    for sk in sample_keys:
        if sk in base_state:
            print(f"  Sample key '{sk}': {len(key_tensors[sk])} shards, shapes: {[t.shape for t in key_tensors[sk][:2]]}, base: {base_state[sk].shape}")

    for k, tensors in key_tensors.items():
        if k not in base_state:
            continue
        base_shape = base_state[k].shape
        shard_shapes = [t.shape for t in tensors]
        num_shards = len(tensors)

        if all(s == base_shape for s in shard_shapes):
            for t in tensors:
                if t.numel() > 0:
                    merged_state[k] = t
                    break
            else:
                merged_state[k] = tensors[0] if tensors else None
            continue

        try:
            merged = False
            if len(base_shape) == 1:
                shard_dim0_sum = sum(s[0] for s in shard_shapes if len(s) > 0)
                if shard_dim0_sum == base_shape[0]:
                    valid_tensors = [t for t in tensors if t.numel() > 0]
                    if valid_tensors and len(valid_tensors) == num_shards:
                        merged_state[k] = torch.cat(valid_tensors, dim=0)
                        merged = True
            else:
                for dim_idx in range(len(base_shape)):
                    shard_dim_sum = sum(s[dim_idx] for s in shard_shapes if dim_idx < len(s))
                    if shard_dim_sum == base_shape[dim_idx]:
                        first_shard_dim = shard_shapes[0][dim_idx] if shard_shapes else 0
                        if shard_dim_sum > first_shard_dim * num_shards:
                            pass
                        elif first_shard_dim * num_shards == shard_dim_sum or shard_dim_sum > first_shard_dim:
                            valid_tensors = [t for t in tensors if t.numel() > 0]
                            if valid_tensors and len(valid_tensors) == num_shards:
                                merged_state[k] = torch.cat(valid_tensors, dim=dim_idx)
                                merged = True
                                break
                        if dim_idx == 0 and not merged:
                            if shard_dim_sum == base_shape[0] and first_shard_dim < base_shape[0]:
                                valid_tensors = [t for t in tensors if t.numel() > 0]
                                if valid_tensors and len(valid_tensors) == num_shards:
                                    merged_state[k] = torch.cat(valid_tensors, dim=0)
                                    merged = True
                                    break

            if not merged and len(base_shape) > 0 and len(shard_shapes) > 0:
                first_shard_shape = shard_shapes[0]
                if len(first_shard_shape) > 0 and first_shard_shape[0] * num_shards == base_shape[0]:
                    if len(first_shard_shape) == len(base_shape):
                        dims_match = all(
                            first_shard_shape[i] == base_shape[i] 
                            for i in range(1, len(base_shape))
                        )
                        if dims_match:
                            valid_tensors = [t for t in tensors if t.numel() > 0]
                            if valid_tensors and len(valid_tensors) == num_shards:
                                merged_state[k] = torch.cat(valid_tensors, dim=0)
                                merged = True

            if not merged:
                if num_shards > 1 and len(base_shape) > 0:
                    valid_tensors = [t for t in tensors if t.numel() > 0]
                    if valid_tensors:
                        try:
                            concatenated = torch.cat(valid_tensors, dim=0)
                            if concatenated.shape[0] == base_shape[0] and len(concatenated.shape) == len(base_shape):
                                if all(concatenated.shape[i] == base_shape[i] for i in range(1, len(base_shape))):
                                    merged_state[k] = concatenated
                                    merged = True
                                elif len(base_shape) == 2 and concatenated.shape[0] == base_shape[0]:
                                    merged_state[k] = concatenated
                                    merged = True
                        except Exception:
                            pass

                if not merged:
                    print(f"Warning: Could not merge key '{k}': {num_shards} shards with shapes {shard_shapes[:2]}... vs base {base_shape}")
                    if tensors:
                        merged_state[k] = max(tensors, key=lambda t: t.numel())

        except Exception as e:
            print(f"Error merging key '{k}': {e}, shapes: {shard_shapes} vs base: {base_shape}")
            if tensors:
                merged_state[k] = tensors[0]

    missing, unexpected = base.load_state_dict(merged_state, strict=False)
    if unexpected:
        print(f"Warning: unexpected parameters while loading shards: {list(unexpected)[:8]} ...")
    if missing:
        print(f"Warning: missing parameters after merge: {len(missing)} keys")

    if target_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_dir = ckpt_path / f"assembled_{timestamp}"
    target_dir = Path(target_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving assembled model to {target_dir}")
    base.save_pretrained(target_dir)

    tok_src = actor_dir / "huggingface"
    if tok_src.exists():
        for item in tok_src.iterdir():
            dst = target_dir / item.name
            if item.is_file():
                shutil.copy2(item, dst)
            elif item.is_dir():
                shutil.copytree(item, dst, dirs_exist_ok=True)

    return str(target_dir)


class VLLMEvaluator:
    """Single-GPU evaluator with optional LoRA at inference time and majority vote."""

    def __init__(
        self,
        model_name: str,
        model_label: Optional[str] = None,
        gpu_id: Optional[int] = None,
        execution_bonus: float = 0.0,
        gpu_memory_utilization: float = 0.7,
        max_model_len: int = 8192,
        max_num_seqs: int = 1,
        max_num_batched_tokens: int = 8192,
        lora_adapter_path: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.model_label = model_label or Path(model_name).name
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            print(f"Using CUDA device: {gpu_id}")

        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens

        self.reward_calculator = ORRewardCalculator()
        self.execution_bonus = execution_bonus

        self.llm: Optional[LLM] = None
        self.sampling_params: Optional[SamplingParams] = None
        self.lora_adapter_path: Optional[str] = lora_adapter_path
        self.lora_request = None

        self._load_model()

    def _load_model(self) -> None:
        enable_lora_flag = bool(self.lora_adapter_path)
        self.llm = LLM(
            model=self.model_name,
            dtype="bfloat16",
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            trust_remote_code=True,
            enforce_eager=True,
            disable_custom_all_reduce=True,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_seqs=self.max_num_seqs,
            tensor_parallel_size=1,
            enable_lora=enable_lora_flag,
        )

        # Sampling parameters:
        # To introduce some randomness for voting, set temperature=0.8, top_p=1.0
        self.sampling_params = SamplingParams(
            temperature=0.6,
            top_p=1.0,
            top_k=-1,
            max_tokens=8192,
            repetition_penalty=1.05,
        )

        if self.lora_adapter_path and VLLM_LORA_AVAILABLE:
            try:
                lora_name = self.model_label or "eval_lora"
                self.lora_request = LoRARequest(
                    lora_name=lora_name,
                    lora_path=self.lora_adapter_path,
                    lora_weight=1.0,
                )
                print(f"✓ LoRA adapter prepared: {self.lora_adapter_path}")
            except Exception as exc:
                print(f"✗ Failed to prepare LoRA adapter: {exc}")
                self.lora_request = None

        print(f"✓ vLLM model loaded: {self.model_name}")
    def format_prompt_for_llama3(self, prompt_text: str) -> Optional[str]:
        """
        格式化prompt以适配Llama 3模型输入格式。
        约定的格式类似：
        <|begin_of_text|><|start_header_id|>user<|end_header_id|>
        用户的指令...
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
        prefix = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
        suffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

        formatted_prompt = f"{prefix}{prompt_text}\n{suffix}"

        # 粗略限制输入长度，防异常输入，约等于~1568 tokens
        if len(formatted_prompt) > 12800:
            print(f"Warning: LLAMA-3 Prompt too long ({len(formatted_prompt)} characters); skipping sample.")
            return None
        return formatted_prompt
    def format_prompt_for_qwen(self, prompt_text: str) -> Optional[str]:
        system_message = "You are an expert in optimization modeling and programming."
        formatted_prompt = (
            f"<|im_start|>system\n{system_message}<|im_end|>\n"
            f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        # Rough character cap to avoid pathological inputs; aligned to ~1568 tokens
        if len(formatted_prompt) > 12800:
            print(f"Warning: Prompt too long ({len(formatted_prompt)} characters); skipping sample.")
            return None
        return formatted_prompt

    def parse_response(self, response: str) -> Tuple[str, str]:
        raw_response = response.strip()
        think_section = ""
        answer_section = raw_response

        def _strip_outer_code_fence(text: str) -> str:
            stripped = text.strip()
            if stripped.startswith("```json"):
                stripped = stripped[len("```json") :].strip()
            elif stripped.startswith("```"):
                stripped = stripped[len("```") :].strip()
            if stripped.endswith("```"):
                stripped = stripped[:-3].strip()
            return stripped

        if raw_response:
            json_candidate = _strip_outer_code_fence(raw_response)
            parsed_payload = None
            try:
                parsed_payload = json.loads(json_candidate)
            except json.JSONDecodeError:
                try:
                    parsed_payload = ast.literal_eval(json_candidate)
                except (ValueError, SyntaxError):
                    parsed_payload = None
            if isinstance(parsed_payload, dict):
                think_value = parsed_payload.get("think", "")
                if isinstance(think_value, str):
                    think_section = think_value.strip()
                answer_value = parsed_payload.get("answer", "")
                if isinstance(answer_value, str):
                    answer_section = answer_value.strip()
                else:
                    answer_section = ""

        if "\n" in answer_section:
            answer_section = answer_section.replace("\n", "\n").replace("\t", "\t")

        code = self._extract_code_from_answer(answer_section if answer_section else raw_response)
        return think_section, code

    @staticmethod
    def _extract_code_from_answer(answer_text: str) -> str:
        if not answer_text:
            return ""
        text = answer_text.strip()

        def _clean_lines(candidate: str) -> str:
            lines = candidate.split("\n")
            cleaned = [line for line in lines if not line.strip().isdigit()]
            return "\n".join(cleaned).strip()

        def _fix_escape_sequences(code: str) -> str:
            if not code:
                return code
            if code.startswith("\\n") or (code.startswith("\\") and "\\n" in code[:50]):
                try:
                    import ast
                    if (code.startswith('"') and code.endswith('"')) or (code.startswith("'") and code.endswith("'")):
                        code = ast.literal_eval(code)
                        return code
                except (ValueError, SyntaxError):
                    pass
                try:
                    decoded = code.encode('latin-1').decode('unicode_escape')
                    if decoded.strip() and not decoded.startswith("\\"):
                        return decoded
                except (UnicodeDecodeError, UnicodeEncodeError):
                    pass
                if code.startswith("\\n"):
                    code = "\n" + code[2:]
                if "\\n" in code and code.count("\\n") > code.count("\n"):
                    code = code.replace("\\n", "\n")
                    code = code.replace("\\t", "\t")
                    code = code.replace("\\r", "\r")
            return code

        if "```python" in text:
            start = text.find("```python") + len("```python")
            end = text.find("```", start)
            if end != -1:
                code = _clean_lines(text[start:end].strip())
                return _fix_escape_sequences(code)
        if "```" in text:
            start = text.find("```") + len("```")
            end = text.find("```", start)
            if end != -1:
                code = _clean_lines(text[start:end].strip())
                return _fix_escape_sequences(code)
        if text.startswith("```python"):
            code = _clean_lines(text[9:].strip())
            return _fix_escape_sequences(code)

        lines = text.split("\n")
        code_lines: List[str] = []
        in_code = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("import ", "from ", "def ", "class ", "if ", "for ", "while ", "try:", "with ")):
                in_code = True
            if in_code:
                code_lines.append(line)
        code = _clean_lines("\n".join(code_lines)) if code_lines else text
        return _fix_escape_sequences(code)

    @staticmethod
    def _result_to_key(result: Any) -> str:
        """
        Convert execution result to a hashable key for voting.
        Handles various result types: numbers, lists, tuples, dicts, etc.
        
        Note: This method should only be called with non-None results that have
        passed the execution success check. The None check here is defensive programming
        only - None values should be filtered out before calling this method.
        """
        import json
        # Defensive check: None values should not reach here (filtered before voting)
        if result is None:
            return "None"
        if isinstance(result, (int, float)):
            # For numerical results, use rounded value to handle floating point precision
            return f"num:{result:.10f}" if isinstance(result, float) else f"num:{result}"
        if isinstance(result, (list, tuple)):
            # Convert list/tuple to JSON string for consistent representation
            try:
                return f"list:{json.dumps(result, sort_keys=True)}"
            except (TypeError, ValueError):
                # Fallback: convert to string representation
                return f"list:{str(result)}"
        if isinstance(result, dict):
            # Convert dict to JSON string with sorted keys
            try:
                return f"dict:{json.dumps(result, sort_keys=True)}"
            except (TypeError, ValueError):
                return f"dict:{str(result)}"
        # For other types, use string representation
        return f"other:{str(result)}"

    def calculate_reward(self, generated_code: str, ground_truth: str, extra_info: str = "") -> Tuple[float, str, bool]:
        try:
            reward, explanation, executed = self.reward_calculator.calculate_reward(
                generated_code, ground_truth, extra_info
            )
            final_reward = reward + (self.execution_bonus if executed else 0.0)
            return float(final_reward), explanation, executed
        except Exception as exc:
            print(f"Error while computing reward: {exc}")
            return 0.0, f"Error while computing reward: {exc}", False

    def load_dataset(self, dataset_path: str, dataset_name: str) -> List[Dict[str, Any]]:
        try:
            df = pd.read_parquet(dataset_path)
        except Exception as exc:
            print(f"Error: failed to load dataset {dataset_name}: {exc}")
            return []
        if df.empty:
            print(f"Warning: {dataset_name} dataset is empty.")
            return []
        print(f"Loaded {dataset_name}: {len(df)} samples")
        return [row.to_dict() for _, row in df.iterrows()]

    def evaluate_batch_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        For each sample, have the LLM generate MAJORITY_VOTE_REPEAT answers,
        run solver for each, and use majority vote among solutions with successfully executed solver.
        """
        if self.llm is None or self.sampling_params is None:
            raise RuntimeError("LLM is not initialized.")
        print(f"Evaluating {len(samples)} samples with {MAJORITY_VOTE_REPEAT} answers per sample (majority vote)...")

        valid_samples: List[Dict[str, Any]] = []
        formatted_prompts: List[str] = []
        prompts_per_sample: List[int] = []

        for sample in samples:
            prompt_data = sample.get("prompt")
            if prompt_data is None:
                print(f"Warning: sample missing prompt field: {sample.keys()}")
                continue
            if isinstance(prompt_data, list) and prompt_data:
                prompt_text = prompt_data[0].get("content", "")
            else:
                prompt_text = str(prompt_data)
            formatted_prompt = self.format_prompt_for_qwen(prompt_text)
            if formatted_prompt is not None:
                valid_samples.append(sample)
                prompts_per_sample.append(len(formatted_prompts))
                formatted_prompts.extend([formatted_prompt] * MAJORITY_VOTE_REPEAT)

        if not formatted_prompts:
            print("No valid prompts remain after formatting (all skipped due to length or missing data).")
            return []

        print(f"Generating total {len(formatted_prompts)} responses with vLLM... (valid samples: {len(valid_samples)})")
        start_time = time.time()
        try:
            if self.lora_request is not None:
                outputs = self.llm.generate(formatted_prompts, self.sampling_params, lora_request=self.lora_request)
            else:
                outputs = self.llm.generate(formatted_prompts, self.sampling_params)
            batch_generation_time = time.time() - start_time
            print(f"✓ Batch generation completed in {batch_generation_time:.2f} seconds")
            all_responses = [output.outputs[0].text for output in outputs]
        except Exception as exc:
            print(f"✗ Batch generation failed: {exc}")
            all_responses = ["Error: Generation failed"] * len(formatted_prompts)
            batch_generation_time = time.time() - start_time

        # Group responses per sample
        results: List[Dict[str, Any]] = []
        for idx, sample in enumerate(valid_samples):
            print(f"\n--- Sample {idx + 1}/{len(valid_samples)} ---")
            # Extract the responses for this sample
            resp_start = idx * MAJORITY_VOTE_REPEAT
            resp_end = resp_start + MAJORITY_VOTE_REPEAT
            responses = all_responses[resp_start:resp_end]

            ground_truth = sample.get("ground_truth", "")

            # Evaluate/parse all responses, collect validated/executed results
            think_traces = []
            codes = []
            rewards = []
            explanations = []
            executed_flags = []
            execution_results = []  # Store execution results
            voted_correct_code = None
            voted_think_trace = None
            voted_explanation = ""
            voted_reward = 0.0
            voted_executed = False
            voted_is_correct = False

            # First collect all successfully executed answer codes
            # Use execution result as voting key instead of code string
            result_successes_counter = PyCounter()
            solution_map = {}  # Maps result_key -> (code, think_trace, explanation, reward)
            successful_codes = []
            successful_think_traces = []
            successful_explanations = []
            successful_rewards = []

            for r_i, response in enumerate(responses):
                think_trace, generated_code = self.parse_response(response)
                
                # Execute code once to get result (avoid duplicate execution)
                # Only results from successfully executed code with non-None values will be used for voting
                execution_result = None
                executed = False
                error_msg = None
                try:
                    success, result, error_msg = self.reward_calculator.interpreter.execute_code(generated_code)
                    # Only mark as executed if: (1) execution succeeded AND (2) result is not None
                    # This filters out: failed executions, None results, and other invalid cases
                    if success and result is not None:
                        execution_result = result
                        executed = True
                except Exception as exc:
                    error_msg = str(exc)
                    executed = False
                
                # Calculate reward based on execution result
                if executed and execution_result is not None:
                    try:
                        # Parse ground truth
                        gt_value = self.reward_calculator._parse_ground_truth(ground_truth)
                        # Compare results to get reward
                        reward, explanation = self.reward_calculator._compare_results(execution_result, gt_value, "")
                        reward = float(reward) + (self.execution_bonus if executed else 0.0)
                    except Exception as exc:
                        reward = 0.0
                        explanation = f"Reward calculation error: {exc}"
                else:
                    reward = 0.0
                    explanation = f"Code execution failed: {error_msg if error_msg else 'Unknown error'}"
                
                # Store info for all responses (including failed ones for logging)
                think_traces.append(think_trace)
                codes.append(generated_code)
                rewards.append(float(reward))
                explanations.append(explanation)
                executed_flags.append(executed)
                execution_results.append(execution_result)
                
                # Only add to voting if: (1) code executed successfully AND (2) result is not None
                # This ensures we only vote on valid execution results, excluding:
                # - Failed code executions (success=False)
                # - None results (result=None)
                # - Other invalid execution outcomes
                if executed and execution_result is not None:
                    # Use execution result as voting candidate
                    # Convert result to a hashable key for voting
                    result_key = self._result_to_key(execution_result)
                    result_successes_counter[result_key] += 1
                    # Store mapping: result_key -> (code, think_trace, explanation, reward, result)
                    if result_key not in solution_map:
                        solution_map[result_key] = (generated_code, think_trace, explanation, float(reward), execution_result)
                    successful_codes.append(generated_code)
                    successful_think_traces.append(think_trace)
                    successful_explanations.append(explanation)
                    successful_rewards.append(float(reward))

            # Majority vote among executable answers (by execution result)
            voted_result_key = None
            if successful_codes:
                majority_result_key, majority_cnt = result_successes_counter.most_common(1)[0]
                voted_result_key = majority_result_key
                voted_correct_code, voted_think_trace, voted_explanation, voted_reward, voted_result = solution_map[majority_result_key]
                voted_executed = True
                voted_is_correct = voted_reward > 0.5
            else:
                # Fallback: use the first response
                voted_correct_code = codes[0] if codes else ""
                voted_think_trace = think_traces[0] if think_traces else ""
                voted_explanation = explanations[0] if explanations else ""
                voted_reward = rewards[0] if rewards else 0.0
                voted_executed = executed_flags[0] if executed_flags else False
                voted_is_correct = voted_reward > 0.5

            # Print report for this sample
            status = "✓ Correct" if voted_is_correct else "✗ Incorrect"
            voted_cnt = result_successes_counter[voted_result_key] if (voted_executed and voted_result_key is not None) else 0
            print(f"{status} (Reward: {voted_reward:.3f}, executed:{voted_executed}, voted_cnt={voted_cnt})")
            if voted_explanation:
                print(f"  Explanation: {voted_explanation[:200] + '...' if len(voted_explanation)>200 else voted_explanation}")
            if voted_think_trace:
                think_preview = voted_think_trace[:200] + "..." if len(voted_think_trace) > 200 else voted_think_trace
                print(f"  Think trace: {think_preview}")
            if voted_correct_code:
                code_preview = voted_correct_code[:200] + "..." if len(voted_correct_code) > 200 else voted_correct_code
                print(f"  Generated code (majority-voted): {code_preview}")

            if isinstance(sample.get("prompt"), list) and sample["prompt"]:
                prompt_for_result = sample["prompt"][0].get("content", "")
            else:
                prompt_for_result = str(sample.get("prompt", "No prompt available"))

            result = {
                "task_id": sample.get("task_id", "N/A"),
                "prompt": prompt_for_result,
                "ground_truth": ground_truth,
                "response": voted_correct_code,
                "think_trace": voted_think_trace,
                "generated_code": voted_correct_code,
                "reward": float(voted_reward),
                "explanation": voted_explanation,
                "is_correct": voted_is_correct,
                "executed_successfully": voted_executed,
                "generation_time": batch_generation_time / len(valid_samples) if valid_samples else 0.0,
                "response_length": len(voted_correct_code),
                "think_length": len(voted_think_trace),
                "code_length": len(voted_correct_code),
                "timestamp": datetime.now().isoformat(),
                "majority_vote_count": voted_cnt,
                "all_codes": codes,
                "all_rewards": rewards,
                "all_executed_flags": executed_flags,
                "all_execution_results": execution_results,
            }
            results.append(result)

        return results

    def evaluate_all_datasets(
        self,
        datasets: List[str],
        base_data_path: str,
        split: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
        all_results: List[Dict[str, Any]] = []
        dataset_stats: Dict[str, Dict[str, float]] = {}

        for dataset_name in datasets:
            print(f"\n{'=' * 60}")
            print(f"Evaluating dataset: {dataset_name} (split: {split})")
            print(f"{'=' * 60}")

            samples: List[Dict[str, Any]] = []
            train_path = Path(base_data_path) / dataset_name / "train.parquet"
            test_path = Path(base_data_path) / dataset_name / "test.parquet"

            if split in ("train", "train+test") and train_path.exists():
                train_samples = self.load_dataset(str(train_path), f"{dataset_name}_train")
                samples.extend(train_samples)
                print(f"Training split: {len(train_samples)} samples.")
            elif split in ("train", "train+test"):
                print(f"Warning: training split missing: {train_path}")

            if split in ("test", "train+test") and test_path.exists():
                test_samples = self.load_dataset(str(test_path), f"{dataset_name}_test")
                samples.extend(test_samples)
                print(f"Test split: {len(test_samples)} samples.")
            elif split in ("test", "train+test"):
                print(f"Warning: test split missing: {test_path}")

            if not samples:
                print(f"Warning: no usable data for dataset {dataset_name}.")
                continue

            dataset_results = self.evaluate_batch_samples(samples)

            total_count = len(dataset_results)
            reward_values = [res["reward"] for res in dataset_results]
            avg_reward = float(np.mean(reward_values)) if reward_values else 0.0
            executed_successes = sum(1 for res in dataset_results if res["executed_successfully"])
            execution_success_rate = executed_successes / total_count if total_count > 0 else 0.0
            avg_generation_time = float(
                np.mean([res["generation_time"] for res in dataset_results if res["generation_time"] > 0])
            ) if dataset_results else 0.0

            print(f"\nDataset {dataset_name} summary:")
            print(f"  Total samples: {total_count}")
            print(f"  Executed successfully: {executed_successes} ({execution_success_rate:.2%})")
            print(f"  Average reward: {avg_reward:.3f}")
            print(f"  Average generation time: {avg_generation_time:.2f} s")

            dataset_stats[dataset_name] = {
                "total_samples": total_count,
                "executed_successes": executed_successes,
                "execution_success_rate": execution_success_rate,
                "avg_reward": avg_reward,
                "avg_generation_time": avg_generation_time,
            }

            for result in dataset_results:
                result["dataset"] = dataset_name
                result["model"] = self.model_label

            all_results.extend(dataset_results)

        return all_results, dataset_stats

    def shutdown(self) -> None:
        if self.llm is not None:
            try:
                self.llm.llm_engine.shutdown()
            except Exception:
                pass
            self.llm = None


def compute_overall_summary(
    results: List[Dict[str, Any]],
    dataset_stats: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    total_samples = len(results)
    reward_values = [res["reward"] for res in results]
    overall_avg_reward = float(np.mean(reward_values)) if reward_values else 0.0
    execution_successes = sum(1 for res in results if res["executed_successfully"])
    execution_success_rate = execution_successes / total_samples if total_samples > 0 else 0.0
    generation_times = [res["generation_time"] for res in results if res["generation_time"] > 0]
    overall_avg_generation_time = float(np.mean(generation_times)) if generation_times else 0.0
    return {
        "total_samples": total_samples,
        "overall_avg_reward": overall_avg_reward,
        "execution_success_rate": execution_success_rate,
        "overall_avg_generation_time": overall_avg_generation_time,
        "executed_successes": execution_successes,
        "datasets_covered": list(dataset_stats.keys()),
    }


def compare_model_statistics(
    pretrained_summary: Optional[Dict[str, Any]],
    trained_summary: Optional[Dict[str, Any]],
    pretrained_dataset_stats: Dict[str, Dict[str, float]],
    trained_dataset_stats: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    comparison: Dict[str, Any] = {"overall": {}, "by_dataset": {}}
    if pretrained_summary and trained_summary:
        comparison["overall"] = {
            "pretrained": pretrained_summary,
            "trained": trained_summary,
            "delta_avg_reward": trained_summary["overall_avg_reward"] - pretrained_summary["overall_avg_reward"],
            "delta_execution_success_rate": (
                trained_summary["execution_success_rate"] - pretrained_summary["execution_success_rate"]
            ),
            "delta_avg_generation_time": (
                trained_summary["overall_avg_generation_time"] - pretrained_summary["overall_avg_generation_time"]
            ),
        }
    elif pretrained_summary:
        comparison["overall"] = {"pretrained": pretrained_summary}
    elif trained_summary:
        comparison["overall"] = {"trained": trained_summary}

    all_datasets = sorted(set(pretrained_dataset_stats.keys()) | set(trained_dataset_stats.keys()))
    for dataset in all_datasets:
        pre_stats = pretrained_dataset_stats.get(dataset)
        post_stats = trained_dataset_stats.get(dataset)
        entry: Dict[str, Any] = {}
        if pre_stats:
            entry["pretrained"] = pre_stats
        if post_stats:
            entry["trained"] = post_stats
        if pre_stats and post_stats:
            entry["delta_avg_reward"] = post_stats["avg_reward"] - pre_stats["avg_reward"]
            entry["delta_execution_success_rate"] = (
                post_stats["execution_success_rate"] - pre_stats["execution_success_rate"]
            )
            entry["delta_avg_generation_time"] = post_stats["avg_generation_time"] - pre_stats["avg_generation_time"]
        comparison["by_dataset"][dataset] = entry
    return comparison


def run_single_evaluation(
    model_path: str,
    label: str,
    datasets: List[str],
    base_data_path: str,
    split: str,
    gpu_id: Optional[int],
    execution_bonus: float,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    lora_adapter_path: Optional[str],
) -> Dict[str, Any]:
    evaluator = VLLMEvaluator(
        model_path,
        model_label=label,
        gpu_id=gpu_id,
        execution_bonus=execution_bonus,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        lora_adapter_path=lora_adapter_path,
    )
    try:
        results, dataset_stats = evaluator.evaluate_all_datasets(datasets, base_data_path, split)
    finally:
        evaluator.shutdown()
    summary = compute_overall_summary(results, dataset_stats)
    return {
        "label": label,
        "model_path": model_path,
        "split": split,
        "summary": summary,
        "dataset_statistics": dataset_stats,
        "detailed_results": results,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="vLLM evaluation script (single-GPU) with majority vote.")
    parser.add_argument("--datasets", type=str, nargs="+", default=["complexor"], help="Datasets to evaluate.")
    parser.add_argument("--split", type=str, choices=["test", "train", "train+test"], default="test")
    parser.add_argument("--base-data-path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default="./models/Qwen2.5-7B-Instruct",
        help="Pretrained model path or HF id (local preferred).",
    )
    parser.add_argument("--trained-model", type=str, default=None, help="Assembled trained model path (HF dir).")
    parser.add_argument(
        "--trained-checkpoint-root",
        type=str,
        default=None,
        help="Path to a VERL checkpoint directory (e.g. .../global_step_111) to assemble and evaluate.",
    )
    parser.add_argument("--skip-pretrained", action="store_true", help="Skip evaluating the pretrained base model.")
    parser.add_argument("--gpu-id", type=int, default=None, help="Single CUDA device index (e.g. 0).")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--execution-bonus", type=float, default=0.0)
    parser.add_argument("--lora-adapter", type=str, default=None, help="Optional PEFT LoRA adapter directory.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./scripts/training/outputs/evaluation",
        help="Directory to store assembled models and evaluation JSON.",
    )

    args = parser.parse_args()

    if not VLLM_AVAILABLE:
        print("Please install vLLM first (`pip install vllm`).")
        return

    datasets = DEFAULT_DATASETS if "all" in args.datasets else args.datasets
    os.makedirs(args.output_dir, exist_ok=True)

    print("CLI arguments:")
    print(f"  Datasets: {datasets}")
    print(f"  Split: {args.split}")
    print(f"  Base data path: {args.base_data_path}")
    print(f"  Pretrained model: {args.pretrained_model}")
    print(f"  Trained model (assembled): {args.trained_model}")
    print(f"  Trained checkpoint root: {args.trained_checkpoint_root}")
    print(f"  Skip pretrained: {args.skip_pretrained}")
    print(f"  GPU id: {args.gpu_id if args.gpu_id is not None else 'default'}")
    print(f"  GPU memory utilization: {args.gpu_memory_utilization}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Majority vote n_repeat: {MAJORITY_VOTE_REPEAT}")

    assembled_model_path = args.trained_model
    if args.trained_checkpoint_root:
        assembled_root = Path(args.output_dir) / "assembled_models"
        assembled_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_dir = assembled_root / f"{Path(args.trained_checkpoint_root).name}_{timestamp}"
        assembled_model_path = assemble_checkpoint_to_hf(args.trained_checkpoint_root, args.pretrained_model, target_dir)
        print(f"Assembled checkpoint saved to: {assembled_model_path}")

    evaluations: Dict[str, Any] = {}
    if not args.skip_pretrained:
        print(f"\n{'#' * 60}\nEvaluating pretrained model\n{'#' * 60}")
        evaluations["pretrained"] = run_single_evaluation(
            args.pretrained_model,
            label="pretrained",
            datasets=datasets,
            base_data_path=args.base_data_path,
            split=args.split,
            gpu_id=args.gpu_id,
            execution_bonus=args.execution_bonus,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            lora_adapter_path=args.lora_adapter,
        )

    if assembled_model_path:
        print(f"\n{'#' * 60}\nEvaluating trained model\n{'#' * 60}")
        evaluations["trained"] = run_single_evaluation(
            assembled_model_path,
            label="trained",
            datasets=datasets,
            base_data_path=args.base_data_path,
            split=args.split,
            gpu_id=args.gpu_id,
            execution_bonus=args.execution_bonus,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            lora_adapter_path=args.lora_adapter,
        )

    pretrained_summary = evaluations.get("pretrained", {}).get("summary")
    trained_summary = evaluations.get("trained", {}).get("summary")
    pretrained_dataset_stats = evaluations.get("pretrained", {}).get("dataset_statistics", {})
    trained_dataset_stats = evaluations.get("trained", {}).get("dataset_statistics", {})

    comparison = compare_model_statistics(pretrained_summary, trained_summary, pretrained_dataset_stats, trained_dataset_stats)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    datasets_str = "_".join(datasets) if len(datasets) <= 3 else f"{len(datasets)}datasets"
    split_tag = args.split.replace("+", "_")
    pre_tag = Path(args.pretrained_model).name.replace(":", "_") if args.pretrained_model else "nopre"
    trained_tag = Path(assembled_model_path).name.replace(":", "_") if assembled_model_path else "notrained"
    output_file = Path(args.output_dir) / f"majority_vote_{trained_tag}_{datasets_str}_{split_tag}_{timestamp}.json"

    final_payload = {
        "timestamp": datetime.now().isoformat(),
        "datasets": datasets,
        "split": args.split,
        "base_data_path": args.base_data_path,
        "pretrained_model": args.pretrained_model if not args.skip_pretrained else None,
        "trained_model": assembled_model_path,
        "execution_bonus": args.execution_bonus,
        "tensor_parallel_size": 1,
        "gpu_id": args.gpu_id,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "pretrained": evaluations.get("pretrained"),
        "trained": evaluations.get("trained"),
        "comparison": comparison,
        "majority_vote_repeat": MAJORITY_VOTE_REPEAT,
    }

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(final_payload, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    print(f"\n{'=' * 60}")
    print("Evaluation complete!")
    print(f"{'=' * 60}")
    print(f"Results saved to: {output_file}")

    if comparison.get("overall"):
        overall = comparison["overall"]
        delta_reward = overall.get("delta_avg_reward")
        delta_success = overall.get("delta_execution_success_rate")
        delta_time = overall.get("delta_avg_generation_time")
        if delta_reward is not None:
            print(
                f"Δreward={delta_reward:+.3f}, Δsuccess={delta_success:+.2%}, Δtime={delta_time:+.2f}s (trained - pretrained)"
            )

    if comparison.get("by_dataset"):
        print("\nPer-dataset comparison:")
        for dataset, stats in comparison["by_dataset"].items():
            pre_stats = stats.get("pretrained")
            post_stats = stats.get("trained")
            line = f"  {dataset}: "
            if pre_stats:
                line += f"pre(avg_reward={pre_stats['avg_reward']:.3f}, success={pre_stats['execution_success_rate']:.2%})"
            if post_stats:
                line += f" | post(avg_reward={post_stats['avg_reward']:.3f}, success={post_stats['execution_success_rate']:.2%})"
            if "delta_avg_reward" in stats:
                line += (
                    f" | Δreward={stats['delta_avg_reward']:+.3f}, "
                    f"Δsuccess={stats.get('delta_execution_success_rate', 0.0):+.2%}, "
                    f"Δtime={stats.get('delta_avg_generation_time', 0.0):+.2f}s"
                )
            print(line)


if __name__ == "__main__":
    main()
