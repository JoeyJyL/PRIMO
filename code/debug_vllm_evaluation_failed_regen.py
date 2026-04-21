#!/usr/bin/env python3
"""
Debug script for re-generating failed executions with a fixed prompt template.

This script:
1. Loads evaluation results from a JSON file
2. Extracts tasks with executed_successfully=False
3. Rebuilds a new prompt using problem/model/code/error context
4. Uses vLLM to regenerate code
5. Re-evaluates regenerated code and merges back
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add parent directory for python_interpreter import.
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from python_interpreter import ORRewardCalculator

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


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return super().default(obj)


class VLLMDebugger:
    """Base debugger with vLLM generation and reward evaluation."""

    def __init__(
        self,
        model_name: str,
        model_label: Optional[str] = None,
        gpu_id: Optional[int] = None,
        execution_bonus: float = 0.0,
        gpu_memory_utilization: float = 0.6,
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
        self.execution_bonus = execution_bonus
        self.reward_calculator = ORRewardCalculator()

        self.llm: Optional[LLM] = None
        self.sampling_params: Optional[SamplingParams] = None
        self.lora_adapter_path = lora_adapter_path
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
        self.sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=8192,
            repetition_penalty=1.1,
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

    def format_prompt_for_qwen(self, prompt_text: str) -> Optional[str]:
        system_message = "You are an expert Gurobipy developer and debugger. Your task is to fix the bug in the Gurobipy code."
        formatted_prompt = (
            f"<|im_start|>system\n{system_message}<|im_end|>\n"
            f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        if len(formatted_prompt) > 32672:
            print(f"Warning: Prompt too long ({len(formatted_prompt)} chars); skipping sample.")
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

        if "```python" in text:
            start = text.find("```python") + len("```python")
            end = text.find("```", start)
            if end != -1:
                return _clean_lines(text[start:end].strip())
        if "```" in text:
            start = text.find("```") + len("```")
            end = text.find("```", start)
            if end != -1:
                return _clean_lines(text[start:end].strip())

        lines = text.split("\n")
        code_lines: List[str] = []
        in_code = False
        starters = ("import ", "from ", "def ", "class ", "if ", "for ", "while ", "try:", "with ")
        for line in lines:
            if line.strip().startswith(starters):
                in_code = True
            if in_code:
                code_lines.append(line)
        return _clean_lines("\n".join(code_lines)) if code_lines else text

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

    def shutdown(self) -> None:
        if self.llm is not None:
            try:
                self.llm.llm_engine.shutdown()
            except Exception:
                pass
            self.llm = None


def load_evaluation_results(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_failed_samples(evaluation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract samples with executed_successfully=False."""
    failed_samples: List[Dict[str, Any]] = []
    if "trained" in evaluation_results and "detailed_results" in evaluation_results["trained"]:
        for result in evaluation_results["trained"]["detailed_results"]:
            if not result.get("executed_successfully", True):
                failed_samples.append(result)
    elif "detailed_results" in evaluation_results:
        for result in evaluation_results["detailed_results"]:
            if not result.get("executed_successfully", True):
                failed_samples.append(result)
    print(f"Found {len(failed_samples)} failed samples")
    return failed_samples


def merge_results(
    original_results: Dict[str, Any],
    debug_results: List[Dict[str, Any]],
    debug_iteration: int,
) -> Dict[str, Any]:
    merged_results = copy.deepcopy(original_results)
    debug_map = {r["task_id"]: r for r in debug_results}

    if "trained" in merged_results and "detailed_results" in merged_results["trained"]:
        updated_results = []
        for result in merged_results["trained"]["detailed_results"]:
            task_id = result.get("task_id")
            if task_id in debug_map:
                updated_results.append(debug_map[task_id])
            else:
                updated_results.append(result)
        merged_results["trained"]["detailed_results"] = updated_results

    if "trained" in merged_results and "summary" in merged_results["trained"]:
        total_samples = len(merged_results["trained"]["detailed_results"])
        executed_successes = sum(
            1 for r in merged_results["trained"]["detailed_results"] if r.get("executed_successfully", False)
        )
        execution_success_rate = executed_successes / total_samples if total_samples > 0 else 0.0
        reward_values = [r.get("reward", 0.0) for r in merged_results["trained"]["detailed_results"]]
        avg_reward = float(np.mean(reward_values)) if reward_values else 0.0

        merged_results["trained"]["summary"]["executed_successes"] = executed_successes
        merged_results["trained"]["summary"]["execution_success_rate"] = execution_success_rate
        merged_results["trained"]["summary"]["overall_avg_reward"] = avg_reward

        if "dataset_statistics" in merged_results["trained"]:
            dataset_stats = merged_results["trained"]["dataset_statistics"]
            for dataset_name in dataset_stats:
                dataset_results = [
                    r
                    for r in merged_results["trained"]["detailed_results"]
                    if r.get("dataset") == dataset_name
                ]
                if dataset_results:
                    dataset_total = len(dataset_results)
                    dataset_executed = sum(1 for r in dataset_results if r.get("executed_successfully", False))
                    dataset_success_rate = dataset_executed / dataset_total if dataset_total > 0 else 0.0
                    dataset_rewards = [r.get("reward", 0.0) for r in dataset_results]
                    dataset_avg_reward = float(np.mean(dataset_rewards)) if dataset_rewards else 0.0

                    dataset_stats[dataset_name]["total_samples"] = dataset_total
                    dataset_stats[dataset_name]["executed_successes"] = dataset_executed
                    dataset_stats[dataset_name]["execution_success_rate"] = dataset_success_rate
                    dataset_stats[dataset_name]["avg_reward"] = dataset_avg_reward

    if "debug_history" not in merged_results:
        merged_results["debug_history"] = []
    merged_results["debug_history"].append(
        {
            "iteration": debug_iteration,
            "timestamp": datetime.now().isoformat(),
            "fixed_samples": len(debug_results),
        }
    )
    merged_results["debug_iteration"] = debug_iteration
    merged_results["debug_timestamp"] = datetime.now().isoformat()
    return merged_results


class FailedCasePromptDebugger(VLLMDebugger):
    """Regenerate failed samples with a fixed debugging prompt template."""

    @staticmethod
    def _extract_prompt_content(prompt_raw: Any) -> str:
        if isinstance(prompt_raw, list) and prompt_raw:
            first = prompt_raw[0]
            if isinstance(first, dict):
                return str(first.get("content", ""))

        if isinstance(prompt_raw, str):
            # Try to parse serialized chat payload first.
            try:
                parsed = json.loads(prompt_raw)
                if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                    return str(parsed[0].get("content", ""))
            except Exception:
                pass

            try:
                parsed = ast.literal_eval(prompt_raw)
                if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                    return str(parsed[0].get("content", ""))
            except Exception:
                pass

            return prompt_raw

        return ""

    @staticmethod
    def _first_non_empty(sample: Dict[str, Any], keys: List[str]) -> str:
        for key in keys:
            value = sample.get(key, "")
            if isinstance(value, str) and value.strip():
                return value
        return ""

    def _build_regen_prompt(self, sample: Dict[str, Any]) -> str:
        prompt_content = self._extract_prompt_content(sample.get("prompt", ""))

        problem_description = self._first_non_empty(
            sample,
            [
                "original_optimization_problem_description",
                "optimization_problem_description",
                "problem_description",
            ],
        )
        if not problem_description:
            problem_description = prompt_content

        mathematical_formulations = self._first_non_empty(
            sample,
            [
                "mathematical_formulations",
                "mathematical_formulation",
                "mathematical_model",
            ],
        )

        solver_code = self._first_non_empty(
            sample,
            [
                "solver_code",
                "generated_code",
            ],
        )
        error_message = self._first_non_empty(
            sample,
            [
                "error_message",
                "explanation",
            ],
        )

        return (
            "You are an expert Gurobipy developer and debugger. Your task is to fix the bug in the Gurobipy code.\n\n"
            "The problem description:\n\n"
            f"{problem_description}\n\n"
            "The mathematical model:\n\n"
            f"{mathematical_formulations}\n\n"
            "The Gurobipy code:\n\n"
            f"{solver_code}\n\n"
            "The error message during code execution:\n\n"
            f"{error_message}\n\n"
            "Please fix the code and return the fixed code:"
        )

    def debug_batch_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.llm is None or self.sampling_params is None:
            raise RuntimeError("LLM is not initialized.")

        print(f"Regenerating {len(samples)} failed samples with fixed prompt...")
        formatted_prompts: List[str] = []
        valid_samples: List[Dict[str, Any]] = []

        for sample in samples:
            new_prompt = self._build_regen_prompt(sample)
            formatted_prompt = self.format_prompt_for_qwen(new_prompt)
            if formatted_prompt is None:
                print(f"Warning: prompt too long, skip task {sample.get('task_id', 'N/A')}")
                continue
            formatted_prompts.append(formatted_prompt)
            valid_samples.append(sample)

        if not formatted_prompts:
            print("No valid samples remain after prompt formatting.")
            return []

        print(f"Generating responses with vLLM... (valid samples: {len(formatted_prompts)})")
        outputs = (
            self.llm.generate(formatted_prompts, self.sampling_params, lora_request=self.lora_request)
            if self.lora_request is not None
            else self.llm.generate(formatted_prompts, self.sampling_params)
        )
        responses = [output.outputs[0].text for output in outputs]

        now = datetime.now().isoformat()
        results: List[Dict[str, Any]] = []
        for sample, response in zip(valid_samples, responses):
            ground_truth = sample.get("ground_truth", "")
            think_trace, fixed_code = self.parse_response(response)
            reward, explanation, executed = self.calculate_reward(fixed_code, ground_truth, "")

            results.append(
                {
                    "task_id": sample.get("task_id", "N/A"),
                    "prompt": sample.get("prompt", ""),
                    "ground_truth": ground_truth,
                    "response": response,
                    "think_trace": think_trace,
                    "generated_code": fixed_code,
                    "reward": float(reward),
                    "explanation": explanation,
                    "is_correct": reward > 0.5,
                    "executed_successfully": executed,
                    "generation_time": 0.0,
                    "response_length": len(response),
                    "think_length": len(think_trace),
                    "code_length": len(fixed_code),
                    "timestamp": now,
                    "dataset": sample.get("dataset", ""),
                    "model": sample.get("model", ""),
                    "debug_iteration": sample.get("debug_iteration", 0) + 1,
                    "original_generated_code": sample.get("generated_code", ""),
                    "original_explanation": sample.get("explanation", ""),
                    "debug_note": "regenerated with fixed failed-case prompt",
                }
            )

        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate failed execution cases with fixed prompt.")
    parser.add_argument("--input-json", type=str, required=True, help="Path to input evaluation JSON.")
    parser.add_argument("--model-path", type=str, default=None, help="Model path, optional if in JSON.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./scripts/training/outputs/evaluation",
        help="Directory to store debug outputs.",
    )
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--execution-bonus", type=float, default=0.0)
    parser.add_argument("--lora-adapter", type=str, default=None)
    args = parser.parse_args()

    if not VLLM_AVAILABLE:
        print("Please install vLLM first (`pip install vllm`).")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    original_results = load_evaluation_results(args.input_json)

    model_path = args.model_path
    if model_path is None:
        model_path = (
            original_results.get("trained_model")
            or original_results.get("trained", {}).get("model_path")
        )
        if not model_path:
            raise ValueError("Model path not specified and not found in JSON. Use --model-path.")

    debugger = FailedCasePromptDebugger(
        model_name=model_path,
        model_label="failed_case_regen",
        gpu_id=args.gpu_id,
        execution_bonus=args.execution_bonus,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        lora_adapter_path=args.lora_adapter,
    )

    try:
        current_results = copy.deepcopy(original_results)
        for iteration in range(1, args.max_iterations + 1):
            failed_samples = extract_failed_samples(current_results)
            if not failed_samples:
                print("No failed samples found. Debugging complete!")
                break

            failed_samples_with_iter = []
            for sample in failed_samples:
                sample_copy = copy.deepcopy(sample)
                sample_copy["debug_iteration"] = iteration - 1
                failed_samples_with_iter.append(sample_copy)

            debug_results = debugger.debug_batch_samples(failed_samples_with_iter)
            current_results = merge_results(current_results, debug_results, iteration)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = Path(args.input_json).stem
            intermediate_file = Path(args.output_dir) / f"{stem}_failed_regen_iter{iteration}_{ts}.json"
            with intermediate_file.open("w", encoding="utf-8") as f:
                json.dump(current_results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
            print(f"Intermediate results saved to: {intermediate_file}")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = Path(args.input_json).stem
        final_file = Path(args.output_dir) / f"{stem}_failed_regen_final_{ts}.json"
        with final_file.open("w", encoding="utf-8") as f:
            json.dump(current_results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        print(f"Final results saved to: {final_file}")
    finally:
        debugger.shutdown()


if __name__ == "__main__":
    main()
