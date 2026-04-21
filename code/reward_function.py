"""
Custom reward function for GRPO training
"""

import ast
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from typing import Any, Dict
import requests
import sys
# Ensure we can import test_prm_api from the same directory as this script
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
import test_prm_api

# Add project path to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from python_interpreter import ORRewardCalculator
from prompt_templates import ORPromptTemplate


# -----------------------------------------------------------------------------
# Logging and timeout configuration
# -----------------------------------------------------------------------------
LOGGER = logging.getLogger("verl.reward")
DEFAULT_TIMEOUT_SECONDS = float(os.environ.get("VERL_REWARD_TIMEOUT_SECONDS", 60.0))


def _run_reward_worker(
    result_queue: "mp.Queue",
    data_source: str,
    generated_code: str,
    ground_truth: str,
    problem_description: str,
    execution_bonus: float,
) -> None:
    """Worker process to execute reward calculation safely."""

    try:
        calculator = ORRewardCalculator()
        reward, explanation, executed = calculator.calculate_reward(
            generated_code, ground_truth, problem_description
        )
        result_queue.put(
            {
                "status": "ok",
                "base_reward": reward,
                "explanation": explanation,
                "executed": executed,
                "execution_bonus": execution_bonus if executed else 0.0,
            }
        )
    except Exception as exc:  # pragma: no cover - defensive path
        result_queue.put(
            {
                "status": "error",
                "base_reward": 0.0,
                "explanation": f"Reward worker exception: {exc}",
                "executed": False,
                "execution_bonus": 0.0,
            }
        )


def test_reward_function():
    """Test reward function"""
    # Test the compute_score function directly
    result = compute_score(
        data_source='complexor',
        solution_str='''```python
import gurobipy as gp
from gurobipy import GRB

model = gp.Model()
x = model.addVar(name='x')
y = model.addVar(name='y')
model.setObjective(2*x + 3*y, GRB.MINIMIZE)
model.addConstr(x + y >= 1, 'constraint1')
model.addConstr(x >= 0, 'constraint2')
model.addConstr(y >= 0, 'constraint3')
model.optimize()
result = model.objVal
print(f'Optimal value: {result}')
```''',
        ground_truth='2.0',
        extra_info={'problem_description': 'Minimize 2x + 3y, subject to: x + y >= 1, x >= 0, y >= 0'}
    )
    
    print(f"Reward: {result}")


# VERL expected standalone function
def compute_score(data_source=None, solution_str=None, ground_truth=None, extra_info=None, **kwargs):
    """
    VERL expected standalone compute_score function
    Modified to support direct api mean_reward input as rm_score (based on test_prm_api.py logic).
    
    Automatically detects training vs validation mode and uses different rm_weight:
    - Training mode: uses VERL_TRAIN_RM_WEIGHT (default: 0.1)
    - Validation mode: uses VERL_VAL_RM_WEIGHT (default: 0.0)
    
    Detection method: checks call stack for "_validate" method or "validation" in file path.
    """
    import requests
    import inspect

    # 检测是训练还是验证模式
    is_validation = False
    try:
        # 方法1: 检查环境变量（手动指定）
        if os.environ.get("VERL_USE_VAL_MODE", "").lower() in {"1", "true", "yes"}:
            is_validation = True
        else:
            # 方法2: 从调用栈自动检测
            stack = inspect.stack()
            for frame_info in stack:
                func_name = frame_info.function.lower()
                file_name = frame_info.filename.lower()
                # 如果调用栈中有_validate方法或包含validation的文件，判定为validation
                if "_validate" in func_name or "validation" in file_name:
                    is_validation = True
                    break
    except Exception:
        # 检测失败时默认使用训练模式
        pass

    mode_str = "validation" if is_validation else "training"

    debug_enabled = os.environ.get("VERL_DEBUG_REWARD", "").lower() in {"1", "true", "yes"}

    # Display debugging information when explicitly enabled
    if debug_enabled and solution_str:
        print("\n" + "="*80)
        print(f"DEBUGGING INFO - MODE: {mode_str.upper()}")
        print("DEBUGGING INFO - MODEL RESPONSE:")
        print(solution_str)
        print("-"*80)
        print("DEBUGGING INFO - GROUND TRUTH:")
        print(ground_truth or '')
        print("="*80 + "\n")
        print(f"DEBUGGING INFO - CONTEXT: data_source={data_source}, kwargs={kwargs}")

    def _coerce_float(value, default=None):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    execution_bonus = kwargs.get('execution_bonus')
    if execution_bonus is None:
        env_bonus = os.environ.get("VERL_EXECUTION_BONUS")
        execution_bonus = _coerce_float(env_bonus, 1)
    else:
        execution_bonus = _coerce_float(execution_bonus, 1)

    timeout_seconds = float(kwargs.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))

    # Extract code from solution_str
    generated_code = _extract_code(solution_str or '')
    # print(f"📝solution_str: {solution_str}")
    # print(f"📝extra_info: {extra_info}")
    
    if not generated_code:
        return 0.0

    problem_description = ""
    if isinstance(extra_info, dict):
        problem_description = extra_info.get("problem_description", "")

    LOGGER.info(
        "compute_score [%s] start dataset=%s timeout=%.1fs code_len=%d bonus=%.3f",
        mode_str,
        data_source,
        timeout_seconds,
        len(generated_code),
        execution_bonus,
    )

    try:
        ctx = mp.get_context("fork")
    except ValueError:  # Fallback for platforms without fork
        ctx = mp.get_context()
    result_queue: "mp.Queue" = ctx.Queue(maxsize=1)
    worker = ctx.Process(
        target=_run_reward_worker,
        args=(
            result_queue,
            data_source,
            generated_code,
            ground_truth or "",
            problem_description,
            execution_bonus,
        ),
        daemon=True,
    )

    start_time = time.perf_counter()
    worker.start()
    worker.join(timeout_seconds)

    result_payload: Dict[str, Any]

    if worker.is_alive():
        worker.terminate()
        worker.join(timeout=5.0)
        duration = time.perf_counter() - start_time
        LOGGER.warning(
            "compute_score [%s] timeout dataset=%s duration=%.2fs", mode_str, data_source, duration
        )
        return 0.0

    duration = time.perf_counter() - start_time

    try:
        result_payload = result_queue.get_nowait()
    except Exception:  # pragma: no cover - defensive path
        LOGGER.error(
            "compute_score [%s] no-result dataset=%s duration=%.2fs", mode_str, data_source, duration
        )
        return 0.0
    finally:
        result_queue.close()
        result_queue.join_thread()

    status = result_payload.get("status")
    base_reward = float(result_payload.get("base_reward", 0.0))
    executed = bool(result_payload.get("executed", False))
    bonus = float(result_payload.get("execution_bonus", 0.0))
    explanation = result_payload.get("explanation")

    if status != "ok":
        LOGGER.warning(
            "compute_score [%s] error dataset=%s duration=%.2fs explanation=%s",
            mode_str,
            data_source,
            duration,
            explanation,
        )
        return 0.0

    rule_based_score = base_reward

    # print(f"❓problem_description: {problem_description}")
    # print(f"❕solution_str: {solution_str}")
    rm_score = test_prm_api.test_api(problem_description, solution_str)
    if rm_score is None:
        rm_score = 0.0
    
    # 根据训练/验证模式使用不同的rm_weight
    if is_validation:
        # Validation模式：使用VERL_VAL_RM_WEIGHT环境变量
        rm_weight = float(os.environ.get("VERL_VAL_RM_WEIGHT", "0.0"))
    else:
        # Training模式：使用VERL_TRAIN_RM_WEIGHT环境变量
        # rm_weight = float(os.environ.get("VERL_TRAIN_RM_WEIGHT", "0.0"))
        rm_weight = 0.1
    
    rm_score_normalized = (rm_score - 0.80) / (1.0 - 0.80)
    
    final_reward = rule_based_score * (1 - rm_weight) + rm_score_normalized * rm_weight + bonus * rm_weight

    # print(f"🚀[{mode_str}] rm_score: {rm_score}, normalized: {rm_score_normalized:.3f}, rm_weight: {rm_weight:.3f}, final_reward: {final_reward:.3f}")

    LOGGER.info(
        "compute_score [%s] done dataset=%s duration=%.2fs reward=%.3f base=%.3f executed=%s",
        mode_str,
        data_source,
        duration,
        final_reward,
        base_reward,
        executed,
    )

    return final_reward


def _extract_code(response: str) -> str:
    """Extract code by directly returning the `answer` field without heuristics."""

    if not response:
        return ""

    raw_response = response.strip()

    def _strip_outer_code_fence(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```json"):
            stripped = stripped[len("```json"):].strip()
        elif stripped.startswith("```"):
            stripped = stripped[len("```"):].strip()
        if stripped.endswith("```"):
            stripped = stripped[:-3].strip()
        return stripped

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
        answer_section = parsed_payload.get("answer")
        if isinstance(answer_section, str):
            code_candidate = answer_section
        else:
            # 支持同时提取 <answer> 后的文本
            raw_section = raw_response
            idx = raw_section.find("<answer>")
            if idx != -1:
                code_candidate = raw_section[idx + len("<answer>"):].strip()
            else:
                code_candidate = ""
    else:
        # 支持直接从字符串中提取 <answer> 后的文本
        idx = raw_response.find("<answer>")
        if idx != -1:
            code_candidate = raw_response[idx + len("<answer>"):].strip()
        else:
            code_candidate = raw_response

    if not code_candidate:
        return ""

    if "\\n" in code_candidate:
        code_candidate = code_candidate.replace("\\n", "\n").replace("\\t", "\t")

    text = code_candidate.strip()

    marker = "```python"
    if marker not in text:
        return ""

    start = text.find(marker) + len(marker)
    snippet = text[start:].strip()

    close_idx = snippet.find("```")
    if close_idx != -1:
        snippet = snippet[:close_idx]

    return snippet.strip()


if __name__ == "__main__":
    test_reward_function()
