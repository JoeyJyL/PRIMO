"""
Python Code Interpreter Module
For executing LLM-generated OR problem solving code and calculating rewards
"""

import ast
import sys
import io
import contextlib
import traceback
import numbers
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import re
import json
import tempfile
import os


class SafePythonInterpreter:
    """Safe Python code interpreter for executing OR problem solving code"""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.allowed_modules = {
            'numpy', 'np', 'pandas', 'pd', 'math', 'random', 
            'gurobipy', 'gp', 'gurobi', 'scipy', 'scipy.optimize',
            'itertools', 'collections', 'functools', 'sys', 'os'
        }
        self.allowed_functions = {
            'min', 'max', 'sum', 'len', 'range', 'enumerate', 
            'zip', 'map', 'filter', 'sorted', 'abs', 'round',
            'int', 'float', 'str', 'list', 'dict', 'set', 'tuple'
        }
    
    def _check_code_safety(self, code: str) -> bool:
        """Check code safety to prevent malicious code execution"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Let the Python interpreter surface syntax errors instead of blocking here.
            return True
        
        # Define the functions and operations that are genuinely dangerous.
        dangerous_functions = {
            'eval', 'exec', 'compile', '__import__', 'open', 'file',
            'input', 'raw_input', 'exit', 'quit', 'reload',
            'dir', 'vars', 'locals', 'globals', 'hasattr', 'getattr', 'setattr',
            'delattr', 'callable', 'isinstance', 'issubclass'
        }
        
        dangerous_modules = {
            'os', 'sys', 'subprocess', 'shutil', 'glob', 'tempfile',
            'pickle', 'marshal', 'shelve', 'dbm', 'sqlite3',
            'socket', 'urllib', 'http', 'ftplib', 'smtplib',
            'threading', 'multiprocessing', 'ctypes'
        }
        
        for node in ast.walk(tree):
            # Reject unsafe import statements.
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in dangerous_modules:
                        return False
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module in dangerous_modules:
                    return False
            
            # Reject calls to blocked functions.
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in dangerous_functions:
                        return False
                elif isinstance(node.func, ast.Attribute):
                    # Reject dangerous method invocations on modules or objects.
                    if hasattr(node.func, 'attr'):
                        dangerous_methods = {
                            'system', 'popen', 'spawn', 'fork', 'kill',
                            'remove', 'unlink', 'rmdir', 'rmtree',
                            'chmod', 'chown', 'rename', 'move',
                            'copyfile', 'copytree', 'move'
                        }
                        if node.func.attr in dangerous_methods:
                            return False
            
            # Prevent overrides of critical namespaces.
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id in ['__builtins__', '__globals__', '__locals__']:
                            return False
            
            # Prevent access to sensitive class metadata.
            if isinstance(node, ast.Attribute):
                if hasattr(node, 'attr'):
                    if node.attr in ['__class__', '__bases__', '__subclasses__', '__mro__']:
                        return False
        
        return True
    
    def execute_code(self, code: str) -> Tuple[bool, Any, str]:
        """
        Execute Python code and return result
        
        Args:
            code: Python code to execute
            
        Returns:
            (success, result, error_message)
        """
        if not self._check_code_safety(code):
            return False, None, "Code contains unsafe operations"
        
        # Create safe execution environment
        safe_globals = {
            '__builtins__': {
                # Core built-ins permitted in the sandbox.
                'min': min, 'max': max, 'sum': sum, 'len': len,
                'range': range, 'enumerate': enumerate, 'zip': zip,
                'map': map, 'filter': filter, 'sorted': sorted,
                'abs': abs, 'round': round, 'int': int, 'float': float,
                'str': str, 'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
                'print': print, 'type': type, 'isinstance': isinstance,
                'all': all, 'any': any, 'bool': bool, 'chr': chr, 'ord': ord,
                'hex': hex, 'oct': oct, 'bin': bin, 'pow': pow, 'divmod': divmod,
                'reversed': reversed, 'slice': slice, 'super': super,
                'property': property, 'staticmethod': staticmethod, 'classmethod': classmethod,
                # Allow __import__ but constrain modules through explicit whitelists.
                '__import__': __import__
            },
            'np': np, 'numpy': np,
            'pd': pd, 'pandas': pd,
            'math': __import__('math'),
            'random': __import__('random'),
            'itertools': __import__('itertools'),
            'collections': __import__('collections'),
            'functools': __import__('functools')
        }
        
        # Add optimization solvers
        try:
            import gurobipy as gp
            from gurobipy import GRB
            safe_globals['gurobipy'] = gp
            safe_globals['gp'] = gp
            safe_globals['GRB'] = GRB
        except ImportError:
            pass
        
        try:
            from scipy import optimize
            safe_globals['scipy'] = __import__('scipy')
            safe_globals['optimize'] = optimize
        except ImportError:
            pass
        
        # Capture output
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        original_cwd = os.getcwd()

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            with tempfile.TemporaryDirectory(prefix="orlm_exec_") as temp_dir:
                try:
                    os.chdir(temp_dir)

                    # Execute code
                    exec(code, safe_globals)

                    # Try to get scalar result
                    result = None
                    fallback_value = None
                    candidate_keys = ['result', 'optimal_value', 'objective_value', 'objVal', 'solution']

                    for key in candidate_keys:
                        if key in safe_globals:
                            value = safe_globals[key]
                            scalar = self._coerce_scalar(value)
                            if scalar is not None:
                                result = scalar
                                break
                            if fallback_value is None:
                                fallback_value = value

                    if result is None:
                        model_scalar = self._extract_model_objective(safe_globals)
                        if model_scalar is not None:
                            result = model_scalar

                    if result is None and fallback_value is not None:
                        result = fallback_value

                    output = stdout_capture.getvalue()
                    error_output = stderr_capture.getvalue()

                    if error_output:
                        return False, None, f"Execution error: {error_output}"

                    return True, result, output

                finally:
                    os.chdir(original_cwd)

        except Exception as e:
            error_msg = f"Code execution error: {str(e)}\n{traceback.format_exc()}"
            return False, None, error_msg

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    @staticmethod
    def _coerce_scalar(value: Any) -> Optional[float]:
        """Return scalar float if value represents a numeric quantity."""
        if isinstance(value, numbers.Number):
            return float(value)

        # Handle 0-d numpy arrays
        if isinstance(value, np.ndarray) and value.ndim == 0:
            return float(value.item())

        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return None

        return None

    @staticmethod
    def _extract_model_objective(env: Dict[str, Any]) -> Optional[float]:
        """Search for optimization models with an objective value."""
        for obj in env.values():
            try:
                if hasattr(obj, 'objVal'):
                    obj_val = obj.objVal
                    if isinstance(obj_val, numbers.Number):
                        return float(obj_val)
            except Exception:
                continue
        return None


class ORRewardCalculator:
    """OR problem reward calculator"""
    
    def __init__(self):
        self.interpreter = SafePythonInterpreter()
    
    def calculate_reward(self, generated_code: str, ground_truth: str, 
                        problem_description: str) -> Tuple[float, str, bool]:
        """
        Calculate reward for OR problems
        
        Args:
            generated_code: LLM-generated solving code
            ground_truth: Standard answer
            problem_description: Problem description
            
        Returns:
            (reward, explanation)
        """
        execution_success = False

        try:
            # Execute generated code
            success, result, error_msg = self.interpreter.execute_code(generated_code)
            
            if not success:
                return 0.0, f"Code execution failed: {error_msg}", execution_success
            
            execution_success = True
            
            # Parse ground truth
            try:
                gt_value = self._parse_ground_truth(ground_truth)
            except Exception as e:
                return 0.0, f"Ground truth parsing failed: {str(e)}", execution_success
            
            # Compare results
            if result is None:
                return 0.0, "Code did not return valid result", execution_success
            
            # Calculate reward
            reward, explanation = self._compare_results(result, gt_value, problem_description)
            
            return reward, explanation, execution_success
            
        except Exception as e:
            return 0.0, f"Reward calculation error: {str(e)}", execution_success
    
    def _parse_ground_truth(self, ground_truth: str) -> Any:
        """Parse ground truth value"""
        # Try to parse as number directly
        try:
            return float(ground_truth)
        except ValueError:
            pass
        
        # Try to parse as JSON
        try:
            return json.loads(ground_truth)
        except json.JSONDecodeError:
            pass
        
        # Try to extract numbers from string
        numbers = re.findall(r'-?\d+\.?\d*', ground_truth)
        if numbers:
            return float(numbers[0])
        
        # If all fail, return original string
        return ground_truth
    
    def _compare_results(self, result: Any, ground_truth: Any, 
                        problem_description: str) -> Tuple[float, str]:
        """Compare results and calculate binary reward (0 or 1)"""
        try:
            # Convert to numerical values for comparison
            if isinstance(result, (int, float)) and isinstance(ground_truth, (int, float)):
                # Binary numerical comparison with tolerance or relative difference within 1%
                if abs(result - ground_truth) < 0.1 or (abs(result - ground_truth) / (abs(ground_truth) + 1e-5) < 0.01):
                    return 1.0, "Result is correct within 0.1/1%"
                else:
                    return 0.0, f"Result is incorrect: {result} vs {ground_truth}"
            
            elif isinstance(result, (list, tuple)) and isinstance(ground_truth, (list, tuple)):
                # Binary list comparison - all elements must be exactly equal
                if len(result) != len(ground_truth):
                    return 0.0, f"Result length mismatch: {len(result)} vs {len(ground_truth)}"
                
                # Check if all elements are exactly equal
                for r, gt in zip(result, ground_truth):
                    if isinstance(r, (int, float)) and isinstance(gt, (int, float)):
                        if abs(r - gt) >= 0.1:
                            return 0.0, f"Result elements differ: {result} vs {ground_truth}"
                    else:
                        if str(r).strip() != str(gt).strip():
                            return 0.0, f"Result elements differ: {result} vs {ground_truth}"
                
                return 1.0, "All result elements are exactly correct"
            
            else:
                # Binary string comparison - must be exactly equal
                if str(result).strip() == str(ground_truth).strip():
                    return 1.0, "Result matches exactly"
                else:
                    return 0.0, f"Result does not match: {result} vs {ground_truth}"
        
        except Exception as e:
            return 0.0, f"Result comparison error: {str(e)}"


def test_interpreter():
    """Test interpreter functionality"""
    interpreter = SafePythonInterpreter()
    calculator = ORRewardCalculator()
    
    # Test simple code
    test_code = """
import numpy as np
result = 2 + 3
"""
    
    success, result, error = interpreter.execute_code(test_code)
    print(f"Test code execution: {success}, result: {result}, error: {error}")
    
    # Test reward calculation
    reward, explanation, executed = calculator.calculate_reward(
        test_code, "5", "Simple addition problem"
    )
    print(f"Reward: {reward}, explanation: {explanation}, executed: {executed}")


if __name__ == "__main__":
    test_interpreter()
