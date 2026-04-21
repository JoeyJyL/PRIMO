"""
OR Problem Prompt Templates
For providing structured OR problem solving prompts to LLM
"""

from typing import Dict, Any, List
import re


class ORPromptTemplate:
    """OR problem prompt template class"""
    
    def __init__(self):
        self.base_prompt = """You are an operations research expert, skilled at converting natural language descriptions of optimization problems into Python code using Gurobi.

Please generate complete Python code to solve the following optimization problem based on the problem description.
Your primary objective is to produce executable Python that models the problem end-to-end, solves it with Gurobi, and assigns the optimal objective value to the variable 'result'.

Response format requirements:
1. Return a JSON object with keys "think" and "answer"
2. Put all reasoning, math derivations, and intermediate discussion in the string value of "think"
3. Put ONLY the final executable Python code inside the "answer" value, wrapped in a single ```python ... ``` block
4. The code must be complete and executable, use Gurobi (gurobipy), and assign the optimal objective value (a single float) to variable 'result'
5. If decision variables or other outputs need to be exposed, store them in separate variables; leave 'result' as the scalar objective
6. If the problem is infeasible, set result = None and document the reason inside the code comments (not in think)
7. Do not duplicate the problem statement or add explanations outside the JSON structure

Problem Description:
{problem_description}

"""

        self.system_prompt = """You are a professional operations research expert, skilled at converting natural language descriptions of optimization problems into executable Python code using Gurobi. Your tasks are:

1. Understand the mathematical modeling requirements of the problem
2. Use Gurobi (gurobipy) as the optimization solver
3. Write complete and correct Python code
4. Ensure the code can execute correctly and return the optimal solution

Always follow these principles:
- Code must be complete and executable
- Use Gurobi (gurobipy) as the primary optimization solver
- Include clear variable definitions and constraint conditions
- Assign the optimal objective value (scalar float) to variable 'result'; place auxiliary data in other variables as needed
- If the problem is infeasible, set result = None and explain the reason in code comments
- Use proper Gurobi syntax and methods"""

    def create_prompt(self, problem_description: str, 
                     problem_type: str = "general") -> str:
        """
        Create simplified OR problem prompt
        
        Args:
            problem_description: Problem description
            problem_type: Problem type (not used in simplified version)
            
        Returns:
            Simplified prompt string
        """
        # Create simple and concise prompt
        prompt = f"""{problem_description}

Return your response as JSON: {{"think": "...", "answer": "```python ... ```"}}. Your goal is to write executable Python that completely solves the problem with Gurobi, storing the optimal objective value (float) in 'result'. Keep all reasoning in think and use separate variables for any extra outputs."""
        
        return prompt
    
    def _get_type_guidance(self, problem_type: str) -> str:
        """Get specific guidance based on problem type"""
        guidance_map = {
            "linear_programming": """
**PROBLEM TYPE: Linear Programming (LP)**

This is a linear programming problem where both the objective function and constraints are linear.

**Modeling Guidelines:**
- **Variables**: Use `model.addVar()` for single variables or `model.addVars()` for multiple variables
- **Objective**: Set using `model.setObjective(expression, GRB.MINIMIZE/MAXIMIZE)`
- **Constraints**: Add using `model.addConstr(expression, sense, rhs)` where sense is '<=', '>=', or '=='
- **Variable Bounds**: Set using `lb` and `ub` parameters in `addVar()`
- **Result**: Extract objective value with `model.objVal` and variable values with `var.X`

**Example Structure:**
```python
# Variables
x = model.addVar(lb=0, name="x")
y = model.addVar(lb=0, name="y")

# Objective
model.setObjective(2*x + 3*y, GRB.MINIMIZE)

# Constraints
model.addConstr(x + y >= 1, "constraint1")
model.addConstr(x <= 5, "constraint2")
```""",
            
            "integer_programming": """
**PROBLEM TYPE: Integer Programming (IP)**

This is an integer programming problem where some or all variables must be integers.

**Modeling Guidelines:**
- **Integer Variables**: Use `vtype=GRB.INTEGER` for integer variables
- **Binary Variables**: Use `vtype=GRB.BINARY` for 0-1 variables
- **Mixed Integer**: Combine continuous and integer variables as needed
- **Objective**: Set using `model.setObjective(expression, GRB.MINIMIZE/MAXIMIZE)`
- **Constraints**: Add using `model.addConstr(expression, sense, rhs)`
- **Result**: Extract objective value with `model.objVal` and variable values with `var.X`

**Example Structure:**
```python
# Binary variables
x = model.addVar(vtype=GRB.BINARY, name="x")
y = model.addVar(vtype=GRB.BINARY, name="y")

# Integer variables
z = model.addVar(vtype=GRB.INTEGER, lb=0, name="z")

# Objective
model.setObjective(x + 2*y + 3*z, GRB.MAXIMIZE)

# Constraints
model.addConstr(x + y <= 1, "binary_constraint")
model.addConstr(z <= 10, "integer_bound")
```""",
            
            "quadratic_programming": """
**PROBLEM TYPE: Quadratic Programming (QP)**

This is a quadratic programming problem with quadratic objective function.

**Modeling Guidelines:**
- **Variables**: Use `model.addVar()` or `model.addVars()`
- **Quadratic Objective**: Use `model.setObjective()` with quadratic terms
- **Linear Constraints**: Add using `model.addConstr()`
- **Quadratic Constraints**: Use `model.addQConstr()` for quadratic constraints
- **Result**: Extract objective value with `model.objVal`

**Example Structure:**
```python
# Variables
x = model.addVar(lb=0, name="x")
y = model.addVar(lb=0, name="y")

# Quadratic objective
model.setObjective(x*x + 2*x*y + y*y, GRB.MINIMIZE)

# Linear constraints
model.addConstr(x + y >= 1, "constraint1")
```""",
            
            "network_flow": """
**PROBLEM TYPE: Network Flow**

This is a network flow problem involving nodes, edges, and flow constraints.

**Modeling Guidelines:**
- **Flow Variables**: Use `model.addVars()` for flow between nodes
- **Flow Balance**: Ensure inflow = outflow for each node (except source/sink)
- **Capacity Constraints**: Limit flow on each edge
- **Objective**: Minimize cost or maximize flow
- **Result**: Extract flow values and objective

**Example Structure:**
```python
# Flow variables: flow[i,j] from node i to node j
flow = model.addVars(nodes, nodes, lb=0, name="flow")

# Flow balance constraints
for i in nodes:
    if i != source and i != sink:
        model.addConstr(flow.sum(i, '*') == flow.sum('*', i), f"balance_{i}")

# Capacity constraints
for i, j in edges:
    model.addConstr(flow[i,j] <= capacity[i,j], f"capacity_{i}_{j}")
```""",
            
            "assignment": """
**PROBLEM TYPE: Assignment Problem**

This is an assignment problem where tasks are assigned to resources.

**Modeling Guidelines:**
- **Assignment Variables**: Use binary variables `vtype=GRB.BINARY`
- **Assignment Constraints**: Each task assigned to exactly one resource
- **Resource Constraints**: Each resource handles at most one task (if applicable)
- **Objective**: Minimize cost or maximize profit
- **Result**: Extract assignment decisions and objective

**Example Structure:**
```python
# Assignment variables: assign[i,j] = 1 if task i assigned to resource j
assign = model.addVars(tasks, resources, vtype=GRB.BINARY, name="assign")

# Each task assigned to exactly one resource
for i in tasks:
    model.addConstr(assign.sum(i, '*') == 1, f"task_{i}_assigned")

# Objective: minimize total cost
model.setObjective(quicksum(cost[i,j] * assign[i,j] for i in tasks for j in resources), GRB.MINIMIZE)
```""",
            
            "general": """
**PROBLEM TYPE: General Optimization**

This is a general optimization problem. Follow these modeling principles:

**General Guidelines:**
- **Identify Variables**: Determine what decisions need to be made
- **Define Variables**: Use `model.addVar()` or `model.addVars()` with appropriate types
- **Set Objective**: Use `model.setObjective()` with the objective function
- **Add Constraints**: Use `model.addConstr()` for each constraint
- **Solve**: Call `model.optimize()`
- **Extract Results**: Use `model.objVal` for objective and `var.X` for variable values

**Common Variable Types:**
- Continuous: `model.addVar(lb=0, name="x")`
- Integer: `model.addVar(vtype=GRB.INTEGER, name="x")`
- Binary: `model.addVar(vtype=GRB.BINARY, name="x")`

**Common Constraint Types:**
- Linear: `model.addConstr(expression, '<=', rhs)`
- Equality: `model.addConstr(expression, '==', rhs)`
- Sum: `model.addConstr(quicksum(vars), '<=', rhs)`"""
        }
        
        return guidance_map.get(problem_type, guidance_map["general"])
    
    def detect_problem_type(self, problem_description: str) -> str:
        """
        Automatically detect problem type based on problem description
        
        Args:
            problem_description: Problem description
            
        Returns:
            Problem type string
        """
        description_lower = problem_description.lower()
        
        # Keyword matching
        if any(keyword in description_lower for keyword in 
               ['linear programming', 'linear program', 'lp']):
            return "linear_programming"
        elif any(keyword in description_lower for keyword in 
                ['integer programming', 'integer program', 'ip', 'binary', 'discrete']):
            return "integer_programming"
        elif any(keyword in description_lower for keyword in 
                ['quadratic programming', 'quadratic program', 'qp']):
            return "quadratic_programming"
        elif any(keyword in description_lower for keyword in 
                ['network flow', 'network', 'flow', 'maximum flow', 'minimum cost flow']):
            return "network_flow"
        elif any(keyword in description_lower for keyword in 
                ['assignment', 'assign', 'matching']):
            return "assignment"
        elif any(keyword in description_lower for keyword in 
                ['traveling salesman', 'tsp', 'route', 'path']):
            return "tsp"
        elif any(keyword in description_lower for keyword in 
                ['knapsack', 'knapsack problem']):
            return "knapsack"
        else:
            return "general"
    
    def create_examples_prompt(self, problem_description: str, 
                             examples: List[Dict[str, str]]) -> str:
        """
        Create prompt with examples
        
        Args:
            problem_description: Problem description
            examples: List of examples, each containing description and code
            
        Returns:
            Prompt string with examples
        """
        examples_text = ""
        for i, example in enumerate(examples, 1):
            examples_text += f"""
Example {i}:
Problem Description: {example['description']}
Python Code:
```python
{example['code']}
```
"""
        
        prompt = f"""You are an operations research expert, skilled at converting natural language descriptions of optimization problems into Python code using Gurobi.

{examples_text}

Please generate complete Python code to solve the following optimization problem based on the problem description.

Requirements:
1. Code must be complete and executable
2. Use Gurobi (gurobipy) as the optimization solver
3. Code should include problem modeling, solving, and result output
4. Assign the final result to variable 'result'
5. If the problem has multiple solutions, output the optimal solution
6. If the problem is infeasible, output None and explain the reason
7. Output ONLY the Python code, no explanations or additional text

Problem Description:
{problem_description}

```python"""
        
        return prompt


def test_prompt_templates():
    """Test prompt template functionality"""
    template = ORPromptTemplate()
    
    # Test problem type detection
    test_problems = [
        "This is a linear programming problem that needs to minimize cost",
        "Assign tasks to employees to minimize total cost",
        "Solve the maximum flow problem in the network",
        "Traveling salesman problem, visit all cities once"
    ]
    
    for problem in test_problems:
        problem_type = template.detect_problem_type(problem)
        print(f"Problem: {problem}")
        print(f"Detected type: {problem_type}")
        print()
    
    # Test prompt generation
    prompt = template.create_prompt(
        "Minimize 2x + 3y, subject to: x + y >= 1, x >= 0, y >= 0",
        "linear_programming"
    )
    print("Generated prompt:")
    print(prompt)


if __name__ == "__main__":
    test_prompt_templates()
