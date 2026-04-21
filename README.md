# PRIMO: Process-rewarded Reasoning LLM for Optimization

<p align="center">
  <img src="fig/PRIMO_framework.png" alt="PRIMO framework: (a) expert-guided instruction tuning, (b) GRPO with process rewards, (c) scalable inference." width="100%"/>
</p>

> PRIMO equips open-source LLMs with optimization-modeling expertise and refines them with
> reinforcement learning under verifiable, dense rewards — so that the model can turn a
> natural-language problem description into a correct, transparent, and executable
> optimization program.

## Abstract

Optimization Modeling (OM) supports analytical decision-making across diverse industries
and scenarios. However, its widespread adoption remains impeded by high technical barriers
and manual formulation inefficiencies. While recent advances in Large Language Models
(LLMs) offer a promising avenue for automating OM, most existing approaches fall short of
consistently generating correct, transparent, and executable optimization models for
reliable deployment.

In this paper, we design **PRIMO** (**P**rocess-rewarded **R**eason**i**ng LL**M** for
**O**ptimization). We first equip an open-source LLM with optimization-modeling expertise
via **expert-guided instruction tuning**. Building on this foundation, PRIMO is further
refined through **reinforcement learning with verifiable rewards (RLVR)**, leveraging
broad trajectory exploration to optimize reasoning paths for OM tasks. To mitigate the
sparse-reward issue in RLVR, we design a **dense feedback mechanism** comprising:

- **Process rewards** — for reasoning quality,
- **Code-execution rewards** — for programming validity, and
- **Accuracy rewards** — for solution correctness.

PRIMO also incorporates a **scalable inference mechanism** to continuously improve solution
quality and robustness. Extensive evaluations across multiple OM datasets spanning over 20
domains demonstrate that PRIMO outperforms existing baselines and generates structured
modeling trajectories that reflect expert-level formulation principles. Ablation studies
confirm the contribution of each component to overall performance. Further analyses on
modeling error patterns, reasoning efficiency, and test-time scaling behavior provide a
comprehensive characterization of PRIMO.

## Framework

PRIMO is organized as three stages, illustrated in the figure above.

### (a) Instruction tuning with expert guidance

Optimization tasks are first labeled by OR experts with full mathematical formulations
(decision variables, objective, constraints) and the corresponding executable Gurobi
programs. The base LLM is then **instruction-fine-tuned** on these expert demonstrations
to internalize the canonical modeling protocol — so that before any RL exploration the
model already produces well-structured, step-by-step formulations.

This stage is driven by `data_cleaned/OR_SFT_data.json` (and the refined
`OR_SFT_data_0107.json`), whose instances follow the schema:

1. Decision-variable definition — with explicit types (continuous / integer / binary).
2. Objective-function construction — the mathematical expression being optimized.
3. Constraint construction — each constraint translated into an equation or inequality.
4. Executable Python (Gurobi) code that solves the problem and assigns the optimal
   objective to `result`.

### (b) Group Relative Policy Optimization (GRPO) with process rewards

Given an optimization task, the post-SFT policy generates a **group** of rollouts
`o₁, …, o_G`, each containing reasoning, modeling, and program code. Three verifiable
reward channels are aggregated into an **overall reward** that supervises every rollout:

| Channel              | What it measures                                                  | Source                          |
|----------------------|-------------------------------------------------------------------|---------------------------------|
| Process reward `rᵖ`  | Reasoning quality along the modeling trajectory                    | Process Reward Model (PRM)      |
| Code-execution reward `rᶜ` | Whether the generated program is syntactically & runtime-valid | Sandboxed Python + Gurobi       |
| Validation reward `rᵛ` | Whether the returned objective matches the ground truth           | `ORRewardCalculator` comparison |

GRPO computes per-rollout **advantages** `A₁, …, A_G` from this group of dense rewards and
performs a policy update. The dense signal is what makes RLVR stable on OM tasks — without
the process / execution terms, the accuracy reward alone is near-zero on any
not-yet-correct problem and the gradient collapses.

Concretely in this repo:

- `code/reward_function.py` — VERL-compatible `compute_score(...)` that fuses the three
  channels (rule-based validation reward + execution bonus + PRM score).
- `code/python_interpreter.py` — sandboxed executor with an OR-focused module allow-list
  and the `ORRewardCalculator` that produces `rᶜ` and `rᵛ`.
- `code/prompt_templates.py` — OR-problem prompt templates that force the
  `{"think": ..., "answer": "```python ...```"}` response format on which the reward
  extraction relies.

### (c) Scalable inference with self-exploration, self-correction, and consensus aggregation

At test time PRIMO further improves robustness through a three-step loop:

1. **Self-exploration.** Sample a group of candidate formulations `o₁, …, o_G` at moderate
   temperature — multiple valid formulations for the same problem are common.
2. **Code execution & tool use.** Each candidate is executed in the Gurobi sandbox;
   invalid programs are filtered out.
3. **Self-correction.** For candidates that fail to execute, the model is re-prompted with
   the error trace to repair the program.
4. **Consensus-based aggregation.** Majority vote over the *successfully executed*
   objective values yields the final answer and its interpretation.

This is implemented in `code/evaluation_with_majority_vote.py`
(`MAJORITY_VOTE_REPEAT = 10` by default) and the failure-recovery loop lives in
`code/debug_vllm_evaluation_failed_regen.py`.

## Repository Layout

```
release/
├── fig/
│   └── PRIMO_framework.png          # Framework figure used above
│
├── code/                            # Training / evaluation utilities
│   ├── prompt_templates.py          # OR-problem prompt templates (Gurobi-oriented)
│   ├── python_interpreter.py        # Sandboxed executor + OR reward calculator
│   ├── reward_function.py           # VERL-compatible compute_score (dense reward)
│   ├── evaluation_with_majority_vote.py    # Single-GPU vLLM eval with majority vote
│   └── debug_vllm_evaluation_failed_regen.py  # Regenerate failed executions
│
├── data_cleaned/                    # Cleaned benchmarks + SFT corpus
│   ├── OR_SFT_data.json   / OR_SFT_data_0107.json    # SFT corpus
│   ├── complexor.json     + complexor/{train,test,all,reward_validation}.parquet
│   ├── industryor.json    + industryor/{train,test,...}.parquet
│   ├── logior.json        + logior/...
│   ├── mamo_easy.json     + mamo_easy/...
│   ├── mamo_complex.json  + mamo_complex/...
│   ├── nl4opt.json        + nl4opt/...
│   ├── nlp4lp.json        + nlp4lp/...
│   ├── optibench.json     + optibench/...
│   └── LowAltitudeOR/{train,test,all}.parquet
│
├── docker/                          # Pre-built training images (GPU / Ascend / ROCm / AWS)
│   ├── Dockerfile.stable.vllm
│   ├── Dockerfile.stable.sglang
│   ├── verl0.4-cu124-torch2.6-fa2.7.4/
│   ├── verl0.5-cu126-torch2.7-fa2.7.4/
│   ├── verl0.5-cu126-torch2.7.1-fa2.8.0/
│   ├── verl0.5-preview-cu128-torch2.7.1-fa2.8.0/
│   ├── verl0.6-cu128-torch2.8.0-fa2.7.4/
│   ├── ascend/                      # Dockerfile.ascend_8.{2,3}.rc1_a{2,3}
│   ├── aws/                         # awsefa / sagemaker
│   └── rocm/                        # Dockerfile.rocm{,7,_verl-0.3.0.post1,_verl-0.4.1}
│
└── models/                          # Drop-in directory for released / fine-tuned checkpoints
```

## Datasets

PRIMO is evaluated on a suite of OM benchmarks spanning 20+ application domains. Each
benchmark is released both as a raw JSON list of problems and as pre-split Parquet files
(`train.parquet` / `test.parquet` / `all.parquet`) ready to be consumed by the VERL
training pipeline.

| Benchmark       | Focus                                                          |
|-----------------|----------------------------------------------------------------|
| ComplexOR       | Industrial / multi-constraint OR problems                      |
| IndustryOR      | Cross-industry OR scenarios                                    |
| LogiOR          | Logistics-centric OR problems                                  |
| Mamo (easy)     | Educational LP / IP                                            |
| Mamo (complex)  | Harder LP / IP / QP                                            |
| NL4Opt          | Natural-language → LP benchmark                                |
| NLP4LP          | NL-described LPs                                               |
| OptiBench       | Broad-coverage optimization benchmark                          |
| LowAltitudeOR   | Low-altitude economy / UAV routing OR                          |

## Quick Start

### 1. Environment

Use one of the provided Docker images (recommended) or install dependencies manually:

```bash
# Option A — pre-built image (see release/docker/README.md for the full matrix)
docker pull verlai/verl:vllm011.latest   # or sgl055.latest
docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" \
  --cap-add=SYS_ADMIN -v $(pwd):/workspace/primo --name primo \
  verlai/verl:vllm011.latest sleep infinity
docker start primo && docker exec -it primo bash

# Option B — manual (Python >= 3.10)
pip install vllm gurobipy numpy pandas pyarrow
# Plus a Gurobi license — https://www.gurobi.com/academia/
```

### 2. Evaluation with Majority Vote (stage c)

Single-GPU evaluation of a pretrained and/or fine-tuned model, with 10 samples per
problem and majority voting over successfully executed programs:

```bash
cd release/code
python evaluation_with_majority_vote.py \
    --datasets complexor industryor logior mamo_easy mamo_complex nl4opt nlp4lp optibench \
    --split test \
    --base-data-path ../data_cleaned \
    --pretrained-model /path/to/Qwen2.5-7B-Instruct \
    --trained-model    /path/to/PRIMO-checkpoint \
    --gpu-id 0 \
    --gpu-memory-utilization 0.6 \
    --output-dir ./outputs/evaluation
```

Pass `--trained-checkpoint-root /path/to/verl/.../global_step_XXX` to let the script
assemble a VERL FSDP checkpoint into an HF directory on the fly, or
`--lora-adapter /path/to/adapter` to evaluate a LoRA on top of the base model.

### 3. Regenerating Failed Executions (stage c, self-correction)

For error-pattern analysis and the self-correction step, rebuild prompts from failed
rollouts and re-sample:

```bash
python debug_vllm_evaluation_failed_regen.py \
    --evaluation-json ./outputs/evaluation/<run>.json \
    --pretrained-model /path/to/base-or-trained-model \
    --output-dir ./outputs/regeneration
```

### 4. Wiring the Reward Function into VERL (stage b)

`reward_function.py` exposes the standard VERL entrypoint
`compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)` that fuses
the three reward signals (`rᵖ`, `rᶜ`, `rᵛ`). It auto-detects training vs. validation
context; key environment variables:

| Variable                       | Purpose                                 | Default |
|--------------------------------|-----------------------------------------|---------|
| `VERL_TRAIN_RM_WEIGHT`         | Weight on PRM score during training     | `0.1`   |
| `VERL_VAL_RM_WEIGHT`           | Weight on PRM score during validation   | `0.0`   |
| `VERL_EXECUTION_BONUS`         | Bonus for successful code execution     | `1`     |
| `VERL_REWARD_TIMEOUT_SECONDS`  | Per-rollout sandbox timeout             | `60`    |
| `VERL_DEBUG_REWARD`            | Print the prompt / response / reward    | `0`     |
| `VERL_USE_VAL_MODE`            | Force validation mode                   | `0`     |

`reward_function.py` imports a companion `test_prm_api` module that queries the Process
Reward Model service; supply your own endpoint/client in that module when deploying.

## Reproducing Paper Results

1. **Stage (a) — SFT.** Fine-tune the base model (e.g. `Qwen2.5-7B-Instruct`) on
   `data_cleaned/OR_SFT_data.json` following the expert-guided template.
2. **Stage (b) — GRPO with process rewards.** Run VERL with
   `reward_function.compute_score` as the custom reward on the
   `train.parquet` / `reward_validation.parquet` splits in `data_cleaned/<benchmark>/`.
3. **Stage (c) — Scalable inference.** Evaluate the resulting checkpoint with
   `evaluation_with_majority_vote.py` across the nine benchmarks listed above.

Detailed training configs (learning rates, batch sizes, clip ratios, rollout sizes,
PRM endpoints, etc.) follow the paper's experimental section.

## Citation

If you find PRIMO useful, please cite:

```bibtex
@article{primo2026,
  title   = {PRIMO: Process-rewarded Reasoning LLM for Optimization},
  author  = {Anonymous},
  year    = {2026}
}
```

## License

Released for research purposes. See the parent project's `LICENSE` file and the
individual data-source licenses under `data_cleaned/` for redistribution terms.

## Acknowledgements

PRIMO's training stack builds on [VERL](https://github.com/volcengine/verl) and
[vLLM](https://github.com/vllm-project/vllm). The benchmarks above are adapted from their
original authors; we thank them for making their data available.
