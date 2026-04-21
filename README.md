# PRIMO: Process-rewarded Reasoning LLM for Optimization

<p align="center">
  <img src="fig/PRIMO_framework.png" alt="PRIMO framework" width="100%"/>
</p>

PRIMO turns a natural-language optimization problem into a correct, executable
Gurobi program. It is trained in three stages, shown above:

- **(a) Expert-guided SFT** on OR problems labeled with formulations and code.
- **(b) GRPO with dense, verifiable rewards** — process + code-execution + validation.
- **(c) Scalable inference** — self-exploration, self-correction, and consensus voting.

## Layout

```
release/
├── code/            # prompt templates, sandboxed executor, reward fn, vLLM eval
├── data_cleaned/    # SFT corpus + 8 OM benchmarks (JSON + Parquet splits)
├── docker/          # verl training images (CUDA / Ascend / ROCm / AWS)
├── fig/             # framework figure
└── models/          # drop-in for fine-tuned checkpoints
```

**Benchmarks:** ComplexOR, IndustryOR, LogiOR, Mamo (easy / complex), NL4Opt,
NLP4LP, OptiBench — as both `*.json` and `train/test/all.parquet` splits.

## Quick start

```bash
# 1. Pull a training image (see docker/ for the full matrix)
docker pull verlai/verl:vllm011.latest

# 2. Install runtime deps (or use the image above)
pip install vllm gurobipy numpy pandas pyarrow   # + a Gurobi license

# 3. Evaluate with majority vote (stage c)
cd release/code
python evaluation_with_majority_vote.py \
    --datasets complexor nl4opt nlp4lp optibench \
    --base-data-path ../data_cleaned \
    --pretrained-model /path/to/Qwen2.5-7B-Instruct \
    --trained-model    /path/to/PRIMO-checkpoint \
    --gpu-id 0
```

The VERL-compatible `compute_score` in `code/reward_function.py` fuses the three
reward channels. Key env vars: `VERL_TRAIN_RM_WEIGHT` (PRM weight, default `0.1`),
`VERL_EXECUTION_BONUS`, `VERL_REWARD_TIMEOUT_SECONDS`.

> Note: `reward_function.py` expects a `test_prm_api` module that queries your
> Process Reward Model; plug in your own endpoint before training.

## Citation

```bibtex
@article{primo2026,
  title  = {PRIMO: Process-rewarded Reasoning LLM for Optimization},
  author = {Anonymous},
  year   = {2026}
}
```

Built on [VERL](https://github.com/volcengine/verl) and
[vLLM](https://github.com/vllm-project/vllm).
