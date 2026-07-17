# GSM8K RL ablations

Readable, single-GPU notebooks comparing PPO, GRPO, GRPO+DIS, and a
synchronous approximation of SAO on `Qwen/Qwen3-0.6B`.

## Install

Install `uv`, copy the environment template, and sync the locked project
environment:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd "$(git rev-parse --show-toplevel)"
cp .env.example .env
# Add your HF_TOKEN and WANDB_API_KEY values to .env.
uv sync
uv run --no-sync --env-file .env jupyter lab
```

If the pinned PyTorch range does not match the machine's CUDA runtime, install
the appropriate CUDA build before running `uv sync`, then update the lockfile
for that environment.

`uv run --no-sync --env-file .env` passes the Hugging Face and W&B credentials to
Jupyter without storing them in notebook output or tracked files. `.env` is
gitignored; `.env.example` documents the required variables.

Flash Attention 2 is used through
`attn_implementation="flash_attention_2"`. Flash Attention 3 is not used
because the RTX 6000 Ada is an sm_89 GPU, not Hopper.

## Run order

1. `notebooks/00_sft.ipynb`
2. `notebooks/01_ppo.ipynb`
3. `notebooks/02_grpo.ipynb`
4. `notebooks/03_grpo_dis.ipynb`
5. `notebooks/04_sao.ipynb`

Every notebook is self-contained. Its first code cell contains one named
variable per hyperparameter; there is no shared helper module or configuration
dictionary.

Test the saved SFT checkpoint with the Hugging Face inference script:

```bash
uv run --no-sync python inference/run.py "A store has 18 apples and sells 7. How many remain?"
```

The pinned `openai/gsm8k` `main` revision provides 7,473 training and 1,319
test problems. SFT uses 2,048 shuffled training problems for two epochs. The RL
ablations use 256 shuffled training problems and 64 shuffled examples from the
official test split.

## Reward

The model must end with `Final answer: <number>`.

- correct numeric answer: `1`
- incorrect numeric answer: `0`
- missing/malformed final answer: `-1`

Training solutions may contain reasoning, but reward depends only on exact
final numeric correctness.

## Expected wall-clock time

Conservative planning ranges for one RTX 6000 Ada 48 GB:

| Notebook | Expected time |
|---|---:|
| SFT | 30-90 minutes |
| PPO | 1-3 hours |
| GRPO | 2-6 hours |
| GRPO+DIS | 2-6 hours |
| SAO | 1-4 hours |

GRPO and GRPO+DIS each generate 2,048 training completions
(`256 problems x 8`). Their default evaluation cadence is every four optimizer
steps, producing four complete 64-problem evaluations. PPO has sixteen
16-problem training updates and evaluates after every eight updates; SAO has
two 128-problem updates and evaluates after each update. SFT performs 256
optimizer steps and evaluates once per epoch. Generation length and the first
model download dominate variance.

## Method details

PPO uses a separate `AutoModelForSequenceClassification(num_labels=1)` critic,
full token-level GAE, four PPO epochs, symmetric `0.2` policy clipping, and
value clipping. It writes per-update rewards, generation time, total rollout
time, seconds per problem, elapsed time, and ETA to
`checkpoints/ppo/metrics.json`. The logical PPO batch remains 16 problems;
gradient-bearing forwards are accumulated in chunks of four to fit GPU memory.

GRPO uses a real TRL 1.8 `GRPOTrainer`, eight generations per problem, classic
`loss_type="grpo"`, within-group reward standardization, and no KL penalty.
KL against the frozen SFT model is telemetry only.

GRPO+DIS keeps the same GRPO rollout and advantage pipeline but replaces
ordinary clipping with strict token masking. A token contributes only when
`1 - epsilon_low < current/rollout < 1 + epsilon_high`; the coding-paper
bounds `epsilon_low=0.8`, `epsilon_high=3.0` are retained.

SAO is explicitly synchronous here. It uses one rollout per problem, two
critic updates before each policy update, critic warm-starting, and a critic
whose attention parameters are frozen while dense MLP and scalar-head
parameters train.

All runs log to W&B project `swe-rl-ablations` with run names
`sft-base`, `ppo`, `grpo`, `grpo-dis`, and `sao`.

## References

- [GSM8K dataset](https://huggingface.co/datasets/openai/gsm8k)
- [TRL 1.8 source](https://github.com/huggingface/trl/tree/v1.8.0)
- [SAO paper, arXiv:2607.07508](https://arxiv.org/abs/2607.07508)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
