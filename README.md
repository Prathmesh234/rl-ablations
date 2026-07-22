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

The completed SFT base is stored privately at
`ppbhatt500/rl-ablations-sft-2026-07-17`. With `HF_TOKEN` in `.env`, the RL
notebooks download it directly and do not require rerunning SFT.

1. Run `notebooks/00_sft.ipynb` only when intentionally retraining the base.
2. Run `notebooks/01_ppo.ipynb`, `01_ppo_2.ipynb`, `02_grpo.ipynb`,
   `03_grpo_dis.ipynb`, or `04_sao.ipynb` against the hosted checkpoint.

Every notebook is self-contained. Its first code cell contains one named
variable per hyperparameter; there is no shared helper module or configuration
dictionary.

Test the saved SFT checkpoint with the Hugging Face inference script:

```bash
uv run --no-sync --env-file .env python inference/run.py "A store has 18 apples and sells 7. How many remain?"
```

Reproduce the same-set base, SFT, and PPO comparison:

```bash
uv run --no-sync --env-file .env python -m evaluations.evaluate_base
uv run --no-sync --env-file .env python -m evaluations.evaluate_sft
uv run --no-sync --env-file .env python -m evaluations.evaluate_ppo
uv run --no-sync --env-file .env python -m evaluations.evaluate_ppo2
uv run --no-sync --env-file .env python -m evaluations.evaluate_grpo
uv run --no-sync --env-file .env python -m evaluations.evaluate_grpo_dis
uv run --no-sync --env-file .env python -m evaluations.evaluate_ppo4
uv run --no-sync --env-file .env python -m evaluations.evaluate_grpo3
uv run --no-sync --env-file .env python -m evaluations.evaluate_grpo_dis3
uv run --no-sync python -m evaluations.compare_results
```

See `evaluations/README.md` for the fixed evaluation protocol.

Visualize the token-level critic values and GAE decomposition from a PPO 3
update (the critic values are scalar value estimates, not vocabulary logits):

```bash
uv run --no-sync python visualizations/plot_gae.py --step 1 --example 0
uv run --no-sync python visualizations/plot_gae.py --step 8 --problem-index 224 --gamma 0.99 --gae-lambda 0.9
```

The script writes a six-panel PNG dashboard and a CSV value list under
`runs/gae_visualizations/`. Use `--list` to show available steps, and pass a
local tokenizer directory with `--tokenizer checkpoints/ppo` to display token
strings instead of token IDs.

The pinned `openai/gsm8k` `main` revision provides 7,473 training and 1,319
test problems. SFT uses 2,048 shuffled training problems for two epochs. PPO
and the current PPO 4 notebook use 1,024 shuffled policy-training problems;
PPO 4 reserves another 128 disjoint official-train problems for critic
warm-starting. The other RL ablations use 256. PPO and PPO 4 evaluate on 256
separate official-test examples; the other RL notebooks evaluate on 64. Both
PPO notebooks explicitly check for overlap between their selected data
partitions.

## Reward

The model must end with `Final answer: <number>`.

- correct numeric answer: `1`
- incorrect numeric answer: `0`
- missing/malformed final answer: `-1`

The latest GRPO, GRPO-DIS, and PPO iterations add two correctness-safe shaping
rewards. Both are zero unless the completion has a parseable final answer:

- substantive `<think>...</think>` trace: `0.75`
- completion length: concave growth up to `0.50` at 512 tokens

The think trace must contain at least 64 tokens, three meaningful reasoning
segments, and explicit arithmetic. The prompts require known quantities,
arithmetic setup, intermediate calculations, and verification inside
`<think>...</think>`, followed by the final answer outside the closing tag.
These runs use a 512-token generation budget for training and notebook
evaluation. The standalone 256-example benchmark intentionally retains its
original prompt and 256-token limit so new checkpoints remain comparable with
the recorded base, SFT, and PPO scores.

## Expected wall-clock time

Conservative planning ranges for one RTX 6000 Ada 48 GB:

| Notebook | Expected time |
|---|---:|
| SFT | 30-90 minutes |
| PPO | 1-3 hours |
| PPO 4 | 2-6 hours |
| GRPO | 2-6 hours |
| GRPO+DIS | 2-6 hours |
| SAO | 1-4 hours |

GRPO and GRPO+DIS each generate 2,048 training completions
(`256 problems x 8`). Their default evaluation cadence is every four optimizer
steps, producing four complete 64-problem evaluations. PPO has thirty-two
32-problem training updates and evaluates 256 held-out problems after every
eight updates; SAO has
two 128-problem updates and evaluates after each update. SFT performs 256
optimizer steps and evaluates once per epoch. Generation length and the first
model download dominate variance.

## Method details

PPO uses a separate `AutoModelForSequenceClassification(num_labels=1)` critic,
full token-level GAE, two PPO epochs, symmetric `0.2` policy clipping, and
value clipping. It writes per-update rewards and timing to
`checkpoints/ppo/metrics.json`, plus complete per-token GAE and critic traces to
`checkpoints/ppo/gae.json`. The logical PPO batch is 32 problems;
gradient-bearing forwards are accumulated in chunks of four to fit GPU memory.
Prompt lengths are measured before model loading and the notebook fails instead
of silently truncating any prompt beyond the 384-token limit.

The next PPO iteration remains implemented in `01_ppo_2.ipynb`. PPO 4 uses
four policy epochs, actor learning rate `2e-6`, critic learning rate `5e-6`,
KL coefficient `0.02`, and 128 critic warm-start problems. Following SAO's
token-level GAE formulation, warm-start and policy training now use the same
fixed-lambda return targets; the separate Monte Carlo target path was removed.
After every policy optimizer step, it runs `K=2` complete critic
forward/backward optimizer steps.

GRPO uses a real TRL 1.8 `GRPOTrainer`, eight generations per problem, classic
`loss_type="grpo"`, within-group standardization of exact-answer and shaping
rewards, and no KL penalty. KL against the frozen SFT model is telemetry only.

GRPO+DIS keeps the same GRPO rollout and advantage pipeline but replaces
ordinary clipping with strict token masking. A token contributes only when
`1 - epsilon_low < current/rollout < 1 + epsilon_high`; the coding-paper
bounds `epsilon_low=0.8`, `epsilon_high=3.0` are retained.

SAO is explicitly synchronous here. It uses one rollout per problem, two
critic updates before each policy update, critic warm-starting, and a critic
whose attention parameters are frozen while dense MLP and scalar-head
parameters train.

All runs log to W&B project `swe-rl-ablations`. The stronger-reward runs are
named `ppo-4`, `grpo-3`, and `grpo-dis-3`, with matching checkpoint
directories. Earlier reasoning-shaped runs retain `ppo-3`, `grpo-2`, and
`grpo-dis-2`.

The completed GRPO checkpoints are stored in private Hugging Face repositories:

- `ppbhatt500/rl-ablations-grpo-2026-07-20`, revision
  `e7f46f9f71edada5fdb19da3da59574671653097`
- `ppbhatt500/rl-ablations-grpo-dis-2026-07-20`, revision
  `672d0d82a07ab9d97ed6e1ac071069ec42bb7c9a`

## References

- [GSM8K dataset](https://huggingface.co/datasets/openai/gsm8k)
- [TRL 1.8 source](https://github.com/huggingface/trl/tree/v1.8.0)
- [SAO paper, arXiv:2607.07508](https://arxiv.org/abs/2607.07508)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
