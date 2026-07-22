# GSM8K training progress

## Evaluation protocol

All models use the same 256 examples:

- Dataset: `openai/gsm8k`, `main`
- Revision: `740312add88f781978c0658806c59bc2815b9866`
- Split: official `test`
- Selection: shuffle with seed `17`, then take the first 256 examples
- Prompt limit: 384 tokens
- Completion limit: 256 tokens
- Decoding: greedy (`do_sample=False`)
- Correctness: the completion must end with `Final answer: <number>` and match
  the GSM8K answer exactly

## Results

| Stage | Correct | Accuracy | Change from SFT |
|---|---:|---:|---:|
| Qwen3-0.6B base, no SFT | 0/256 | 0.00% | -44.53 pp |
| SFT | 114/256 | 44.53% | baseline |
| PPO, final step 32 | 116/256 | 45.31% | +0.78 pp |

The final PPO model answered two more evaluation problems correctly than SFT.
This difference is too small to establish a meaningful improvement on a
256-problem evaluation.

These figures are reproduced by the standalone scripts in `evaluations/` and
their detailed outputs in `evaluations/results/`.

## Experiment inventory

| Ablation | Implementation | Training | Fixed 256-problem evaluation |
|---|---|---|---|
| SFT | complete | complete | 114/256 (44.53%) |
| PPO | complete | complete | 116/256 (45.31%) |
| PPO 2 | complete | not run | not run |
| GRPO | complete | complete | not recorded |
| GRPO-DIS | complete | complete | not recorded |
| PPO 3 reasoning rewards | complete | complete | 118/256 (46.09%) |
| GRPO 2 reasoning rewards | complete | not run | not run |
| GRPO-DIS 2 reasoning rewards | complete | complete | 109/256 (42.58%) |
| PPO 4 stronger rewards/tuning | complete | not run | not run |
| GRPO 3 stronger rewards | complete | not run | not run |
| GRPO-DIS 3 stronger rewards | complete | not run | not run |
| synchronous SAO approximation | complete | not recorded | not run |

The private completed GRPO and GRPO-DIS checkpoints are pinned in `README.md`.
Their standalone evaluators exist, but comparable result JSON files are not
tracked. The checked-in `checkpoints/ppo/metrics.json` belongs to an earlier
256-problem PPO run; the reported 1,024-problem PPO result comes from the final
checkpoint evaluated by `evaluations/evaluate_ppo.py`.

## Interpretation

The base model produced no contract-valid correct answers: 255 of 256
completions did not finish with `Final answer: <number>` within the shared
256-token completion limit, and the remaining formatted answer was incorrect.
Its 0.00% therefore reflects exact-answer and format compliance under the common
evaluation protocol, not necessarily zero latent arithmetic ability.

SFT delivered the major gain by teaching the model the GSM8K reasoning and
answer format. The first PPO run did not materially improve that SFT baseline.

## PPO 4 implementation

`notebooks/01_ppo_2.ipynb` is implemented but has not been run. It keeps PPO
1's 1,024 policy-training problems and 256-example held-out evaluation, while
adding:

1. A separate 128-problem critic warm-start using the same full-GAE lambda
   returns as policy training, matching SAO's token-level GAE formulation.
2. Four PPO policy epochs and two complete critic optimizer steps after every
   policy optimizer step (`K=2`).
3. Actor/critic learning rates of `2e-6`/`5e-6` and KL coefficient `0.02`.
4. A 512-token budget, `0.75` for a substantive reasoning trace, and a concave
   length reward up to `0.50`. Shaping requires a valid final answer.
5. Warm-start and PPO critic traces in `gae.json`, including reward components,
   token rewards, targets, values, advantages, and returns.

## Next phase: GSM8K hill climbing

The next experiments will move from method-only ablations to controlled
hyperparameter tuning and additional reward functions. Every candidate should
start from the same pinned SFT checkpoint and keep the fixed 256-problem greedy
evaluation unchanged so gains are measured as additional correct answers and
percentage-point improvement, not only higher sampled training reward.

Record each run's changed hyperparameters, reward components, seed, training
subset, checkpoint revision, and fixed-protocol score. The immediate objective
is a sequence of reproducible held-out improvements above the 116/256 PPO
result.
