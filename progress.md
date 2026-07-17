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

## Interpretation

The base model produced no contract-valid correct answers: 255 of 256
completions did not finish with `Final answer: <number>` within the shared
256-token completion limit, and the remaining formatted answer was incorrect.
Its 0.00% therefore reflects exact-answer and format compliance under the common
evaluation protocol, not necessarily zero latent arithmetic ability.

SFT delivered the major gain by teaching the model the GSM8K reasoning and
answer format. The first PPO run did not materially improve that SFT baseline.

## PPO 2 implementation

`notebooks/01_ppo_2.ipynb` is implemented but has not been run. It keeps PPO
1's 1,024 policy-training problems, two policy epochs, and 256-example held-out
evaluation, while adding:

1. A separate 64-problem critic warm-start using Monte Carlo token returns.
2. Two complete critic optimizer steps after every policy optimizer step
   (`K=2`), yielding four critic steps per rollout batch.
3. Warm-start and PPO critic traces in `gae.json`, including token rewards,
   targets, initial values, final values, advantages, and returns.
