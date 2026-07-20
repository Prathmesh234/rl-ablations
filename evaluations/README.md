# Model evaluations

These scripts compare the untouched Qwen base model, the private SFT checkpoint,
and the final PPO 1, PPO 2, GRPO, and GRPO-DIS checkpoints under one fixed
evaluation protocol.

All six use:

- the same 256 examples from the official GSM8K test split;
- dataset shuffle seed `17`;
- the same SFT tokenizer and chat template;
- greedy decoding with 384 prompt tokens and 256 completion tokens;
- the exact `Final answer: <number>` parser used during PPO training.

Run from the repository root:

```bash
uv run --no-sync --env-file .env python -m evaluations.evaluate_base
uv run --no-sync --env-file .env python -m evaluations.evaluate_sft
uv run --no-sync --env-file .env python -m evaluations.evaluate_ppo
uv run --no-sync --env-file .env python -m evaluations.evaluate_ppo2
uv run --no-sync --env-file .env python -m evaluations.evaluate_grpo
uv run --no-sync --env-file .env python -m evaluations.evaluate_grpo_dis
uv run --no-sync python -m evaluations.compare_results
```

Detailed per-example results are written to `evaluations/results/`. Those JSON
files are gitignored because they include every generated completion and can be
regenerated from the pinned model and dataset revisions.

`compare_results` rejects results produced with a different dataset revision,
selection seed, tokenizer, generation limits, or example count. Missing model
evaluations are shown as `not run`.

The base-model score includes format compliance. A mathematically useful answer
that does not end with the required final-answer line is classified as malformed,
matching the reward contract used for training and evaluation.
