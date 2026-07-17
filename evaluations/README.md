# Model evaluations

These scripts compare the untouched Qwen base model, the private SFT checkpoint,
and the final PPO 1 and PPO 2 checkpoints under one fixed evaluation protocol.

All four use:

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
```

Detailed per-example results are written to `evaluations/results/`. Those JSON
files are gitignored because they include every generated completion and can be
regenerated from the pinned model and dataset revisions.

The base-model score includes format compliance. A mathematically useful answer
that does not end with the required final-answer line is classified as malformed,
matching the reward contract used for training and evaluation.
