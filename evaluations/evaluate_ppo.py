"""Evaluate the final 1,024-problem PPO checkpoint.

The default points to the completed local step-32 checkpoint. It uses the same
fixed test rows and deterministic decoding as the base and SFT scripts. This
makes the reported difference attributable to PPO weights rather than sampling,
prompt, tokenizer, or evaluation-set changes.
"""

from ._common import EvaluationSpec, evaluate


SPEC = EvaluationSpec(
    label="PPO step 32",
    model="checkpoints/ppo-1024-2026-07-17",
    model_revision=None,
    output_name="ppo.json",
    local_model=True,
)


if __name__ == "__main__":
    evaluate(SPEC)
