"""Evaluate the stronger-reward PPO 4 checkpoint."""

from ._common import EvaluationSpec, evaluate


SPEC = EvaluationSpec(
    label="PPO 4 step 32",
    model="checkpoints/ppo-4",
    model_revision=None,
    output_name="ppo4.json",
    local_model=True,
)


if __name__ == "__main__":
    evaluate(SPEC)
