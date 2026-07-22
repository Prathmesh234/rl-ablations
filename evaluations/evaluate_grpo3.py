"""Evaluate the stronger-reward GRPO 3 checkpoint."""

from ._common import EvaluationSpec, evaluate


SPEC = EvaluationSpec(
    label="GRPO 3 step 16",
    model="checkpoints/grpo-3",
    model_revision=None,
    output_name="grpo3.json",
    local_model=True,
)


if __name__ == "__main__":
    evaluate(SPEC)
