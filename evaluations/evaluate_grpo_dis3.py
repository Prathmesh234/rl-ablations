"""Evaluate the stronger-reward GRPO-DIS 3 checkpoint."""

from ._common import EvaluationSpec, evaluate


SPEC = EvaluationSpec(
    label="GRPO-DIS 3 step 16",
    model="checkpoints/grpo-dis-3",
    model_revision=None,
    output_name="grpo_dis3.json",
    local_model=True,
)


if __name__ == "__main__":
    evaluate(SPEC)
