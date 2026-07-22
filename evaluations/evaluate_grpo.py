"""Evaluate the reasoning-shaped GRPO 2 checkpoint.

The checkpoint is scored on the same fixed GSM8K rows, prompt, tokenizer,
parser, and greedy decoding settings as every existing model evaluation.
"""

from ._common import EvaluationSpec, evaluate


SPEC = EvaluationSpec(
    label="GRPO 2 step 16",
    model="checkpoints/grpo-2",
    model_revision=None,
    output_name="grpo2.json",
    local_model=True,
)


if __name__ == "__main__":
    evaluate(SPEC)
