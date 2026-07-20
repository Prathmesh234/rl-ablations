"""Evaluate the final 256-problem GRPO checkpoint.

The checkpoint is scored on the same fixed GSM8K rows, prompt, tokenizer,
parser, and greedy decoding settings as every existing model evaluation.
"""

from ._common import EvaluationSpec, evaluate


SPEC = EvaluationSpec(
    label="GRPO step 16",
    model="checkpoints/grpo",
    model_revision=None,
    output_name="grpo.json",
    local_model=True,
)


if __name__ == "__main__":
    evaluate(SPEC)
