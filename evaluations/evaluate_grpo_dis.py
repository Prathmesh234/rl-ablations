"""Evaluate the final 256-problem GRPO-DIS checkpoint.

The checkpoint is scored on the same fixed GSM8K rows, prompt, tokenizer,
parser, and greedy decoding settings as every existing model evaluation.
"""

from ._common import EvaluationSpec, evaluate


SPEC = EvaluationSpec(
    label="GRPO-DIS step 16",
    model="checkpoints/grpo-dis",
    model_revision=None,
    output_name="grpo_dis.json",
    local_model=True,
)


if __name__ == "__main__":
    evaluate(SPEC)
