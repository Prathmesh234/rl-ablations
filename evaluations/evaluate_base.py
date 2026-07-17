"""Evaluate the untouched Qwen3-0.6B model before any project training.

The evaluator intentionally uses the SFT checkpoint's tokenizer for all model
stages. That keeps the prompt template and token IDs identical, so this script
isolates the effect of using the original base-model weights. Under the shared
256-token completion budget, verbose base-model reasoning may fail the strict
`Final answer:` contract and is counted as malformed, exactly as in PPO.
"""

from ._common import EvaluationSpec, evaluate


SPEC = EvaluationSpec(
    label="Qwen3-0.6B base",
    model="Qwen/Qwen3-0.6B",
    model_revision="c1899de289a04d12100db370d81485cdf75e47ca",
    output_name="base.json",
)


if __name__ == "__main__":
    evaluate(SPEC)
