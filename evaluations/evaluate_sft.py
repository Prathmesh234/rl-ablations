"""Evaluate the private supervised fine-tuned checkpoint.

This is the PPO step-zero baseline: the same model, tokenizer, 256 held-out
questions, prompt, parser, and greedy generation used immediately before PPO
training. The result measures how much supervised training taught both GSM8K
reasoning and the required final-answer format.
"""

from ._common import EvaluationSpec, evaluate


SPEC = EvaluationSpec(
    label="GSM8K SFT",
    model="ppbhatt500/rl-ablations-sft-2026-07-17",
    model_revision="6fbe054f4c351a109c7f99188a74aca3b72d3a3f",
    output_name="sft.json",
)


if __name__ == "__main__":
    evaluate(SPEC)
