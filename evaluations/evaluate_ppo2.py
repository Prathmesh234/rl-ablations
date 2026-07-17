"""Evaluate the final PPO 2 checkpoint.

PPO 2 adds a 64-problem critic warm-start and two critic optimizer steps after
each policy update. This script keeps the evaluation set, tokenizer, prompt,
parser, and greedy decoding identical to the existing base, SFT, and PPO 1
evaluations so only the learned model weights differ.
"""

from ._common import EvaluationSpec, evaluate


SPEC = EvaluationSpec(
    label="PPO 2 step 32",
    model="checkpoints/ppo-2-2026-07-17",
    model_revision=None,
    output_name="ppo2.json",
    local_model=True,
)


if __name__ == "__main__":
    evaluate(SPEC)
