"""Evaluate the reasoning-shaped PPO 3 checkpoint.

PPO 3 adds reasoning-trace rewards and uses GAE lambda returns for both critic
warm-start and policy training. This script deliberately keeps the original
256-token fixed evaluation protocol so only the learned model weights differ.
"""

from ._common import EvaluationSpec, evaluate


SPEC = EvaluationSpec(
    label="PPO 3 step 32",
    model="checkpoints/ppo-3",
    model_revision=None,
    output_name="ppo3.json",
    local_model=True,
)


if __name__ == "__main__":
    evaluate(SPEC)
