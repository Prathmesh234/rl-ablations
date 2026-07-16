"""Dependency-free validation of the notebook-first project."""
from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_METRICS = {
    "train/reward",
    "eval/reward",
    "loss/policy",
    "objective/kl_sft",
    "policy/tokens_clipped_or_masked_pct",
    "grad_norm/policy",
    "time/step_s",
    "time/policy_forward_backward_s",
    "gpu/peak_memory_gb",
}


def main() -> None:
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert [path.name for path in notebooks] == [
        "00_sft.ipynb",
        "01_ppo.ipynb",
        "02_grpo.ipynb",
        "03_grpo_dis.ipynb",
        "04_sao.ipynb",
    ]
    assert not (ROOT / "notebooks" / "rl_common.py").exists()

    combined = ""
    for path in notebooks:
        notebook = json.loads(path.read_text())
        assert notebook["nbformat"] == 4
        code = "\n".join(
            cell["source"]
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        ast.parse(code, filename=str(path))
        assert "CONFIG" not in code
        assert "openai/gsm8k" in code
        assert "dataset_revision" in code
        assert "Final answer:" in code
        combined += code

    assert "trainer.train()" in json.loads((ROOT / "notebooks" / "02_grpo.ipynb").read_text())["cells"][-1]["source"]
    assert "trainer.train()" in json.loads((ROOT / "notebooks" / "03_grpo_dis.ipynb").read_text())["cells"][-1]["source"]
    assert "class ExactAnswerPPOTrainer(PPOTrainer)" in combined
    assert "class DISGRPOTrainer(GRPOTrainer)" in combined
    assert "ratio > 1 - self.dis_epsilon_low" in combined
    assert "ratio < 1 + self.dis_epsilon_high" in combined
    assert REQUIRED_METRICS.issubset(set(metric for metric in REQUIRED_METRICS if metric in combined))
    assert "fix_patch" not in combined
    assert "SequenceMatcher" not in combined
    print(f"validated {len(notebooks)} inline GSM8K notebooks")


if __name__ == "__main__":
    main()
