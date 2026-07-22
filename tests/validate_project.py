"""Dependency-free validation of the notebook-first project."""
from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HOSTED_SFT_CHECKPOINT = "ppbhatt500/rl-ablations-sft-2026-07-17"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
REQUIRED_METRICS = {
    "train/reward",
    "eval/reward",
    "loss/policy",
    "objective/kl_sft",
    "policy/tokens_clipped_or_masked_pct",
    "grad_norm/policy",
    "time/step_s",
    "time/step_per_problem_s",
    "time/generation_s",
    "time/rollout_s",
    "time/rollout_per_problem_s",
    "time/eta_s",
    "time/policy_forward_backward_s",
    "gpu/peak_memory_gb",
}


def cell_source(cell: dict) -> str:
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else source


def main() -> None:
    evaluation_files = [
        ROOT / "evaluations" / "_common.py",
        ROOT / "evaluations" / "evaluate_base.py",
        ROOT / "evaluations" / "evaluate_sft.py",
        ROOT / "evaluations" / "evaluate_ppo.py",
        ROOT / "evaluations" / "evaluate_ppo2.py",
        ROOT / "evaluations" / "evaluate_grpo.py",
        ROOT / "evaluations" / "evaluate_grpo_dis.py",
        ROOT / "evaluations" / "evaluate_ppo4.py",
        ROOT / "evaluations" / "evaluate_grpo3.py",
        ROOT / "evaluations" / "evaluate_grpo_dis3.py",
        ROOT / "evaluations" / "compare_results.py",
    ]
    for path in evaluation_files:
        ast.parse(path.read_text(), filename=str(path))
    evaluation_code = "\n".join(path.read_text() for path in evaluation_files)
    assert GSM8K_REVISION in evaluation_code
    assert "DEFAULT_EVAL_PROBLEMS = 256" in evaluation_code
    assert "DEFAULT_SEED = 17" in evaluation_code
    assert '"do_sample": False' in evaluation_code
    assert "checkpoints/ppo-1024-2026-07-17" in evaluation_code
    assert 'model="checkpoints/ppo-3"' in evaluation_code
    assert 'model="checkpoints/grpo-2"' in evaluation_code
    assert 'model="checkpoints/grpo-dis-2"' in evaluation_code
    assert '("PPO 3", "ppo3.json")' in evaluation_code
    assert '("GRPO 2", "grpo2.json")' in evaluation_code
    assert '("GRPO-DIS 2", "grpo_dis2.json")' in evaluation_code
    assert 'model="checkpoints/ppo-4"' in evaluation_code
    assert 'model="checkpoints/grpo-3"' in evaluation_code
    assert 'model="checkpoints/grpo-dis-3"' in evaluation_code
    assert '("PPO 4", "ppo4.json")' in evaluation_code
    assert '("GRPO 3", "grpo3.json")' in evaluation_code
    assert '("GRPO-DIS 3", "grpo_dis3.json")' in evaluation_code

    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert [path.name for path in notebooks] == [
        "00_sft.ipynb",
        "01_ppo.ipynb",
        "01_ppo_2.ipynb",
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
            cell_source(cell)
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        ast.parse(code, filename=str(path))
        assert "CONFIG" not in code
        assert "openai/gsm8k" in code
        assert "dataset_revision" in code
        assert "Final answer:" in code
        combined += code

    assert "trainer.train()" in cell_source(json.loads((ROOT / "notebooks" / "02_grpo.ipynb").read_text())["cells"][-1])
    assert "trainer.train()" in cell_source(json.loads((ROOT / "notebooks" / "03_grpo_dis.ipynb").read_text())["cells"][-1])
    assert "class ExactAnswerPPOTrainer(PPOTrainer)" in combined
    assert "class ExactAnswerPPO2Trainer(PPOTrainer)" in combined
    assert 'output_dir / "metrics.json"' in combined
    assert 'output_dir / "gae.json"' in combined
    assert '"raw_advantages"' in combined
    assert '"critic_epochs"' in combined
    assert "Prompt truncation required" in combined
    assert "train_problems = 1024" in combined
    assert "eval_problems = 256" in combined
    assert "batch_size = 32" in combined
    assert "num_ppo_epochs = 2" in combined
    assert "max_prompt_tokens = 384" in combined
    assert "Train/eval overlap detected" in combined
    assert '"train_split": "train"' in combined
    assert '"eval_split": "test"' in combined
    assert "class DISGRPOTrainer(GRPOTrainer)" in combined
    assert "ratio > 1 - self.dis_epsilon_low" in combined
    assert "ratio < 1 + self.dis_epsilon_high" in combined
    assert combined.count("generation_batch_size = None") == 2
    assert combined.count(HOSTED_SFT_CHECKPOINT) == 5
    assert HOSTED_SFT_CHECKPOINT in (ROOT / "notebooks" / "00_sft.ipynb").read_text()
    assert combined.count('token=hf_token') >= 12
    assert "Run 00_sft.ipynb first." not in combined
    assert REQUIRED_METRICS.issubset(set(metric for metric in REQUIRED_METRICS if metric in combined))
    assert "fix_patch" not in combined
    assert "SequenceMatcher" not in combined

    for notebook_name, run_name, output_path in [
        ("01_ppo_2.ipynb", "ppo-4", "./checkpoints/ppo-4"),
        ("02_grpo.ipynb", "grpo-3", "./checkpoints/grpo-3"),
        ("03_grpo_dis.ipynb", "grpo-dis-3", "./checkpoints/grpo-dis-3"),
    ]:
        notebook = json.loads((ROOT / "notebooks" / notebook_name).read_text())
        code = "\n".join(
            cell_source(cell)
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        assert "max_completion_tokens = 512" in code
        assert "think_tag_reward_weight = 0.75" in code
        assert "length_reward_weight = 0.50" in code
        assert "length_reward_target_tokens = 512" in code
        assert "length_reward_exponent = 0.35" in code
        assert "minimum_think_tokens = 64" in code
        assert "minimum_reasoning_steps = 3" in code
        assert "def substantive_think_trace(" in code
        assert "def think_tag_reward(" in code
        assert "def length_reward(" in code
        assert '"role": "system"' in code
        assert "You MUST solve the problem through a substantive derivation" in code
        assert "token_count >= minimum_think_tokens" in code
        assert "len(steps) >= minimum_reasoning_steps" in code
        assert "has_math_work" in code
        assert "predicted_number(completion) is not None" in code
        assert f'wandb_run_name = "{run_name}"' in code
        assert f'output_path = "{output_path}"' in code

    ppo2 = json.loads((ROOT / "notebooks" / "01_ppo_2.ipynb").read_text())
    ppo2_code_cells = [
        cell_source(cell)
        for cell in ppo2["cells"]
        if cell["cell_type"] == "code"
    ]
    ppo2_code = "\n".join(ppo2_code_cells)
    assert len(ppo2["cells"]) >= 40
    assert max(map(len, ppo2_code_cells)) < 5_000
    assert "critic_warmstart_problems = 128" in ppo2_code
    assert "train_problems = 1024" in ppo2_code
    assert "eval_problems = 256" in ppo2_code
    assert "critic_updates_per_policy_update = 2" in ppo2_code
    assert "num_ppo_epochs = 4" in ppo2_code
    assert "actor_learning_rate = 2e-6" in ppo2_code
    assert "critic_learning_rate = 5e-6" in ppo2_code
    assert "kl_coefficient = 0.02" in ppo2_code
    assert "Critic warm-start rows overlap PPO policy-training rows." in ppo2_code
    assert "monte_carlo_returns" not in ppo2_code
    assert 'advantages, returns, deltas, raw_advantages = full_gae(' in ppo2_code
    assert "clip_values=False" in ppo2_code
    assert ppo2_code.count(
        "for critic_update_index in range(critic_updates_per_policy_update):"
    ) == 2
    assert '"lambda_returns": return_rows[row_index]' in ppo2_code
    assert '"exact_rewards": exact_rewards' in ppo2_code
    assert '"think_rewards": think_rewards' in ppo2_code
    assert '"length_rewards": length_rewards' in ppo2_code
    assert '"initial_values": initial_value_rows[row_index]' in ppo2_code
    assert 'example["final_values"] = values' in ppo2_code
    print(f"validated {len(notebooks)} inline GSM8K notebooks")


if __name__ == "__main__":
    main()
