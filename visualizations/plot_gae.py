#!/usr/bin/env python3
"""Visualize one PPO update from a ``gae.json`` trace.

The critic's ``old_values`` and ``final_critic.values`` entries are scalar
value estimates for each generated token, not vocabulary logits.  This script
creates a batch-level value heatmap, a detailed per-token GAE dashboard, and a
CSV containing the selected example's complete value/advantage trace.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot token-level critic values and GAE at one PPO step."
    )
    parser.add_argument(
        "gae_path",
        nargs="?",
        type=Path,
        default=Path("checkpoints/ppo-3/gae.json"),
        help="Path to gae.json (default: checkpoints/ppo-3/gae.json).",
    )
    parser.add_argument("--step", type=int, default=1, help="PPO update step to inspect.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--example",
        type=int,
        default=0,
        help="Zero-based example offset within the selected step (default: 0).",
    )
    selection.add_argument(
        "--problem-index",
        type=int,
        help="Select an example by its dataset problem_index instead.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        help="Counterfactual discount factor; defaults to the value stored in gae.json.",
    )
    parser.add_argument(
        "--gae-lambda",
        dest="gae_lambda",
        type=float,
        help="Counterfactual GAE lambda; defaults to the stored value.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="Optional local Hugging Face tokenizer directory for readable token labels.",
    )
    parser.add_argument("--output", type=Path, help="Output PNG path.")
    parser.add_argument("--csv", type=Path, help="Output CSV path.")
    parser.add_argument("--no-csv", action="store_true", help="Do not write the value-list CSV.")
    parser.add_argument("--show", action="store_true", help="Open the figure after saving it.")
    parser.add_argument("--list", action="store_true", help="List available steps/examples and exit.")
    return parser.parse_args()


def load_document(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"GAE trace does not exist: {path}")
    document = json.loads(path.read_text())
    if not document.get("updates"):
        raise ValueError(f"No policy updates were found in {path}")
    return document


def get_update(document: dict[str, Any], step: int) -> dict[str, Any]:
    for update in document["updates"]:
        if update.get("step") == step:
            return update
    available = ", ".join(str(update.get("step")) for update in document["updates"])
    raise ValueError(f"Step {step} was not found. Available steps: {available}")


def get_example(
    update: dict[str, Any], example_offset: int, problem_index: int | None
) -> tuple[int, dict[str, Any]]:
    examples = update.get("examples", [])
    if problem_index is not None:
        for offset, example in enumerate(examples):
            if example.get("problem_index") == problem_index:
                return offset, example
        available = ", ".join(str(example.get("problem_index")) for example in examples)
        raise ValueError(
            f"Problem {problem_index} is not in step {update['step']}. "
            f"Available problem indices: {available}"
        )
    if not 0 <= example_offset < len(examples):
        raise ValueError(
            f"Example offset {example_offset} is outside 0..{len(examples) - 1} "
            f"for step {update['step']}."
        )
    return example_offset, examples[example_offset]


def validate_hyperparameter(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1; received {value}.")


def recompute_trace(example: dict[str, Any], gamma: float, gae_lambda: float) -> dict[str, list[float]]:
    values = [float(value) for value in example["old_values"]]
    rewards = [float(reward) for reward in example["token_rewards"]]
    if len(values) != len(rewards):
        raise ValueError("old_values and token_rewards have different lengths")

    next_values = values[1:] + [0.0]
    discounted_next_values = [gamma * value for value in next_values]
    deltas = [
        reward + discounted_next - value
        for reward, discounted_next, value in zip(rewards, discounted_next_values, values)
    ]
    raw_advantages = [0.0] * len(values)
    running = 0.0
    for index in range(len(values) - 1, -1, -1):
        running = deltas[index] + gamma * gae_lambda * running
        raw_advantages[index] = running
    returns = [advantage + value for advantage, value in zip(raw_advantages, values)]
    return {
        "next_values": next_values,
        "discounted_next_values": discounted_next_values,
        "deltas": deltas,
        "raw_advantages": raw_advantages,
        "returns": returns,
    }


def recompute_batch(
    update: dict[str, Any], gamma: float, gae_lambda: float
) -> list[dict[str, list[float]]]:
    traces = [recompute_trace(example, gamma, gae_lambda) for example in update["examples"]]
    all_advantages = [
        advantage for trace in traces for advantage in trace["raw_advantages"]
    ]
    mean = sum(all_advantages) / len(all_advantages)
    variance = sum((value - mean) ** 2 for value in all_advantages) / len(all_advantages)
    std = math.sqrt(variance)
    for trace in traces:
        trace["normalized_advantages"] = [
            (value - mean) / (std + 1e-8) for value in trace["raw_advantages"]
        ]
    return traces


def decode_tokens(tokenizer_path: Path | None, token_ids: list[int]) -> list[str]:
    if tokenizer_path is None:
        return [str(token_id) for token_id in token_ids]
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Token decoding requires the project's transformers dependency") from error
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    return [str(token) for token in tokenizer.convert_ids_to_tokens(token_ids)]


def compact_token(token: str, width: int = 16) -> str:
    token = token.replace("\n", "\\n").replace("\t", "\\t")
    return token if len(token) <= width else token[: width - 1] + "…"


def write_trace_csv(
    path: Path,
    example: dict[str, Any],
    trace: dict[str, list[float]],
    final_values: list[float],
    token_labels: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "token_position",
        "token_id",
        "token",
        "token_reward",
        "old_critic_value",
        "next_critic_value",
        "discounted_next_value",
        "stored_delta",
        "recomputed_delta",
        "stored_raw_advantage",
        "recomputed_raw_advantage",
        "stored_normalized_advantage",
        "recomputed_normalized_advantage",
        "stored_return",
        "recomputed_return",
        "final_critic_value",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, token_id in enumerate(example["token_ids"]):
            writer.writerow(
                {
                    "token_position": index,
                    "token_id": token_id,
                    "token": token_labels[index],
                    "token_reward": example["token_rewards"][index],
                    "old_critic_value": example["old_values"][index],
                    "next_critic_value": trace["next_values"][index],
                    "discounted_next_value": trace["discounted_next_values"][index],
                    "stored_delta": example["deltas"][index],
                    "recomputed_delta": trace["deltas"][index],
                    "stored_raw_advantage": example["raw_advantages"][index],
                    "recomputed_raw_advantage": trace["raw_advantages"][index],
                    "stored_normalized_advantage": example["normalized_advantages"][index],
                    "recomputed_normalized_advantage": trace["normalized_advantages"][index],
                    "stored_return": example["returns"][index],
                    "recomputed_return": trace["returns"][index],
                    "final_critic_value": final_values[index] if index < len(final_values) else "",
                }
            )


def plot_dashboard(
    document: dict[str, Any],
    update: dict[str, Any],
    example_offset: int,
    example: dict[str, Any],
    trace: dict[str, list[float]],
    gamma: float,
    gae_lambda: float,
    token_labels: list[str],
    output_path: Path,
    show: bool,
) -> None:
    import numpy as np

    matplotlib_cache = Path(tempfile.gettempdir()) / "rl-ablations-matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    examples = update["examples"]
    final_rows = update.get("final_critic", {}).get("values", [])
    final_values = final_rows[example_offset] if example_offset < len(final_rows) else []
    positions = np.arange(len(example["old_values"]))

    figure, axes = plt.subplots(3, 2, figsize=(19, 16), constrained_layout=True)
    figure.suptitle(
        f"PPO token-level GAE — step {update['step']}, problem {example['problem_index']}\n"
        f"γ={gamma:.4g}, λ={gae_lambda:.4g}, reward={example['reward']:.4f} "
        f"(exact={example['exact_reward']:.3g}, think={example['think_reward']:.3g}, "
        f"length={example['length_reward']:.3g})",
        fontsize=16,
        fontweight="bold",
    )
    figure.get_layout_engine().set(rect=(0.0, 0.045, 1.0, 0.95))

    # Batch-wide critic value list.
    max_tokens = max(len(item["old_values"]) for item in examples)
    value_matrix = np.full((len(examples), max_tokens), np.nan)
    for row, item in enumerate(examples):
        values = np.asarray(item["old_values"], dtype=float)
        value_matrix[row, : len(values)] = values
    finite = value_matrix[np.isfinite(value_matrix)]
    limit = max(abs(float(finite.min())), abs(float(finite.max())), 1e-6)
    image = axes[0, 0].imshow(
        value_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )
    axes[0, 0].axhline(example_offset, color="gold", linewidth=2)
    axes[0, 0].set_title("Critic scalar outputs for the complete step (not vocabulary logits)")
    axes[0, 0].set_xlabel("Generated-token position")
    axes[0, 0].set_ylabel("Example offset in rollout batch")
    figure.colorbar(image, ax=axes[0, 0], label="Old critic value V(sₜ)")

    # Selected example: critic before/after and return target.
    axes[0, 1].plot(positions, example["old_values"], label="Old critic V(sₜ)", linewidth=1.5)
    axes[0, 1].plot(positions, trace["returns"], label="GAE return target Aₜ + V(sₜ)", linewidth=2)
    if final_values:
        axes[0, 1].plot(positions[: len(final_values)], final_values, label="Final critic after updates", alpha=0.8)
    axes[0, 1].axhline(0, color="black", linewidth=0.7)
    axes[0, 1].set_title("Selected example: value trace and critic target")
    axes[0, 1].set_xlabel("Generated-token position")
    axes[0, 1].set_ylabel("Scalar value")
    axes[0, 1].legend(fontsize=9)

    # TD residual decomposition.
    axes[1, 0].plot(positions, example["token_rewards"], label="rₜ", linewidth=1.3)
    axes[1, 0].plot(positions, trace["discounted_next_values"], label="γV(sₜ₊₁)", alpha=0.75)
    axes[1, 0].plot(positions, [-value for value in example["old_values"]], label="−V(sₜ)", alpha=0.75)
    axes[1, 0].plot(positions, trace["deltas"], label="δₜ (sum)", color="black", linewidth=1.5)
    axes[1, 0].axhline(0, color="black", linewidth=0.7)
    axes[1, 0].set_title("TD residual: δₜ = rₜ + γV(sₜ₊₁) − V(sₜ)")
    axes[1, 0].set_xlabel("Generated-token position")
    axes[1, 0].set_ylabel("Contribution")
    axes[1, 0].legend(fontsize=9, ncols=2)

    # Raw and globally normalized policy advantages.
    raw_axis = axes[1, 1]
    normalized_axis = raw_axis.twinx()
    raw_axis.plot(positions, example["raw_advantages"], label="Stored raw Aₜ", color="#1f77b4", linewidth=2)
    if gamma != float(document["gamma"]) or gae_lambda != float(document["gae_lambda"]):
        raw_axis.plot(positions, trace["raw_advantages"], label="Counterfactual raw Aₜ", color="#ff7f0e", linewidth=1.5)
    normalized_axis.plot(
        positions,
        trace["normalized_advantages"],
        label="Batch-normalized Aₜ",
        color="#2ca02c",
        alpha=0.8,
    )
    raw_axis.axhline(0, color="black", linewidth=0.7)
    raw_axis.set_title("GAE recursion: Aₜ = δₜ + γλAₜ₊₁")
    raw_axis.set_xlabel("Generated-token position")
    raw_axis.set_ylabel("Raw advantage", color="#1f77b4")
    normalized_axis.set_ylabel("Normalized policy advantage", color="#2ca02c")
    lines = raw_axis.lines[:-1] + normalized_axis.lines
    raw_axis.legend(lines, [line.get_label() for line in lines], fontsize=9)

    # Effective GAE weighting of future residuals.
    offsets = np.arange(min(len(positions), 80))
    weights = (gamma * gae_lambda) ** offsets
    axes[2, 0].plot(offsets, weights, color="#9467bd", linewidth=2)
    axes[2, 0].fill_between(offsets, weights, alpha=0.2, color="#9467bd")
    axes[2, 0].set_ylim(bottom=0)
    axes[2, 0].set_title(f"Future TD-residual weight (γλ)ᵏ; γλ={gamma * gae_lambda:.4f}")
    axes[2, 0].set_xlabel("Future-token offset k")
    axes[2, 0].set_ylabel("Weight applied to δₜ₊ₖ")

    # Sensitivity of advantage magnitude to gamma and lambda.
    lambda_grid = np.linspace(0.0, 1.0, 101)
    gamma_grid = sorted({0.9, 0.95, 0.99, 1.0, round(gamma, 6)})
    for candidate_gamma in gamma_grid:
        magnitudes = [
            np.mean(np.abs(recompute_trace(example, candidate_gamma, candidate_lambda)["raw_advantages"]))
            for candidate_lambda in lambda_grid
        ]
        emphasis = 2.5 if math.isclose(candidate_gamma, gamma) else 1.2
        axes[2, 1].plot(lambda_grid, magnitudes, label=f"γ={candidate_gamma:g}", linewidth=emphasis)
    axes[2, 1].axvline(gae_lambda, color="black", linestyle="--", linewidth=1, label=f"selected λ={gae_lambda:g}")
    axes[2, 1].set_title("Hyperparameter sensitivity on the selected completion")
    axes[2, 1].set_xlabel("GAE λ")
    axes[2, 1].set_ylabel("Mean |raw advantage|")
    axes[2, 1].legend(fontsize=9, ncols=2)

    tick_count = min(12, len(positions))
    if tick_count:
        tick_positions = np.unique(np.linspace(0, len(positions) - 1, tick_count, dtype=int))
        tick_labels = [compact_token(token_labels[index]) for index in tick_positions]
        for axis in (axes[0, 1], axes[1, 0], raw_axis):
            axis.set_xticks(tick_positions, tick_labels, rotation=45, ha="right", fontsize=8)

    question = " ".join(str(example.get("question", "")).split())
    completion = " ".join(str(example.get("completion", "")).split())
    figure.text(
        0.01,
        0.002,
        f"Question: {question[:180]}{'…' if len(question) > 180 else ''}\n"
        f"Completion: {completion[:180]}{'…' if len(completion) > 180 else ''}",
        fontsize=8,
        va="bottom",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def main() -> None:
    args = parse_args()
    document = load_document(args.gae_path)

    if args.list:
        for update in document["updates"]:
            indices = [example.get("problem_index") for example in update.get("examples", [])]
            print(
                f"step={update.get('step'):>3} examples={len(indices):>3} "
                f"problem_indices={min(indices) if indices else '-'}..{max(indices) if indices else '-'}"
            )
        return

    update = get_update(document, args.step)
    example_offset, example = get_example(update, args.example, args.problem_index)
    gamma = float(document["gamma"] if args.gamma is None else args.gamma)
    gae_lambda = float(document["gae_lambda"] if args.gae_lambda is None else args.gae_lambda)
    validate_hyperparameter("gamma", gamma)
    validate_hyperparameter("gae_lambda", gae_lambda)

    traces = recompute_batch(update, gamma, gae_lambda)
    trace = traces[example_offset]
    token_labels = decode_tokens(args.tokenizer, example["token_ids"])
    problem_index = example["problem_index"]
    output_path = args.output or Path("runs/gae_visualizations") / (
        f"step_{args.step:03d}_problem_{problem_index}.png"
    )
    csv_path = args.csv or output_path.with_suffix(".csv")
    final_rows = update.get("final_critic", {}).get("values", [])
    final_values = final_rows[example_offset] if example_offset < len(final_rows) else []

    plot_dashboard(
        document,
        update,
        example_offset,
        example,
        trace,
        gamma,
        gae_lambda,
        token_labels,
        output_path,
        args.show,
    )
    if not args.no_csv:
        write_trace_csv(csv_path, example, trace, final_values, token_labels)

    max_delta_error = max(
        abs(stored - recomputed)
        for stored, recomputed in zip(example["deltas"], trace["deltas"])
    )
    print(f"Saved dashboard: {output_path}")
    if not args.no_csv:
        print(f"Saved token value list: {csv_path}")
    print(
        f"Selected step {args.step}, example offset {example_offset}, problem {problem_index}, "
        f"tokens {len(example['token_ids'])}; max stored/recomputed delta error={max_delta_error:.3g}"
    )


if __name__ == "__main__":
    main()
