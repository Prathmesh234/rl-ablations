"""Print a comparable summary of all available fixed-protocol evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ._common import (
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_REVISION,
    DEFAULT_EVAL_PROBLEMS,
    DEFAULT_MAX_COMPLETION_TOKENS,
    DEFAULT_MAX_PROMPT_TOKENS,
    DEFAULT_SEED,
    ROOT,
    SFT_REVISION,
    SFT_TOKENIZER,
)


RESULTS = (
    ("Base", "base.json"),
    ("SFT", "sft.json"),
    ("PPO", "ppo.json"),
    ("PPO 2", "ppo2.json"),
    ("GRPO", "grpo.json"),
    ("GRPO-DIS", "grpo_dis.json"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare all available fixed-protocol evaluation results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "evaluations" / "results",
    )
    return parser.parse_args()


def validate_result(path: Path, document: dict) -> None:
    dataset = document.get("dataset", {})
    tokenizer = document.get("tokenizer", {})
    generation = document.get("generation", {})
    summary = document.get("summary", {})
    expected = {
        "dataset.id": (dataset.get("id"), DATASET_ID),
        "dataset.config": (dataset.get("config"), DATASET_CONFIG),
        "dataset.revision": (dataset.get("revision"), DATASET_REVISION),
        "dataset.split": (dataset.get("split"), "test"),
        "dataset.shuffle_seed": (dataset.get("shuffle_seed"), DEFAULT_SEED),
        "tokenizer.id": (tokenizer.get("id"), SFT_TOKENIZER),
        "tokenizer.revision": (tokenizer.get("revision"), SFT_REVISION),
        "generation.do_sample": (generation.get("do_sample"), False),
        "generation.max_prompt_tokens": (
            generation.get("max_prompt_tokens"),
            DEFAULT_MAX_PROMPT_TOKENS,
        ),
        "generation.max_completion_tokens": (
            generation.get("max_completion_tokens"),
            DEFAULT_MAX_COMPLETION_TOKENS,
        ),
        "summary.examples": (summary.get("examples"), DEFAULT_EVAL_PROBLEMS),
    }
    mismatches = [
        f"{name}={actual!r}, expected {wanted!r}"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if mismatches:
        raise ValueError(f"Incompatible result {path}: {'; '.join(mismatches)}")


def main() -> None:
    args = parse_args()
    rows = []
    for label, filename in RESULTS:
        path = args.results_dir / filename
        if not path.exists():
            rows.append((label, None))
            continue
        document = json.loads(path.read_text())
        validate_result(path, document)
        rows.append((label, document["summary"]))

    sft_summary = next(
        (summary for label, summary in rows if label == "SFT"),
        None,
    )
    sft_accuracy = None if sft_summary is None else sft_summary["accuracy"]
    print("| Stage | Correct | Accuracy | Change from SFT |")
    print("|---|---:|---:|---:|")
    for label, summary in rows:
        if summary is None:
            print(f"| {label} | not run | - | - |")
            continue
        change = (
            "-"
            if sft_accuracy is None
            else f"{100 * (summary['accuracy'] - sft_accuracy):+.2f} pp"
        )
        print(
            f"| {label} | {summary['correct']}/{summary['examples']} | "
            f"{100 * summary['accuracy']:.2f}% | {change} |"
        )


if __name__ == "__main__":
    main()
