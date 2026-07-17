from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "openai/gsm8k"
DATASET_CONFIG = "main"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
SFT_TOKENIZER = "ppbhatt500/rl-ablations-sft-2026-07-17"
SFT_REVISION = "6fbe054f4c351a109c7f99188a74aca3b72d3a3f"
DEFAULT_SEED = 17
DEFAULT_EVAL_PROBLEMS = 256
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_PROMPT_TOKENS = 384
DEFAULT_MAX_COMPLETION_TOKENS = 256


@dataclass(frozen=True)
class EvaluationSpec:
    label: str
    model: str
    model_revision: str | None
    output_name: str
    local_model: bool = False


def gold_number(answer: str) -> str:
    """Extract the canonical numeric answer from a GSM8K reference solution."""
    match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*$", answer)
    if not match:
        raise ValueError(f"Malformed GSM8K answer: {answer!r}")
    return match.group(1).replace(",", "")


def predicted_number(completion: str) -> str | None:
    """Apply the same strict final-answer contract used during PPO evaluation."""
    match = re.search(
        r"Final answer:\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*$",
        completion.strip(),
        re.IGNORECASE,
    )
    return None if not match else match.group(1).replace(",", "")


def exact_reward(completion: str, answer: str) -> float:
    prediction = predicted_number(completion)
    if prediction is None:
        return -1.0
    return 1.0 if float(prediction) == float(gold_number(answer)) else 0.0


def prompt_text(tokenizer, question: str) -> str:
    """Use one shared prompt and chat template so only model weights differ."""
    messages = [
        {
            "role": "user",
            "content": (
                "Solve this grade-school math problem. Show your reasoning, then "
                "end with `Final answer: <number>`.\n\n" + question
            ),
        }
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def parse_args(spec: EvaluationSpec) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Evaluate {spec.label} on the fixed held-out GSM8K comparison set."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evaluations" / "results" / spec.output_name,
    )
    parser.add_argument("--eval-problems", type=int, default=DEFAULT_EVAL_PROBLEMS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=DEFAULT_MAX_PROMPT_TOKENS,
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=DEFAULT_MAX_COMPLETION_TOKENS,
    )
    return parser.parse_args()


def evaluate(spec: EvaluationSpec) -> None:
    args = parse_args(spec)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for these evaluations.")

    # All three evaluations use the tokenizer saved with SFT. This fixes the
    # chat template, special tokens, and prompt tokenization across models.
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required for the private SFT tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(
        SFT_TOKENIZER,
        revision=SFT_REVISION,
        token=hf_token,
        use_fast=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # The exact comparison set is the first N examples after deterministically
    # shuffling the official GSM8K test split with seed 17.
    rows = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split="test",
        revision=DATASET_REVISION,
    ).shuffle(seed=DEFAULT_SEED).select(range(args.eval_problems))
    prompts = [prompt_text(tokenizer, question) for question in rows["question"]]
    prompt_lengths = [
        len(input_ids)
        for input_ids in tokenizer(prompts, truncation=False)["input_ids"]
    ]
    if max(prompt_lengths) > args.max_prompt_tokens:
        raise RuntimeError(
            f"Prompt truncation required: max={max(prompt_lengths)}, "
            f"limit={args.max_prompt_tokens}"
        )

    # Hosted models are pinned to immutable revisions. The PPO checkpoint is a
    # local path because it has not been uploaded to Hugging Face.
    if spec.local_model:
        model_source = (ROOT / spec.model).resolve()
        if not (model_source / "config.json").exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_source}")
        model_kwargs = {}
    else:
        model_source = spec.model
        model_kwargs = {"revision": spec.model_revision, "token": hf_token}
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        **model_kwargs,
    ).to("cuda")
    model.eval()

    results = []
    started_at = time.perf_counter()
    for start in range(0, len(rows), args.batch_size):
        stop = min(start + args.batch_size, len(rows))
        batch_prompts = prompts[start:stop]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(model.device)

        # Greedy decoding removes sampling noise. The output slice starts after
        # the padded prompt width, matching ExactAnswerPPOTrainer.exact_evaluate.
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_completion_tokens,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        completions = tokenizer.batch_decode(
            generated[:, encoded.input_ids.shape[1] :],
            skip_special_tokens=True,
        )
        for index, completion in enumerate(completions, start=start):
            answer = rows[index]["answer"]
            prediction = predicted_number(completion)
            reward = exact_reward(completion, answer)
            results.append(
                {
                    "index": index,
                    "question": rows[index]["question"],
                    "gold": gold_number(answer),
                    "prediction": prediction,
                    "reward": reward,
                    "completion": completion,
                }
            )
        correct = sum(result["reward"] == 1.0 for result in results)
        print(f"evaluated={stop}/{len(rows)} correct={correct}", flush=True)

    correct = sum(result["reward"] == 1.0 for result in results)
    incorrect = sum(result["reward"] == 0.0 for result in results)
    malformed = sum(result["reward"] == -1.0 for result in results)
    document = {
        "evaluation": asdict(spec),
        "dataset": {
            "id": DATASET_ID,
            "config": DATASET_CONFIG,
            "revision": DATASET_REVISION,
            "split": "test",
            "shuffle_seed": DEFAULT_SEED,
        },
        "tokenizer": {
            "id": SFT_TOKENIZER,
            "revision": SFT_REVISION,
        },
        "generation": {
            "do_sample": False,
            "batch_size": args.batch_size,
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_completion_tokens": args.max_completion_tokens,
            "prompt_max_observed": max(prompt_lengths),
        },
        "summary": {
            "examples": len(results),
            "correct": correct,
            "incorrect": incorrect,
            "malformed": malformed,
            "accuracy": correct / len(results),
            "elapsed_s": time.perf_counter() - started_at,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(document, indent=2) + "\n")
    temporary_path.replace(args.output)
    print(
        f"{spec.label}: {correct}/{len(results)} "
        f"({100 * document['summary']['accuracy']:.2f}%)"
    )
    print(f"Saved {args.output}")
