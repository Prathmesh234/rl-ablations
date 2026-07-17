from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "checkpoints" / "sft_base"
DEFAULT_PROMPT = "If a box has 12 pencils and 5 are removed, how many pencils remain?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fine-tuned GSM8K model.")
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(f"No model checkpoint found at {model_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this model.")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda")
    model.eval()

    messages = [
        {
            "role": "user",
            "content": (
                "Solve this grade-school math problem. Show your reasoning, then "
                f"end with `Final answer: <number>`.\n\n{args.prompt}"
            ),
        }
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generation_kwargs = {
        "do_sample": args.temperature > 0,
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if args.temperature > 0:
        generation_kwargs.update(temperature=args.temperature, top_p=args.top_p)

    with torch.inference_mode():
        output = model.generate(**inputs, **generation_kwargs)

    completion = output[0, inputs.input_ids.shape[1] :]
    print(tokenizer.decode(completion, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
