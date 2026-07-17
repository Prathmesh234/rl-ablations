from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_PATH = "ppbhatt500/rl-ablations-sft-2026-07-17"
DEFAULT_PROMPT = "If a box has 12 pencils and 5 are removed, how many pencils remain?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fine-tuned GSM8K model.")
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path
    hf_token = os.environ.get("HF_TOKEN")
    if model_path == DEFAULT_MODEL_PATH and not hf_token:
        raise RuntimeError("HF_TOKEN is required to load the private SFT checkpoint.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this model.")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        token=hf_token,
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
