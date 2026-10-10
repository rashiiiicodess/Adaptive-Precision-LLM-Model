import argparse
import csv
import math
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_OUTPUT = "results/cuda_perplexity.csv"


def load_quantized_model(precision):
    """Load only one precision at a time."""
    if precision == "int4":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        kwargs = {"quantization_config": quant_config}
    else:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        kwargs = {
            "quantization_config": quant_config,
            "dtype": torch.float16,
        }

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map="cuda",
        **kwargs,
    )
    model.eval()
    model.config.use_cache = False
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Measure WikiText-2 perplexity for a quantized CUDA model."
    )
    parser.add_argument(
        "--precision",
        required=True,
        choices=["int4", "int8"],
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum input context length.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=256,
        help="Step between windows; must not exceed max-length.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional token limit for a quick smoke test.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="CSV path for results.",
    )
    args = parser.parse_args()

    if args.max_length < 2:
        parser.error("--max-length must be at least 2.")
    if not 1 <= args.stride <= args.max_length:
        parser.error("--stride must be between 1 and max-length.")
    if args.max_tokens is not None and args.max_tokens < 2:
        parser.error("--max-tokens must be at least 2.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check your PyTorch environment.")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {MODEL}")
    print(f"Precision: {args.precision}")

    print("\nLoading WikiText-2 test split...")
    dataset = load_dataset(
        "parquet",
        data_files={"test": "data/wikitext/test.parquet"},
        split="test",
    )
    text = "\n\n".join(dataset["text"])

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    token_ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]

    if args.max_tokens is not None:
        token_ids = token_ids[:, :args.max_tokens]

    total_input_tokens = token_ids.shape[1]
    print(f"Input tokens: {total_input_tokens:,}")
    print("Loading quantized model...")

    model = load_quantized_model(args.precision)
    torch.cuda.reset_peak_memory_stats()

    total_nll = 0.0
    total_scored_tokens = 0
    previous_end = 0
    start_time = time.perf_counter()

    print("\nEvaluating sliding windows...")

    with torch.inference_mode():
        for begin in range(0, total_input_tokens, args.stride):
            end = min(begin + args.max_length, total_input_tokens)

            input_ids = token_ids[:, begin:end].to("cuda")
            target_len = end - previous_end

            labels = input_ids.clone()
            if target_len < labels.shape[1]:
                labels[:, :-target_len] = -100

            outputs = model(
                input_ids=input_ids,
                labels=labels,
                use_cache=False,
            )

            # Causal-LM loss shifts labels by one position internally.
            scored = int((labels[:, 1:] != -100).sum().item())

            if scored:
                total_nll += float(outputs.loss.float().item()) * scored
                total_scored_tokens += scored

            previous_end = end

            if end == total_input_tokens:
                break

            if (begin // args.stride) % 50 == 0:
                print(
                    f"Processed through token {end:,} / "
                    f"{total_input_tokens:,}"
                )

    elapsed = time.perf_counter() - start_time

    if total_scored_tokens == 0:
        raise RuntimeError("No tokens were scored. Check the input dataset.")

    average_nll = total_nll / total_scored_tokens
    perplexity = math.exp(average_nll)
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)

    row = {
        "model": MODEL,
        "dataset": "wikitext-2-raw-v1-test",
        "precision": args.precision,
        "max_length": args.max_length,
        "stride": args.stride,
        "input_tokens": total_input_tokens,
        "scored_tokens": total_scored_tokens,
        "average_cross_entropy": average_nll,
        "perplexity": perplexity,
        "evaluation_seconds": elapsed,
        "peak_vram_mb": peak_vram_mb,
        "token_limit": args.max_tokens or "full",
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print("\n===== PERPLEXITY RESULTS =====")
    print(f"Precision:       {args.precision}")
    print(f"Input tokens:    {total_input_tokens:,}")
    print(f"Scored tokens:   {total_scored_tokens:,}")
    print(f"Average loss:    {average_nll:.6f}")
    print(f"Perplexity:      {perplexity:.4f}")
    print(f"Evaluation time: {elapsed:.2f} seconds")
    print(f"Peak GPU memory: {peak_vram_mb:.1f} MiB")
    print(f"Saved to:        {output_path}")

    del model
    del tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
