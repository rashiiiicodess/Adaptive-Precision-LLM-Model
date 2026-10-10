import argparse
import csv
import json
import os
import sys

# Ensure clean UTF-8 stdout encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from evaluate import score_output


def run_benchmark(backend_name: str, mode: str, dataset_path: str = "data/eval.jsonl"):
    # 1. Initialize backend
    print(f"\n[*] Initializing backend: {backend_name} for mode: {mode}...")

    if backend_name == "cuda":
        from backends.backend_cuda import CudaBackend

        backend = CudaBackend()

        if mode == "static_int4":
            backend.load(("int4",))
        elif mode == "static_int8":
            backend.load(("int8",))
        elif mode == "adaptive":
            backend.load(("int4", "int8"))
        else:
            raise ValueError(f"Unsupported mode for CUDA: {mode}")

    elif backend_name == "openvino":
        from backends.backend_openvino import OpenVinoBackend

        backend = OpenVinoBackend()
        backend.load()

    elif backend_name == "dummy":
        from common.dummy_backend import DummyBackend

        backend = DummyBackend()
        backend.load()

    elif backend_name == "cpu":
        from backends.backend_llamacpp import LlamaCppBackend

        backend = LlamaCppBackend()
        if mode == "static_int4":
            backend.load(("int4",))
        elif mode == "static_int8":
            backend.load(("int8",))
        elif mode == "adaptive":
            backend.load(("int4", "int8"))
        else:
            raise ValueError(f"Unsupported mode for CPU: {mode}")

    else:
        raise ValueError(f"Unknown backend: {backend_name}")

    # If adaptive mode, initialize Controller
    ctl = None
    if mode == "adaptive":
        from controller import Controller
        ctl = Controller(backend)

    # 2. Load dataset
    if not os.path.exists(dataset_path):
        print(f"[!] Error: Dataset file not found at {dataset_path}")
        sys.exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    print(f"[*] Loaded {len(questions)} evaluation questions from {dataset_path}")

    # Prepare output results directory and CSV file
    os.makedirs("results", exist_ok=True)
    out_csv = f"results/{backend_name}_{mode}.csv"

    fieldnames = [
        "id",
        "difficulty",
        "mode",
        "precision_used",
        "correct",
        "latency_s",
        "tokens",
        "memory_mb",
        "retried",
    ]

    results = []
    total_correct = 0

    print(f"\n[*] Starting evaluation...")
    for idx, item in enumerate(questions, 1):
        q_id = item["id"]
        prompt = item["question"]
        expected = item["answer"]
        difficulty = item["difficulty"]
        max_tokens = 256 if difficulty == "hard" else 128

        retried = 0
        if mode == "static_int4":
            precision_used = "int4"
            r = backend.generate(prompt, precision="int4", max_new_tokens=max_tokens)
        elif mode == "static_int8":
            precision_used = "int8"
            r = backend.generate(prompt, precision="int8", max_new_tokens=max_tokens)
        elif mode == "adaptive":
            r, level, precision_used, retried_bool = ctl.ask(prompt, max_new_tokens=max_tokens)
            retried = 1 if retried_bool else 0
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose static_int4, static_int8, or adaptive.")

        # Grade the answer
        is_correct = score_output(r.text, expected, difficulty)
        total_correct += is_correct

        row = {
            "id": q_id,
            "difficulty": difficulty,
            "mode": mode,
            "precision_used": precision_used,
            "correct": is_correct,
            "latency_s": round(r.latency_s, 4),
            "tokens": r.tokens,
            "memory_mb": round(r.memory_mb, 2),
            "retried": retried,
        }
        results.append(row)

        mark = "[PASS]" if is_correct else "[FAIL]"
        print(f"[{idx}/{len(questions)}] Q{q_id} ({difficulty}) | {precision_used} | {mark} | {r.latency_s:.2f}s")

    # Write CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    acc = (total_correct / len(questions)) * 100
    avg_lat = sum(r["latency_s"] for r in results) / len(results)
    print("\n" + "=" * 50)
    print(f"Benchmark Complete: {backend_name} ({mode})")
    print(f"Saved to: {out_csv}")
    print(f"Overall Accuracy: {total_correct}/{len(questions)} ({acc:.1f}%)")
    print(f"Average Latency: {avg_lat:.2f}s per query")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive Precision LLM Benchmark")
    parser.add_argument("--backend", default="openvino", choices=["openvino", "dummy", "cpu", "cuda"])
    parser.add_argument("--mode", required=True, choices=["static_int4", "static_int8", "adaptive"])
    parser.add_argument("--dataset", default="data/eval.jsonl")
    args = parser.parse_args()

    run_benchmark(args.backend, args.mode, args.dataset)
