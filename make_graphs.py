"""
make_graphs.py - Visualization suite for Adaptive Precision LLM Inference.
Produces all 6 required project graphs from benchmark results:
1. Accuracy by mode, per backend
2. Average latency by mode
3. Memory by mode
4. Accuracy on easy vs hard questions
5. Fraction of queries routed to INT4 vs INT8 (adaptive mode)
6. Retry rate (adaptive mode)
"""

import glob
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

OUTPUT_DIR = "graphs"
RESULTS_DIR = "results"


def load_all_results():
    csv_files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
    records = []
    for f in csv_files:
        basename = os.path.splitext(os.path.basename(f))[0]
        # Ignore dummy test CSVs if hardware results exist
        if basename.startswith("dummy_"):
            continue
        parts = basename.split("_", 1)
        if len(parts) != 2:
            continue
        backend, mode = parts[0], parts[1]
        try:
            df = pd.read_csv(f)
            df["backend"] = backend
            df["run_mode"] = mode
            records.append(df)
        except Exception as e:
            print(f"[!] Warning: Could not read {f}: {e}")
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def plot_graphs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_all_results()
    if df.empty:
        print("[!] No benchmark result files found to plot.")
        return

    # Style settings
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.titlesize": 14,
    })

    backends = sorted(df["backend"].unique())
    modes = ["static_int4", "static_int8", "adaptive"]
    available_modes = [m for m in modes if m in df["run_mode"].unique()]

    # 1. Accuracy by mode, per backend
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(backends))
    width = 0.25
    colors = ["#3498db", "#2ecc71", "#e74c3c"]

    for i, m in enumerate(available_modes):
        sub = df[df["run_mode"] == m]
        accs = [
            sub[sub["backend"] == b]["correct"].mean() * 100
            if len(sub[sub["backend"] == b]) > 0 else 0
            for b in backends
        ]
        offset = (i - len(available_modes) / 2 + 0.5) * width
        rects = ax.bar(x + offset, accs, width, label=m.replace("_", " ").title(), color=colors[i % len(colors)], alpha=0.9)
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f"{height:.1f}%",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("1. Accuracy by Mode across Hardware Backends")
    ax.set_xticks(x)
    ax.set_xticklabels([b.upper() for b in backends], fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "1_accuracy_by_mode.png"), dpi=300)
    plt.close(fig)
    print(" -> Saved 1_accuracy_by_mode.png")

    # 2. Average Latency by mode
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(available_modes):
        sub = df[df["run_mode"] == m]
        lats = [
            sub[sub["backend"] == b]["latency_s"].mean()
            if len(sub[sub["backend"] == b]) > 0 else 0
            for b in backends
        ]
        offset = (i - len(available_modes) / 2 + 0.5) * width
        rects = ax.bar(x + offset, lats, width, label=m.replace("_", " ").title(), color=colors[i % len(colors)], alpha=0.9)
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f"{height:.2f}s",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Average Latency (seconds)")
    ax.set_title("2. Average Latency by Mode across Hardware Backends")
    ax.set_xticks(x)
    ax.set_xticklabels([b.upper() for b in backends], fontweight="bold")
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "2_latency_by_mode.png"), dpi=300)
    plt.close(fig)
    print(" -> Saved 2_latency_by_mode.png")

    # 3. Memory by mode
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(available_modes):
        sub = df[df["run_mode"] == m]
        mems = [
            sub[sub["backend"] == b]["memory_mb"].mean()
            if len(sub[sub["backend"] == b]) > 0 else 0
            for b in backends
        ]
        offset = (i - len(available_modes) / 2 + 0.5) * width
        rects = ax.bar(x + offset, mems, width, label=m.replace("_", " ").title(), color=colors[i % len(colors)], alpha=0.9)
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f"{int(height)}MB",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Resident Memory (MB)")
    ax.set_title("3. Memory Footprint by Mode across Backends")
    ax.set_xticks(x)
    ax.set_xticklabels([b.upper() for b in backends], fontweight="bold")
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "3_memory_by_mode.png"), dpi=300)
    plt.close(fig)
    print(" -> Saved 3_memory_by_mode.png")

    # 4. Accuracy on easy vs hard questions
    fig, ax = plt.subplots(figsize=(9, 5))
    diff_labels = ["easy", "hard"]
    x_sub = np.arange(len(diff_labels))
    w = 0.8 / len(backends)
    c_list = ["#9b59b6", "#e67e22", "#1abc9c", "#34495e"]

    for idx, b in enumerate(backends):
        sub_b = df[df["backend"] == b]
        vals = [
            sub_b[sub_b["difficulty"] == diff]["correct"].mean() * 100
            if len(sub_b[sub_b["difficulty"] == diff]) > 0 else 0
            for diff in diff_labels
        ]
        offset = (idx - len(backends) / 2 + 0.5) * w
        rects = ax.bar(x_sub + offset, vals, w, label=b.upper(), color=c_list[idx % len(c_list)])
        for rect in rects:
            h = rect.get_height()
            if h > 0:
                ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("4. Accuracy Breakdown: Easy vs Hard Queries")
    ax.set_xticks(x_sub)
    ax.set_xticklabels(["Easy (TriviaQA)", "Hard (GSM8K)"], fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(loc="lower left")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "4_easy_vs_hard_accuracy.png"), dpi=300)
    plt.close(fig)
    print(" -> Saved 4_easy_vs_hard_accuracy.png")

    # 5. Fraction of queries routed to INT4 vs INT8 (adaptive mode)
    adaptive_df = df[df["run_mode"] == "adaptive"]
    if not adaptive_df.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        adaptive_backends = sorted(adaptive_df["backend"].unique())
        x_ad = np.arange(len(adaptive_backends))
        int4_pcts = []
        int8_pcts = []
        for b in adaptive_backends:
            b_df = adaptive_df[adaptive_df["backend"] == b]
            total = len(b_df)
            int4_cnt = (b_df["precision_used"] == "int4").sum()
            int8_cnt = (b_df["precision_used"] == "int8").sum()
            int4_pcts.append((int4_cnt / total) * 100 if total > 0 else 0)
            int8_pcts.append((int8_cnt / total) * 100 if total > 0 else 0)

        ax.bar(x_ad, int4_pcts, 0.45, label="INT4 Precision", color="#3498db")
        ax.bar(x_ad, int8_pcts, 0.45, bottom=int4_pcts, label="INT8 Precision", color="#2ecc71")
        for i, (p4, p8) in enumerate(zip(int4_pcts, int8_pcts)):
            if p4 > 5:
                ax.text(i, p4 / 2, f"{p4:.1f}%", ha="center", va="center", color="white", fontweight="bold")
            if p8 > 5:
                ax.text(i, p4 + p8 / 2, f"{p8:.1f}%", ha="center", va="center", color="white", fontweight="bold")

        ax.set_ylabel("Routing Share (%)")
        ax.set_title("5. Adaptive Routing Distribution: INT4 vs INT8")
        ax.set_xticks(x_ad)
        ax.set_xticklabels([b.upper() for b in adaptive_backends], fontweight="bold")
        ax.set_ylim(0, 105)
        ax.legend(loc="upper right")
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "5_routing_share.png"), dpi=300)
        plt.close(fig)
        print(" -> Saved 5_routing_share.png")

        # 6. Retry rate (adaptive mode)
        fig, ax = plt.subplots(figsize=(7, 5))
        retry_pcts = [
            (adaptive_df[adaptive_df["backend"] == b]["retried"].mean() * 100)
            for b in adaptive_backends
        ]
        rects = ax.bar(x_ad, retry_pcts, 0.4, color="#e67e22")
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylabel("Retry Rate (%)")
        ax.set_title("6. Adaptive Quality Feedback Loop Retry Rate")
        ax.set_xticks(x_ad)
        ax.set_xticklabels([b.upper() for b in adaptive_backends], fontweight="bold")
        ax.set_ylim(0, max(25, max(retry_pcts) + 10 if retry_pcts else 25))
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "6_retry_rate.png"), dpi=300)
        plt.close(fig)
        print(" -> Saved 6_retry_rate.png")

    print(f"\n[PASS] All available graphs successfully generated in '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    plot_graphs()
