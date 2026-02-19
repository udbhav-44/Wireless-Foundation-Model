import argparse
import csv
from collections import defaultdict

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot benchmark results CSV.")
    parser.add_argument("--csv", required=True, help="Path to benchmark results CSV.")
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output image path (e.g., results.png). If omitted, shows window.",
    )
    parser.add_argument(
        "--title",
        default="Benchmark F1 vs Split Ratio",
        help="Plot title.",
    )
    return parser.parse_args()


def load_results(path):
    data = defaultdict(list)
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["model"], row["input_type"])
            data[key].append(
                (
                    float(row["split_ratio"]),
                    float(row["f1_mean"]),
                    float(row["f1_std"]),
                )
            )
    for key in data:
        data[key].sort(key=lambda x: x[0])
    return data


def main():
    args = parse_args()
    data = load_results(args.csv)

    plt.figure(figsize=(8, 5))
    for (model, input_type), rows in data.items():
        ratios = [r for r, _, _ in rows]
        means = [m for _, m, _ in rows]
        stds = [s for _, _, s in rows]
        label = f"{model}-{input_type}"
        plt.plot(ratios, means, marker="o", label=label)
        if any(s > 0 for s in stds):
            upper = [m + s for m, s in zip(means, stds)]
            lower = [m - s for m, s in zip(means, stds)]
            plt.fill_between(ratios, lower, upper, alpha=0.15)

    plt.xlabel("Split ratio")
    plt.ylabel("F1 score")
    plt.title(args.title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=200)
        print(f"Saved plot to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
