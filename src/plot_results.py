#!/usr/bin/env python3
"""
Heatmap Plotting Script for Needle in a Haystack Benchmark.

Generates a 2D performance heatmap (Context Length vs. Document Depth)
similar to the standard Needle in a Haystack evaluation charts.
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_heatmap(input_path: Path, output_path: Path, provider: str):
    """Reads benchmark JSON and renders a 2D heatmap image."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Results file not found at: {input_path}\n"
            f"Please run the benchmark first:\n"
            f"  python src/needle_haystack.py --provider {provider} --lengths 2000 8000 --depths 0 50 100"
        )

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        raise ValueError(f"No results found in {input_path}")

    model_name = data.get("model", provider)
    df = pd.DataFrame(results)

    # Pivot into a 2D matrix: rows=depth_percent, cols=context_length
    pivot_table = df.pivot(index="depth_percent", columns="context_length", values="score")

    # Ensure depth sorted 0% at the top to 100% at the bottom
    pivot_table = pivot_table.sort_index(ascending=True)

    # Set up matplotlib figure
    plt.figure(figsize=(9, 6), dpi=300)
    sns.set_theme(style="white")

    # Custom colormap: Red (0.0 / Fail) to Green (1.0 / Pass)
    cmap = sns.color_palette("RdYlGn", as_cmap=True)

    ax = sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".1f",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Retrieval Score (1.0 = Pass, 0.0 = Fail)"},
        linewidths=1.5,
        linecolor="white",
        square=False,
    )

    # Styling and Labels
    plt.title(
        f"Needle in a Haystack Retrieval: {provider.upper()} ({model_name})",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Context Length", fontsize=11, fontweight="bold", labelpad=10)
    plt.ylabel("Document Depth (%)", fontsize=11, fontweight="bold", labelpad=10)

    # Format axis tick labels
    ax.set_yticklabels([f"{int(float(y.get_text()))}%" for y in ax.get_yticklabels()], rotation=0)

    # Output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()

    print("=" * 65)
    print(f"[✓] Heatmap generated successfully!")
    print(f"    -> {output_path}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Plot Needle in a Haystack Heatmap")
    parser.add_argument("--provider", type=str, default="gemini", help="Provider name (e.g. gemini)")
    parser.add_argument("--input", type=str, default=None, help="Path to results JSON file")
    parser.add_argument("--output", type=str, default=None, help="Path to save heatmap PNG")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    # Determine input path
    if args.input:
        input_path = Path(args.input)
    else:
        # Check standard location
        candidate = project_root / "results" / f"{args.provider}_results.json"
        if not candidate.exists():
            candidate_alt = project_root / "results" / f"results_{args.provider}.json"
            if candidate_alt.exists():
                candidate = candidate_alt
        input_path = candidate

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = project_root / "results" / f"{args.provider}_heatmap.png"

    plot_heatmap(input_path=input_path, output_path=output_path, provider=args.provider)


if __name__ == "__main__":
    main()
