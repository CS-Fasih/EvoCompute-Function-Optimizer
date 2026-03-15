"""Generate presentation-ready plots for EA aggregated results.

Inputs:
- results/function_1_sphere_aggregated_results.csv
- results/function_2_rosenbrock_aggregated_results.csv

Outputs (PNG, 300 dpi) in plots/:
- function_1_sphere_average_bsf.png
- function_1_sphere_average_avg_fit.png
- function_2_rosenbrock_average_bsf.png
- function_2_rosenbrock_average_avg_fit.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"

INPUT_FILES = [
    RESULTS_DIR / "function_1_sphere_aggregated_results.csv",
    RESULTS_DIR / "function_2_rosenbrock_aggregated_results.csv",
]

# ---------- Plot Style ----------
sns.set_theme(style="whitegrid", context="talk")

COMBO_ORDER = [
    "fps__truncation",
    "rbs__truncation",
    "binary_tournament__truncation",
    "fps__binary_tournament",
    "rbs__binary_tournament",
    "binary_tournament__binary_tournament",
]

LINESTYLE_MAP = {
    "fps__truncation": "-",
    "rbs__truncation": "--",
    "binary_tournament__truncation": "-.",
    "fps__binary_tournament": ":",
    "rbs__binary_tournament": (0, (5, 1)),
    "binary_tournament__binary_tournament": (0, (3, 1, 1, 1)),
}

DISPLAY_LABELS = {
    "fps__truncation": "FPS + Truncation",
    "rbs__truncation": "RBS + Truncation",
    "binary_tournament__truncation": "Binary Tournament + Truncation",
    "fps__binary_tournament": "FPS + Binary Tournament",
    "rbs__binary_tournament": "RBS + Binary Tournament",
    "binary_tournament__binary_tournament": "Binary Tournament + Binary Tournament",
}


def plot_metric(df: pd.DataFrame, function_name: str, metric_col: str, y_label: str, out_file: Path) -> None:
    """Create one line plot for a specific metric."""
    fig, ax = plt.subplots(figsize=(12, 7))

    palette = sns.color_palette("tab10", n_colors=len(COMBO_ORDER))

    for idx, combo in enumerate(COMBO_ORDER):
        sub = df[df["Combination"] == combo].sort_values("Generation")
        if sub.empty:
            continue

        ax.plot(
            sub["Generation"],
            sub[metric_col],
            label=DISPLAY_LABELS.get(combo, combo),
            color=palette[idx],
            linestyle=LINESTYLE_MAP.get(combo, "-"),
            linewidth=2.4,
            alpha=0.95,
        )

    metric_title = "Average BSF" if metric_col == "Average_BSF" else "Average Avg. Fit"
    function_title = function_name.replace("_", " ").title()

    ax.set_title(f"{function_title}: Generation vs {metric_title}", fontsize=16, pad=12)
    ax.set_xlabel("Generation", fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.tick_params(axis="both", labelsize=11)

    legend = ax.legend(
        title="Selection Combination",
        title_fontsize=11,
        fontsize=10,
        loc="best",
        frameon=True,
        ncol=1,
    )
    legend.get_frame().set_alpha(0.95)

    plt.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for csv_path in INPUT_FILES:
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing input file: {csv_path}")

        df = pd.read_csv(csv_path)
        function_name = str(df["Function"].iloc[0])

        # Plot 1: Generation vs Average_BSF
        out_bsf = PLOTS_DIR / f"{function_name}_average_bsf.png"
        plot_metric(
            df=df,
            function_name=function_name,
            metric_col="Average_BSF",
            y_label="Average BSF",
            out_file=out_bsf,
        )

        # Plot 2: Generation vs Average_Avg_Fit
        out_avg = PLOTS_DIR / f"{function_name}_average_avg_fit.png"
        plot_metric(
            df=df,
            function_name=function_name,
            metric_col="Average_Avg_Fit",
            y_label="Average Avg. Fit",
            out_file=out_avg,
        )

        print(f"Saved: {out_bsf}")
        print(f"Saved: {out_avg}")


if __name__ == "__main__":
    main()
