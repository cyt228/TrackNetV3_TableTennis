# python speed_analysis/plot_speed.py --input /path/to/your_stroke_zone.csv
# python speed_analysis/plot_speed.py --input pred_result_NO7

import argparse
from pathlib import Path
from typing import Optional, List

import matplotlib.pyplot as plt
import pandas as pd


def plot_one_csv(
    csv_path: Path,
    speed_col: str,
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    df = pd.read_csv(csv_path)

    required_cols = ["stroke_id", speed_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[SKIP] {csv_path} missing columns: {missing}")
        return None

    plot_df = df[["stroke_id", speed_col]].copy()
    plot_df["stroke_id"] = pd.to_numeric(plot_df["stroke_id"], errors="coerce")
    plot_df[speed_col] = pd.to_numeric(plot_df[speed_col], errors="coerce")
    plot_df = plot_df.dropna(subset=["stroke_id", speed_col]).sort_values("stroke_id")

    if plot_df.empty:
        print(f"[SKIP] {csv_path} has no valid {speed_col} data")
        return None

    plot_df["stroke_id"] = plot_df["stroke_id"].astype(int)

    mean_val = plot_df[speed_col].mean()
    max_val = plot_df[speed_col].max()
    median_val = plot_df[speed_col].median()

    save_dir = csv_path.parent if out_dir is None else Path(out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    safe_speed_col = speed_col.replace("/", "_").replace("\\", "_").replace(" ", "_")
    out_path = save_dir / f"{csv_path.stem}_{safe_speed_col}_line.png"

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[5, 1])

    ax = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])

    ax.plot(
        plot_df["stroke_id"],
        plot_df[speed_col],
        marker="o",
    )
    ax.set_ylim(bottom=0)
    
    for _, row in plot_df.iterrows():
        x = int(row["stroke_id"])
        y = float(row[speed_col])
        ax.annotate(
            f"{y:.1f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )

    ax.set_title(f"{speed_col}\n{csv_path.name}")
    ax.set_xlabel("Stroke ID")
    ax.set_ylabel("Speed (km/h)")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(plot_df["stroke_id"].tolist())

    stats_text = (
        f"Mean: {mean_val:.2f} km/h\n"
        f"Max: {max_val:.2f} km/h\n"
        f"Median: {median_val:.2f} km/h"
    )

    ax_info.axis("off")
    ax_info.text(
        0.05,
        0.95,
        stats_text,
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[OK] {csv_path} -> {out_path}")
    return out_path


def collect_csv_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        return sorted(input_path.rglob("*_stroke_zone.csv"))

    raise FileNotFoundError(f"Input path not found: {input_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot a selected speed column for one CSV or all *_stroke_zone.csv files under a folder."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to one *_stroke_zone.csv file or a root folder.",
    )
    parser.add_argument(
        "--speed",
        type=str,
        default="net_zone_max_speed_kmh",
        help="Speed column name to plot. Example: net_zone_max_speed_kmh / max_speed_kmh / avg_speed_kmh",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Optional output folder. Default: save next to each CSV.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    speed_col = args.speed
    out_dir = args.out_dir

    csv_files = collect_csv_files(input_path)
    if not csv_files:
        print(f"[INFO] No *_stroke_zone.csv found under: {input_path}")
        return

    print(f"[INFO] Found {len(csv_files)} csv files")
    print(f"[INFO] Plot speed column: {speed_col}")

    ok_count = 0
    skip_count = 0

    for csv_path in csv_files:
        result = plot_one_csv(csv_path, speed_col=speed_col, out_dir=out_dir)
        if result is None:
            skip_count += 1
        else:
            ok_count += 1

    print("=" * 60)
    print(f"[DONE] success={ok_count}, skipped={skip_count}, total={len(csv_files)}")


if __name__ == "__main__":
    main()