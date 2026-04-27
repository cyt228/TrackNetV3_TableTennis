import argparse
from pathlib import Path
from typing import Optional, List

import matplotlib.pyplot as plt
import pandas as pd


def collect_compare_csv_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        files = sorted(input_path.rglob("*compare.csv"))
        if not files:
            files = sorted(input_path.rglob("compare.csv"))
        return files

    raise FileNotFoundError(f"Input path not found: {input_path}")


def plot_one_compare_csv(
    csv_path: Path,
    id_col: str,
    speed_base: str,
    out_dir: Optional[str] = None,
) -> Optional[Path]:
    raw_col = f"{speed_base}_raw"
    corr_col = f"{speed_base}_corr"

    df = pd.read_csv(csv_path)

    required_cols = [id_col, raw_col, corr_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[SKIP] {csv_path} missing columns: {missing}")
        return None

    plot_df = df[[id_col, raw_col, corr_col]].copy()
    plot_df[id_col] = pd.to_numeric(plot_df[id_col], errors="coerce")
    plot_df[raw_col] = pd.to_numeric(plot_df[raw_col], errors="coerce")
    plot_df[corr_col] = pd.to_numeric(plot_df[corr_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[id_col, raw_col, corr_col]).sort_values(id_col)

    if plot_df.empty:
        print(f"[SKIP] {csv_path} has no valid compare data")
        return None

    plot_df[id_col] = plot_df[id_col].astype(int)

    raw_mean = plot_df[raw_col].mean()
    raw_max = plot_df[raw_col].max()
    raw_median = plot_df[raw_col].median()

    corr_mean = plot_df[corr_col].mean()
    corr_max = plot_df[corr_col].max()
    corr_median = plot_df[corr_col].median()

    save_dir = csv_path.parent if out_dir is None else Path(out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    safe_speed_base = speed_base.replace("/", "_").replace("\\", "_").replace(" ", "_")
    out_path = save_dir / f"{csv_path.stem}_{safe_speed_base}_raw_vs_corr.png"

    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[5, 1.4])

    ax = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])

    ax.plot(plot_df[id_col], plot_df[raw_col], marker="o", label="raw")
    ax.plot(plot_df[id_col], plot_df[corr_col], marker="o", label="corr")

    for _, row in plot_df.iterrows():
        x = int(row[id_col])
        y_raw = float(row[raw_col])
        y_corr = float(row[corr_col])

        ax.annotate(
            f"{y_raw:.1f}",
            (x, y_raw),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
        ax.annotate(
            f"{y_corr:.1f}",
            (x, y_corr),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=8,
        )

    ax.set_title(f"{speed_base} raw vs corr\n{csv_path.name}")
    ax.set_xlabel(id_col)
    ax.set_ylabel("Speed (km/h)")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(plot_df[id_col].tolist())
    ax.legend()

    stats_text = (
        f"[raw]\n"
        f"Mean: {raw_mean:.2f}\n"
        f"Max: {raw_max:.2f}\n"
        f"Median: {raw_median:.2f}\n\n"
        f"[corr]\n"
        f"Mean: {corr_mean:.2f}\n"
        f"Max: {corr_max:.2f}\n"
        f"Median: {corr_median:.2f}"
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


def main():
    parser = argparse.ArgumentParser(
        description="Plot xxx_raw vs xxx_corr from one compare.csv or all compare.csv files under a folder."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to one compare.csv file or a root folder.",
    )
    parser.add_argument(
        "--id_col",
        type=str,
        default="stroke_id",
        help="ID column name, default: stroke_id",
    )
    parser.add_argument(
        "--speed_base",
        type=str,
        default="net_zone_max_speed_kmh",
        help="Base speed name. Will read <speed_base>_raw and <speed_base>_corr",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Optional output folder. Default: save next to each compare.csv",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    csv_files = collect_compare_csv_files(input_path)

    if not csv_files:
        print(f"[INFO] No compare.csv found under: {input_path}")
        return

    print(f"[INFO] Found {len(csv_files)} compare csv files")
    print(f"[INFO] Compare columns: {args.speed_base}_raw vs {args.speed_base}_corr")

    ok_count = 0
    skip_count = 0

    for csv_path in csv_files:
        result = plot_one_compare_csv(
            csv_path=csv_path,
            id_col=args.id_col,
            speed_base=args.speed_base,
            out_dir=args.out_dir,
        )
        if result is None:
            skip_count += 1
        else:
            ok_count += 1

    print("=" * 60)
    print(f"[DONE] success={ok_count}, skipped={skip_count}, total={len(csv_files)}")


if __name__ == "__main__":
    main()