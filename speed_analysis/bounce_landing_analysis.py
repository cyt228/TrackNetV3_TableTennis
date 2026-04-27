"""
bounce_landing_analysis.py
==========================
落點分析：把每個 stroke 的 bounce_frame 球座標
透過四個桌角做透視變換，轉換成桌面 cm 座標，
畫出 6×3 格落點熱力圖。

需要的檔案（放在同一個資料夾）：
  - strokes.csv          : 含 stroke_id, bounce_frame, table_p1~p4 的 CSV
  - ball_trajectory.csv  : 含 Frame, Visibility, X, Y 的 CSV

輸出：
  - landing_heatmap.png  : 6×3 格熱力圖
  - landing_zones.png    : 每球落點散佈圖
  - zone_stats.csv       : 每格命中次數統計
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import os
import sys

# ─────────────────────────────────────────────
# 0. 參數設定
# ─────────────────────────────────────────────
TABLE_W   = 274.0   # cm，標準桌球桌長度
TABLE_H   = 152.5   # cm，標準桌球桌寬度
GRID_COLS = 6       # 橫向格數
GRID_ROWS = 3       # 縱向格數
DPI       = 150

STROKES_CSV    = "data/pred_result_NO7/044/C0091_predict_stroke_zone.csv"
TRAJECTORY_CSV = "data/pred_result_NO7/044/C0091_ball.csv"
OUT_DIR        = "runs/"   # 輸出資料夾，可改成其他路徑

# ─────────────────────────────────────────────
# 1. 讀取資料
# ─────────────────────────────────────────────
def load_data():
    if not os.path.exists(STROKES_CSV):
        sys.exit(f"[錯誤] 找不到 {STROKES_CSV}")
    if not os.path.exists(TRAJECTORY_CSV):
        sys.exit(f"[錯誤] 找不到 {TRAJECTORY_CSV}")

    strokes = pd.read_csv(STROKES_CSV)
    traj    = pd.read_csv(TRAJECTORY_CSV)

    # 統一欄位名稱（去除空白）
    strokes.columns = strokes.columns.str.strip()
    traj.columns    = traj.columns.str.strip()

    print(f"[載入] strokes: {len(strokes)} 列，trajectory: {len(traj)} 列")
    return strokes, traj

# ─────────────────────────────────────────────
# 2. 對每個 stroke 做透視變換
# ─────────────────────────────────────────────
def pixel_to_cm(ball_px, ball_py, table_corners_px):
    """
    ball_px, ball_py   : 球的原始像素座標
    table_corners_px   : 4×2 array，桌面四角像素座標
                         順序：左上、右上、右下、左下（順時針）

    回傳 (x_cm, y_cm)，原點在桌面左上角
    """
    # 目標桌面（cm 座標）
    dst_corners = np.array([
        [0,       0      ],
        [TABLE_W, 0      ],
        [TABLE_W, TABLE_H],
        [0,       TABLE_H],
    ], dtype=np.float32)

    src_corners = table_corners_px.astype(np.float32)
    H, _ = cv2.findHomography(src_corners, dst_corners)

    pt = np.array([[[ball_px, ball_py]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(pt, H)
    return float(dst[0, 0, 0]), float(dst[0, 0, 1])


def get_table_corners(row):
    """
    從 stroke row 取出四個桌角像素座標。
    CSV 欄位：table_p1_x, table_p1_y, table_p2_x, table_p2_y,
               table_p3_x, table_p3_y, table_p4_x, table_p4_y
    預設對應順序：p1=左上, p2=右上, p3=右下, p4=左下
    （請依照你實際標記順序調整）
    """
    corners = np.array([
        [row["table_p1_x"], row["table_p1_y"]],
        [row["table_p2_x"], row["table_p2_y"]],
        [row["table_p3_x"], row["table_p3_y"]],
        [row["table_p4_x"], row["table_p4_y"]],
    ], dtype=np.float64)
    return corners


def assign_zone(x_cm, y_cm):
    """
    回傳 (col, row) 從 0 開始，代表落在哪個格子
    col: 0~5（左→右），row: 0~2（上→下）
    """
    col_w = TABLE_W / GRID_COLS
    row_h = TABLE_H / GRID_ROWS
    col = int(np.clip(x_cm // col_w, 0, GRID_COLS - 1))
    row = int(np.clip(y_cm // row_h, 0, GRID_ROWS - 1))
    return col, row



def compute_landings(strokes, traj):
   # note 轉成字串，避免 NaN 出錯
    notes = strokes["note"].fillna("").astype(str)

    # 只保留 bounce_frame > 0，且 note 不包含 net_hit 的 stroke
    bounce_strokes = strokes[
        (strokes["bounce_frame"] > 0) &
        (~notes.str.contains("net_hit", na=False))
    ].copy()

    print(f"[分析] 有 bounce_frame 且非 net_hit 的 stroke：{len(bounce_strokes)} 筆")

    visible = traj[traj["Visibility"] == 1].set_index("Frame")
    
    records = []
    skipped = 0

    for _, s in bounce_strokes.iterrows():
        bf = int(s["bounce_frame"])

        # 找 bounce_frame 的球座標
        if bf not in visible.index:
            # 嘗試往前後各找 2 frame
            found = False
            for delta in [1, -1, 2, -2]:
                if (bf + delta) in visible.index:
                    bf = bf + delta
                    found = True
                    break
            if not found:
                print(f"  [略過] stroke {int(s['stroke_id'])}：bounce_frame {int(s['bounce_frame'])} 附近沒有可見球")
                skipped += 1
                continue

        ball_row = visible.loc[bf]
        bx, by = float(ball_row["X"]), float(ball_row["Y"])

        # 取桌角
        corners = get_table_corners(s)

        # 透視變換
        try:
            x_cm, y_cm = pixel_to_cm(bx, by, corners)
        except Exception as e:
            print(f"  [略過] stroke {int(s['stroke_id'])}：透視變換失敗 ({e})")
            skipped += 1
            continue

        # 是否在桌面範圍內
        in_table = (-5 <= x_cm <= TABLE_W + 5) and (-5 <= y_cm <= TABLE_H + 5)

        col, row = assign_zone(x_cm, y_cm)

        records.append({
            "stroke_id"  : int(s["stroke_id"]),
            "bounce_frame": int(s["bounce_frame"]),
            "ball_px"    : bx,
            "ball_py"    : by,
            "x_cm"       : round(x_cm, 1),
            "y_cm"       : round(y_cm, 1),
            "zone_col"   : col,
            "zone_row"   : row,
            "zone_label" : f"C{col+1}R{row+1}",
            "in_table"   : in_table,
        })

    print(f"[結果] 成功計算 {len(records)} 筆，略過 {skipped} 筆")
    return pd.DataFrame(records)



def plot_heatmap(df):
    # 建立 zone 計數矩陣（row × col）
    zone_count = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    for _, r in df[df["in_table"]].iterrows():
        zone_count[int(r["zone_row"]), int(r["zone_col"])] += 1

    fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)

    # 自訂顏色：白 → 橘 → 紅
    cmap = LinearSegmentedColormap.from_list(
        "tt", ["#f7f7f7", "#f4a261", "#e63946"])

    im = ax.imshow(zone_count, cmap=cmap, aspect="auto",
                   vmin=0, vmax=max(zone_count.max(), 1),
                   extent=[0, TABLE_W, TABLE_H, 0])

    # 格線
    for c in np.linspace(0, TABLE_W, GRID_COLS + 1):
        ax.axvline(c, color="k", lw=1.2)
    for r in np.linspace(0, TABLE_H, GRID_ROWS + 1):
        ax.axhline(r, color="k", lw=1.2)

    # 中線（網）
    ax.axvline(TABLE_W / 2, color="white", lw=2.5, ls="--", label="net")

    # 每格標數字
    col_w = TABLE_W / GRID_COLS
    row_h = TABLE_H / GRID_ROWS
    for ri in range(GRID_ROWS):
        for ci in range(GRID_COLS):
            cnt = zone_count[ri, ci]
            cx  = ci * col_w + col_w / 2
            cy  = ri * row_h + row_h / 2
            ax.text(cx, cy, str(cnt),
                    ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color="k" if cnt < zone_count.max() * 0.6 else "white")
            ax.text(cx, cy + row_h * 0.3,
                    f"C{ci+1}R{ri+1}",
                    ha="center", va="center",
                    fontsize=8, color="gray")

    plt.colorbar(im, ax=ax, label="ball count")
    ax.set_xlabel("X (cm) →", fontsize=12)
    ax.set_ylabel("Y (cm) ↓", fontsize=12)
    ax.set_title(f"heatmap({GRID_COLS}×{GRID_ROWS} columns)\n"
                 f" {int(df['in_table'].sum())} ball on table ", fontsize=13)
    ax.legend(loc="upper right", fontsize=9)

    out = os.path.join(OUT_DIR, "landing_heatmap.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[儲存] {out}")



def plot_scatter(df):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)

    # 桌面背景
    ax.add_patch(patches.Rectangle(
        (0, 0), TABLE_W, TABLE_H,
        linewidth=2, edgecolor="k", facecolor="#d4edda"))

    # 格線
    for c in np.linspace(0, TABLE_W, GRID_COLS + 1):
        ax.axvline(c, color="k", lw=0.8, alpha=0.5)
    for r in np.linspace(0, TABLE_H, GRID_ROWS + 1):
        ax.axhline(r, color="k", lw=0.8, alpha=0.5)

    # 網（中線）
    ax.axvline(TABLE_W / 2, color="white", lw=3, ls="--", label="net")

    # 在桌面內的落點
    inside  = df[df["in_table"]]
    outside = df[~df["in_table"]]

    ax.scatter(inside["x_cm"], inside["y_cm"],
               c="#e63946", s=80, zorder=5, label=f"inside  ({len(inside)})")
    ax.scatter(outside["x_cm"], outside["y_cm"],
               c="#adb5bd", s=60, marker="x", zorder=4, label=f"outside  ({len(outside)})")

    # 標 stroke_id
    for _, r in inside.iterrows():
        ax.annotate(str(int(r["stroke_id"])),
                    (r["x_cm"], r["y_cm"]),
                    textcoords="offset points", xytext=(4, 4),
                    fontsize=7, color="#333")

    ax.set_xlim(-10, TABLE_W + 10)
    ax.set_ylim(-10, TABLE_H + 10)
    ax.invert_yaxis()
    ax.set_xlabel("X (cm) →", fontsize=12)
    ax.set_ylabel("Y (cm) ↓", fontsize=12)
    ax.set_title("scatter plot", fontsize=13)
    ax.legend(loc="best", fontsize=9)

    out = os.path.join(OUT_DIR, "landing_zones.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[儲存] {out}")



def save_stats(df):
    # 詳細落點
    detail_path = os.path.join(OUT_DIR, "landing_detail.csv")
    df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"[儲存] {detail_path}")

    # 每格統計
    stats = (df[df["in_table"]]
             .groupby("zone_label")
             .size()
             .reset_index(name="count")
             .sort_values("zone_label"))
    stats_path = os.path.join(OUT_DIR, "zone_stats.csv")
    stats.to_csv(stats_path, index=False, encoding="utf-8-sig")
    print(f"[儲存] {stats_path}")

    # 印出摘要
    # print("\n── 區域統計 ──")
    # print(stats.to_string(index=False))
    # print(f"\n桌面外落點：{int((~df['in_table']).sum())} 球")



if __name__ == "__main__":
    strokes, traj = load_data()
    df_land = compute_landings(strokes, traj)

    if df_land.empty:
        print("[警告] 沒有任何落點可以分析，請檢查資料。")
        sys.exit(1)

    plot_heatmap(df_land)
    plot_scatter(df_land)
    save_stats(df_land)

    print("\n完成！輸出檔案：")
    print("  landing_heatmap.png  — 熱力圖")
    print("  landing_zones.png    — 散佈圖")
    print("  landing_detail.csv   — 詳細落點座標")
    print("  zone_stats.csv       — 各格命中統計")