"""Standalone net-zone speed test using helper_table geometry.

This file is meant for GT/ball CSV speed alignment checks. It reads a ball/GT CSV and helper_table JSON, keeps only rightward ball segments inside the near-net box, and exports max speed using 1f / 2f / c2f logic.

<<<<<<< HEAD
Example: python net_zone_speed_base_alpha.py --video_file C0086.MP4 --ball_csv C0086_gt.csv --helper_table_json C0086_helper_table.json --save_dir output --base_alpha 0.15 --save_debug_video
CSV-only mode: python net_zone_speed_base_alpha.py --ball_csv C0086_gt.csv --helper_table_json C0086_helper_table.json --save_dir output -base_alpha 0.15
=======
Example: python net_zone_speed_base_alpha.py --video_file C0086.MP4 --ball_csv C0086_gt.csv --helper_table_json C0086_helper_table.json --save_dir output --base_alpha 0.25 --save_debug_video
CSV-only mode: python net_zone_speed_base_alpha.py --ball_csv C0086_gt.csv --helper_table_json C0086_helper_table.json --save_dir output --fps 119.88 --frame_w 1920 --frame_h 1080 --base_alpha 0.25
>>>>>>> origin/master
"""

import argparse
import math
import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from helper_table import BOX_EDGES, BOX_HEIGHT, NEAR_NET_DIST, NearNetRegion, load_table_corners


TABLE_W = 274.0
TABLE_H = 152.5
MAX_SPEED_KMH = 115.0


def load_ball_csv(ball_csv: str) -> pd.DataFrame:
    df = pd.read_csv(ball_csv)
    frame_col = next((c for c in ["Frame", "frame"] if c in df.columns), None)
    x_col = next((c for c in ["X", "x", "ball_x", "cx"] if c in df.columns), None)
    y_col = next((c for c in ["Y", "y", "ball_y", "cy"] if c in df.columns), None)
    vis_col = next((c for c in ["Visibility", "visibility", "vis", "valid"] if c in df.columns), None)

    if frame_col is None:
        raise ValueError("CSV missing required column: Frame or frame")
    if x_col is None or y_col is None:
        raise ValueError("CSV missing required x/y columns, e.g. X,Y or x,y")

    out = pd.DataFrame()
    out["Frame"] = df[frame_col].astype(int)
    out["X"] = pd.to_numeric(df[x_col], errors="coerce")
    out["Y"] = pd.to_numeric(df[y_col], errors="coerce")

    if vis_col is not None:
        out["Visibility"] = pd.to_numeric(df[vis_col], errors="coerce").fillna(0).astype(int)
    else:
        out["Visibility"] = ((out["X"] > 0) & (out["Y"] > 0)).astype(int)

    return out.sort_values("Frame").reset_index(drop=True)


def read_video_info(video_file: Optional[str], fps: float, frame_w: int, frame_h: int) -> Tuple[float, int, int]:
    if video_file is None:
        return float(fps), int(frame_w), int(frame_h)

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_file}")

    v_fps = float(cap.get(cv2.CAP_PROP_FPS))
    v_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    v_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return v_fps if v_fps > 0 else float(fps), v_w if v_w > 0 else int(frame_w), v_h if v_h > 0 else int(frame_h)


def is_valid_point(row) -> bool:
    return int(row["Visibility"]) == 1 and float(row["X"]) > 0 and float(row["Y"]) > 0


def calc_step(x1, y1, x2, y2) -> float:
    return math.hypot(float(x2) - float(x1), float(y2) - float(y1))


def helper_corners_to_table_corners(corners_lf_rf_rb_lb) -> np.ndarray:
    # helper_table order: LF, RF, RB, LB. Return LB, RB, RF, LF.
    lf, rf, rb, lb = np.asarray(corners_lf_rf_rb_lb, dtype=np.float32)
    return np.array([lb, rb, rf, lf], dtype=np.float32)


def build_geometry_from_helper_table(helper_table_json: str, frame_w: int, frame_h: int, near_dist: float = NEAR_NET_DIST, box_height: float = BOX_HEIGHT) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(helper_table_json):
        raise FileNotFoundError(f"helper_table_json not found: {helper_table_json}")

    corners_lf_rf_rb_lb = load_table_corners(helper_table_json)
    region = NearNetRegion(image_corners=corners_lf_rf_rb_lb, frame_shape=(int(frame_h), int(frame_w), 3), near_dist=near_dist, box_height=box_height)
    table_corners = helper_corners_to_table_corners(corners_lf_rf_rb_lb)
    net_zone_points = np.asarray(region.get_box_vertices_2d(as_int=False), dtype=np.float32)

    if table_corners.shape != (4, 2):
        raise ValueError(f"table_corners must have shape (4, 2), got {table_corners.shape}")
    if net_zone_points.shape != (8, 2):
        raise ValueError(f"helper_table net box must have shape (8, 2), got {net_zone_points.shape}")

    return table_corners, net_zone_points


def point_in_net_zone(px: float, py: float, net_zone_points: np.ndarray) -> bool:
    pts = np.asarray(net_zone_points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape != (8, 2):
        return False

    # Same visual 6-point outer contour as stroke_zone_analysis.py: N1 -> N4 -> N3 -> N7 -> N6 -> N5
    poly = pts[[0, 3, 2, 6, 5, 4]].astype(np.float32)
    return cv2.pointPolygonTest(poly, (float(px), float(py)), False) >= 0


def compute_table_geometry_metrics(table_corners: np.ndarray, frame_w: int, frame_h: int, base_alpha: float) -> Dict:
    p1, p2, p3, p4 = np.asarray(table_corners, dtype=np.float32)

    top_px = calc_step(p1[0], p1[1], p2[0], p2[1])
    bottom_px = calc_step(p4[0], p4[1], p3[0], p3[1])
    left_px = calc_step(p1[0], p1[1], p4[0], p4[1])
    right_px = calc_step(p2[0], p2[1], p3[0], p3[1])

    top_sx = TABLE_W / top_px
    bottom_sx = TABLE_W / bottom_px
    left_sy = TABLE_H / left_px
    right_sy = TABLE_H / right_px

<<<<<<< HEAD
    #sx = float((top_sx + bottom_sx)/2)
    #sy = float((left_sy + right_sy)/2)
=======
>>>>>>> origin/master
    sx = float(max(top_sx, bottom_sx))
    sy = float(max(left_sy, right_sy))
    scale = float(sx + float(base_alpha) * (sy - sx))

    area = float(abs(cv2.contourArea(np.asarray(table_corners, dtype=np.float32))))
    frame_area = float(max(1, int(frame_w) * int(frame_h)))
    long_edge_diff = abs(top_px - bottom_px) / max(top_px, bottom_px)
    short_edge_diff = abs(left_px - right_px) / max(left_px, right_px)

    return {
        "table_area_px2": area,
        "table_area_frame_ratio": float(area / frame_area),
        "top_px": float(top_px),
        "bottom_px": float(bottom_px),
        "left_px": float(left_px),
        "right_px": float(right_px),
        "sx_cm_per_px": sx,
        "sy_cm_per_px": sy,
        "sx_sy_ratio": float(sx / sy) if sy > 0 else 1.0,
        "top_bottom_ratio": float(top_px / bottom_px) if bottom_px > 0 else 1.0,
        "left_right_ratio": float(left_px / right_px) if right_px > 0 else 1.0,
        "long_edge_diff": float(long_edge_diff),
        "short_edge_diff": float(short_edge_diff),
        "base_alpha": float(base_alpha),
        "scale_alpha_used": float(base_alpha),
        "scale_cm_per_px": scale,
    }


def calc_segment_speed_basic_kmh(x1, y1, x2, y2, fps, sx, sy, dt_frames, base_alpha: float) -> float:
    scale = sx + float(base_alpha) * (sy - sx)
    dx_cm = (float(x2) - float(x1)) * scale
    dy_cm = (float(y2) - float(y1)) * scale
<<<<<<< HEAD
    #dx_cm = (float(x2) - float(x1)) * sx
    #dy_cm = (float(y2) - float(y1)) * sy
=======
>>>>>>> origin/master
    v_cm_s = math.hypot(dx_cm, dy_cm) / (float(dt_frames) / float(fps))
    return float(v_cm_s * 0.036)


def make_speed_segment(df: pd.DataFrame, i: int, j: int, fps: float, sx: float, sy: float, dt_frames: int, base_alpha: float):
    if i < 0 or j >= len(df):
        return None

    r1 = df.iloc[i]
    r2 = df.iloc[j]
    f1 = int(r1["Frame"])
    f2 = int(r2["Frame"])

    if f2 - f1 != dt_frames:
        return None
    if not is_valid_point(r1) or not is_valid_point(r2):
        return None

    x1 = float(r1["X"])
    y1 = float(r1["Y"])
    x2 = float(r2["X"])
    y2 = float(r2["Y"])

    # Only keep rightward speed segments. In this dataset, left-to-right stroke means X increases.
    if x2 <= x1:
        return None

    speed = calc_segment_speed_basic_kmh(x1, y1, x2, y2, fps, sx, sy, dt_frames, base_alpha)
    if not np.isfinite(speed) or speed > MAX_SPEED_KMH:
        return None

    return float(speed), f1, f2


<<<<<<< HEAD
def compute_net_zone_speeds(df: pd.DataFrame, fps: float, table_corners: np.ndarray, net_zone_points: np.ndarray, frame_w: int, frame_h: int, base_alpha: float = 0.15, expand_net_neighbor: bool = True) -> Tuple[pd.DataFrame, Dict]:
=======
def compute_net_zone_speeds(df: pd.DataFrame, fps: float, table_corners: np.ndarray, net_zone_points: np.ndarray, frame_w: int, frame_h: int, base_alpha: float = 0.25, expand_net_neighbor: bool = True) -> Tuple[pd.DataFrame, Dict]:
>>>>>>> origin/master
    metrics = compute_table_geometry_metrics(table_corners, frame_w, frame_h, base_alpha=base_alpha)
    sx = float(metrics["sx_cm_per_px"])
    sy = float(metrics["sy_cm_per_px"])

    rows = []
    for i in range(len(df)):
        row = df.iloc[i]
        frame_id = int(row["Frame"])
        in_net = point_in_net_zone(float(row["X"]), float(row["Y"]), net_zone_points) if is_valid_point(row) else False

        speed_candidates = []
        for speed_type, start_i, end_i, dt_frames in (("1f", i - 1, i, 1), ("2f", i, i + 2, 2), ("c2f", i - 1, i + 1, 2)):
            seg = make_speed_segment(df, start_i, end_i, fps, sx, sy, dt_frames, base_alpha)
            if seg is not None:
                speed_candidates.append((speed_type, *seg))

        speed_map = {speed_type: speed for speed_type, speed, _, _ in speed_candidates}
<<<<<<< HEAD
        # find max in 3 ways
=======
>>>>>>> origin/master
        if speed_candidates:
            best_type, best_speed, best_start, best_end = max(speed_candidates, key=lambda item: item[1])
        else:
            best_type, best_speed, best_start, best_end = "", np.nan, "", ""
<<<<<<< HEAD
        '''
        # avg of 3 ways
        if speed_candidates:
            speeds = [speed for _, speed, _, _ in speed_candidates]

            best_speed = float(np.mean(speeds))

            best_type = "avg"

            best_start = min(start for _, _, start, _ in speed_candidates)
            best_end = max(end for _, _, _, end in speed_candidates)

        else:
            best_type, best_speed, best_start, best_end = "", np.nan, "", ""
        '''
=======

>>>>>>> origin/master
        rows.append({
            "Frame": frame_id,
            "X": float(row["X"]) if pd.notna(row["X"]) else np.nan,
            "Y": float(row["Y"]) if pd.notna(row["Y"]) else np.nan,
            "Visibility": int(row["Visibility"]),
            "in_net": bool(in_net),
            "speed_1f_kmh": speed_map.get("1f", np.nan),
            "speed_2f_kmh": speed_map.get("2f", np.nan),
            "speed_c2f_kmh": speed_map.get("c2f", np.nan),
            "best_speed_kmh": best_speed,
            "best_speed_type": best_type,
            "best_speed_start_frame": best_start,
            "best_speed_end_frame": best_end,
            "use_for_net_max": False,
        })

    detail = pd.DataFrame(rows)
    net_indices = set(detail.index[detail["in_net"]].tolist())

    if expand_net_neighbor:
        expanded = set()
        for idx in net_indices:
            expanded.update(j for j in (idx - 1, idx, idx + 1) if 0 <= j < len(detail))
        net_indices = expanded

    valid_speed_mask = detail["best_speed_kmh"].notna()
    for idx in sorted(net_indices):
        if bool(valid_speed_mask.iloc[idx]):
            detail.at[idx, "use_for_net_max"] = True

    net_detail = detail[detail["use_for_net_max"]].copy()
    if len(net_detail) > 0:
        best_idx = net_detail["best_speed_kmh"].idxmax()
        best_row = detail.loc[best_idx]
        summary = {
            "net_zone_max_speed_kmh": best_row["best_speed_kmh"],
            "net_zone_max_speed_type": best_row["best_speed_type"],
            "net_zone_max_speed_start_frame": best_row["best_speed_start_frame"],
            "net_zone_max_speed_end_frame": best_row["best_speed_end_frame"],
            "net_zone_max_speed_1f_kmh": net_detail["speed_1f_kmh"].max(skipna=True),
            "net_zone_max_speed_2f_kmh": net_detail["speed_2f_kmh"].max(skipna=True),
            "net_zone_max_speed_c2f_kmh": net_detail["speed_c2f_kmh"].max(skipna=True),
            "net_zone_candidate_frames": int(len(net_detail)),
        }
    else:
        summary = {
            "net_zone_max_speed_kmh": "",
            "net_zone_max_speed_type": "",
            "net_zone_max_speed_start_frame": "",
            "net_zone_max_speed_end_frame": "",
            "net_zone_max_speed_1f_kmh": "",
            "net_zone_max_speed_2f_kmh": "",
            "net_zone_max_speed_c2f_kmh": "",
            "net_zone_candidate_frames": 0,
        }

    summary.update({"fps": float(fps), "table_width_cm": TABLE_W, "table_height_cm": TABLE_H, "max_speed_cap_kmh": MAX_SPEED_KMH})
    summary.update(metrics)

    for i, (x, y) in enumerate(table_corners, start=1):
        summary[f"table_p{i}_x"] = float(x)
        summary[f"table_p{i}_y"] = float(y)
    for i, (x, y) in enumerate(net_zone_points, start=1):
        summary[f"net_p{i}_x"] = float(x)
        summary[f"net_p{i}_y"] = float(y)

    return detail, summary


def draw_helper_box(frame, box_pts, color=(0, 255, 255), thickness=2, fill_alpha=0.12):
    pts = np.asarray(box_pts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape != (8, 2):
        return

    pts_i = pts.astype(np.int32)
    if fill_alpha and fill_alpha > 0:
        overlay = frame.copy()
        hull = cv2.convexHull(pts_i)
        cv2.fillPoly(overlay, [hull], color)
        cv2.addWeighted(overlay, fill_alpha, frame, 1 - fill_alpha, 0, frame)

    for i, j in BOX_EDGES:
        cv2.line(frame, tuple(pts_i[i]), tuple(pts_i[j]), color, thickness, cv2.LINE_AA)

    for idx, p in enumerate(pts_i):
        cv2.circle(frame, tuple(p), 4, color, -1)
        cv2.putText(frame, f"N{idx + 1}", (int(p[0]) + 5, int(p[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def save_debug_video(video_file: str, out_video: str, detail: pd.DataFrame, table_corners: np.ndarray, net_zone_points: np.ndarray):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_file}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    if not writer.isOpened():
        cap.release()
        print(f"[WARN] Cannot open writer: {out_video}")
        return

    table_np = np.asarray(table_corners, dtype=np.int32)
    frame_to_row = {int(row["Frame"]): idx for idx, row in detail.iterrows()}
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        cv2.polylines(frame, [table_np], isClosed=True, color=(255, 0, 0), thickness=2)
        draw_helper_box(frame, net_zone_points, color=(0, 255, 255), thickness=2, fill_alpha=0.12)

        if frame_idx in frame_to_row:
            r = frame_to_row[frame_idx]
            row = detail.iloc[r]
            x = row["X"]
            y = row["Y"]
            if int(row["Visibility"]) == 1 and pd.notna(x) and pd.notna(y) and x > 0 and y > 0:
                color = (0, 255, 0) if bool(row["use_for_net_max"]) else ((0, 200, 255) if bool(row["in_net"]) else (0, 0, 255))
                cv2.circle(frame, (int(x), int(y)), 5, color, -1)
                text = f"best={row['best_speed_kmh']:.2f} {row['best_speed_type']}" if pd.notna(row["best_speed_kmh"]) else "best=nan"
                cv2.putText(frame, f"Frame: {frame_idx}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, text, (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"in_net={bool(row['in_net'])} use={bool(row['use_for_net_max'])}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_file", type=str, default=None, help="optional; used for fps/frame size/debug video")
    parser.add_argument("--ball_csv", type=str, required=True, help="GT or ball CSV")
    parser.add_argument("--helper_table_json", type=str, required=True, help="output JSON from helper_table.py")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--fps", type=float, default=120.0, help="used only when video_file is omitted or video fps cannot be read")
    parser.add_argument("--frame_w", type=int, default=1920, help="used in CSV-only mode")
    parser.add_argument("--frame_h", type=int, default=1080, help="used in CSV-only mode")
    parser.add_argument("--near_dist", type=float, default=NEAR_NET_DIST)
    parser.add_argument("--box_height", type=float, default=BOX_HEIGHT)
<<<<<<< HEAD
    parser.add_argument("--base_alpha", type=float, default=0.15, help="calibrated alpha for scale = sx + alpha * (sy - sx)")
=======
    parser.add_argument("--base_alpha", type=float, default=0.25, help="calibrated alpha for scale = sx + alpha * (sy - sx)")
>>>>>>> origin/master
    parser.add_argument("--scale_alpha", type=float, default=None, help="alias of --base_alpha; kept for old commands")
    parser.add_argument("--no_expand_net_neighbor", action="store_true", help="do not include +/-1 neighboring rows around in-net frames")
    parser.add_argument("--save_debug_video", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    fps, frame_w, frame_h = read_video_info(args.video_file, args.fps, args.frame_w, args.frame_h)
    df = load_ball_csv(args.ball_csv)
    table_corners, net_zone_points = build_geometry_from_helper_table(args.helper_table_json, frame_w=frame_w, frame_h=frame_h, near_dist=args.near_dist, box_height=args.box_height)

    base_alpha = float(args.scale_alpha) if args.scale_alpha is not None else float(args.base_alpha)

    detail, summary = compute_net_zone_speeds(df=df, fps=fps, table_corners=table_corners, net_zone_points=net_zone_points, frame_w=frame_w, frame_h=frame_h, base_alpha=base_alpha, expand_net_neighbor=not args.no_expand_net_neighbor)

    base = os.path.splitext(os.path.basename(args.ball_csv))[0]
    detail_csv = os.path.join(args.save_dir, f"{base}_net_zone_speed_detail.csv")
    summary_csv = os.path.join(args.save_dir, f"{base}_net_zone_speed_summary.csv")

    detail.to_csv(detail_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame([summary]).to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print(f"[INFO] use helper_table geometry: {args.helper_table_json}")
    print(f"[INFO] detail saved : {detail_csv}")
    print(f"[INFO] summary saved: {summary_csv}")
    print(f"[INFO] fps          : {summary['fps']}")
    print(f"[INFO] sx cm/px     : {summary['sx_cm_per_px']}")
    print(f"[INFO] sy cm/px     : {summary['sy_cm_per_px']}")
    print(f"[INFO] base alpha   : {summary['base_alpha']}")
    print(f"[INFO] alpha used   : {summary['scale_alpha_used']}")
    print(f"[INFO] scale cm/px  : {summary['scale_cm_per_px']}")
    print(f"[INFO] net max speed: {summary['net_zone_max_speed_kmh']}")
    print(f"[INFO] net max type : {summary['net_zone_max_speed_type']}")
    print(f"[INFO] net max frame: {summary['net_zone_max_speed_start_frame']} -> {summary['net_zone_max_speed_end_frame']}")

    if args.save_debug_video:
        if args.video_file is None:
            print("[WARN] --save_debug_video requires --video_file. Skip debug video.")
        else:
            out_video = os.path.join(args.save_dir, f"{base}_net_zone_speed_debug.mp4")
            save_debug_video(args.video_file, out_video, detail, table_corners, net_zone_points)
            print(f"[INFO] debug video saved: {out_video}")


if __name__ == "__main__":
<<<<<<< HEAD
    main()
=======
    main()
>>>>>>> origin/master
