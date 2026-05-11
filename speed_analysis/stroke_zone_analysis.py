"""Stroke-zone analysis entry point.

This file detects strokes from a TrackNet ball CSV, computes net-zone speed,
merges landing/bounce results, and optionally writes a visualized MP4.
"""

import argparse
import math
import os
import subprocess
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

import bounce_landing_analysis as landing

from stroke_analysis import (
    build_frame_to_stroke_map,
    calc_step,
    detect_strokes_from_runs,
    draw_polygon,
    ensure_dir,
    extract_zone_points,
    is_valid_point,
    safe_int,
)
from helper_table import BOX_EDGES, BOX_HEIGHT, NEAR_NET_DIST, NearNetRegion, load_table_corners


TABLE_W = 274.0
TABLE_H = 152.5
MAX_SPEED_KMH = 115.0

class FFmpegWriter:
    """
    FFmpeg subprocess writer, compatible with cv2.VideoWriter-style write/release.
    Input frame must be BGR uint8 ndarray. Output is mp4.

    codec:
        h264_nvenc: NVIDIA NVENC hardware encoder
        libx264   : CPU encoder fallback
    """
    def __init__(self, save_file, width, height, fps, codec="h264_nvenc", preset=None, cq=23):
        self.save_file = save_file
        self.width = int(width)
        self.height = int(height)
        self.closed = False

        common_in = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", f"{fps}",
            "-i", "-",
            "-an",
        ]
        common_out = [
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            save_file,
        ]

        if codec == "h264_nvenc":
            preset = preset or "p4"
            enc = ["-c:v", "h264_nvenc", "-preset", preset, "-cq", str(cq)]
        elif codec == "libx264":
            preset = preset or "veryfast"
            enc = ["-c:v", "libx264", "-preset", preset, "-crf", str(cq)]
        else:
            raise ValueError(f"Unsupported codec: {codec}")

        cmd = common_in + enc + common_out
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame_bgr):
        if frame_bgr.shape[0] != self.height or frame_bgr.shape[1] != self.width:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))
        if frame_bgr.dtype != np.uint8:
            frame_bgr = frame_bgr.astype(np.uint8)
        if not frame_bgr.flags["C_CONTIGUOUS"]:
            frame_bgr = np.ascontiguousarray(frame_bgr)
        try:
            self.proc.stdin.write(frame_bgr.tobytes())
        except BrokenPipeError:
            err = self.proc.stderr.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"ffmpeg pipe broken:\n{err}")

    def release(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            ret = self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            ret = -1
        if ret != 0:
            err = self.proc.stderr.read().decode("utf-8", errors="ignore")
            print(f"[FFmpegWriter] ffmpeg exit={ret}\n{err}")

    def isOpened(self):
        return not self.closed and self.proc.poll() is None


TABLE_OUT_COLS = [
    "table_p1_x", "table_p1_y",
    "table_p2_x", "table_p2_y",
    "table_p3_x", "table_p3_y",
    "table_p4_x", "table_p4_y",
]
NET_OUT_COLS = [
    # helper_table Near Box 原始 8 個投影點，順序完全保留 helper_table.get_box_vertices_2d():
    # 1~4: bottom face, 5~8: top face
    "net_p1_x", "net_p1_y",
    "net_p2_x", "net_p2_y",
    "net_p3_x", "net_p3_y",
    "net_p4_x", "net_p4_y",
    "net_p5_x", "net_p5_y",
    "net_p6_x", "net_p6_y",
    "net_p7_x", "net_p7_y",
    "net_p8_x", "net_p8_y",
]
ZONE_OUT_COLS = TABLE_OUT_COLS + NET_OUT_COLS


class FrameReader:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.fps <= 0:
            self.fps = 120.0

    def read_frame(self, frame_id: int):
        if frame_id < 0 or frame_id >= self.total_frames:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
        ok, frame = self.cap.read()
        return frame if ok and frame is not None else None

    def release(self):
        self.cap.release()


class CsvFrameInfo:
    def __init__(self, df: pd.DataFrame, fps: float = 120.0, frame_w: int = 1920, frame_h: int = 1080):
        self.video_path = None
        self.cap = None
        self.total_frames = int(df["Frame"].max()) + 1 if "Frame" in df.columns and len(df) > 0 else 0
        self.fps = float(fps)
        self.width = int(frame_w)
        self.height = int(frame_h)

    def read_frame(self, frame_id: int):
        return None

    def release(self):
        pass


def collect_ball_csvs(root_dir: str, csv_suffixes=("_ball.csv", "_bass.csv")) -> List[str]:
    csv_files = []
    for root, _, files in os.walk(root_dir):
        for fname in files:
            if any(fname.endswith(suffix) for suffix in csv_suffixes):
                csv_files.append(os.path.join(root, fname))
    return sorted(csv_files)


def strip_csv_suffix(csv_path: str, csv_suffixes=("_ball.csv", "_bass.csv")) -> str:
    name = os.path.basename(csv_path)
    for suffix in csv_suffixes:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return os.path.splitext(name)[0]


def find_video_for_ball_csv(ball_csv: str, video_root: Optional[str] = None, csv_suffixes=("_ball.csv", "_bass.csv")) -> Optional[str]:
    stem = strip_csv_suffix(ball_csv, csv_suffixes)

    search_dirs = []
    if video_root is not None:
        search_dirs.append(video_root)
    search_dirs.append(os.path.dirname(ball_csv))

    for search_dir in search_dirs:
        if search_dir is None or not os.path.exists(search_dir):
            continue

        direct_candidates = [
            os.path.join(search_dir, f"{stem}.mp4"),
            os.path.join(search_dir, f"{stem}.MP4"),
            os.path.join(search_dir, f"{stem}_predict.mp4"),
        ]
        for video_path in direct_candidates:
            if os.path.exists(video_path):
                return video_path

        for root, _, files in os.walk(search_dir):
            for fname in files:
                lower = fname.lower()
                if not lower.endswith(".mp4"):
                    continue
                name_no_ext = os.path.splitext(fname)[0]
                if name_no_ext == stem or name_no_ext == f"{stem}_predict":
                    return os.path.join(root, fname)

    return None


def polygon_to_result_dict(table_corners, net_zone) -> Dict:
    result = {col: "" for col in ZONE_OUT_COLS}

    if table_corners is not None:
        for i in range(4):
            result[f"table_p{i + 1}_x"] = float(table_corners[i, 0])
            result[f"table_p{i + 1}_y"] = float(table_corners[i, 1])

    if net_zone is not None:
        net_zone = np.asarray(net_zone, dtype=np.float32)
        max_net_points = min(len(net_zone), len(NET_OUT_COLS) // 2)
        for i in range(max_net_points):
            result[f"net_p{i + 1}_x"] = float(net_zone[i, 0])
            result[f"net_p{i + 1}_y"] = float(net_zone[i, 1])

    return result


def helper_corners_to_table_corners(corners_lf_rf_rb_lb):
    """
    helper_table 點選順序:
        LF -> RF -> RB -> LB
        左前 -> 右前 -> 右後 -> 左後

    table/world 座標定義:
        LB 左後 = (0, 0)
        RB 右後 = (274, 0)
        RF 右前 = (274, 152.5)
        LF 左前 = (0, 152.5)

    bounce_landing_analysis 需要:
        table_p1 = (0, 0)
        table_p2 = (274, 0)
        table_p3 = (274, 152.5)
        table_p4 = (0, 152.5)
    """
    lf, rf, rb, lb = corners_lf_rf_rb_lb
    return np.array([lb, rb, rf, lf], dtype=np.float32)


def helper_box_vertices_to_net_points(box_vertices_2d):
    """Return helper_table Near Box 8 projected vertices without resampling or truncation.

    This intentionally keeps the exact output of NearNetRegion.get_box_vertices_2d():
      0~3: bottom face, 4~7: top face.
    No polygon hull conversion, no 6-point sampling, no bounding-box approximation.
    """
    if box_vertices_2d is None:
        return None

    pts = np.asarray(box_vertices_2d, dtype=np.float32)
    if pts.ndim != 2 or pts.shape != (8, 2):
        raise ValueError(f"helper_table box vertices must have shape (8, 2), got {pts.shape}")
    return pts.copy()

def build_zone_info_from_helper_table(corners_lf_rf_rb_lb, frame_shape, near_dist=NEAR_NET_DIST, box_height=BOX_HEIGHT):
    """Build table_p* and raw helper_table Near Box net_p1~net_p8.

    Net zone points are NOT converted to hull, NOT resampled, and NOT compressed to 6 points.
    They are exactly NearNetRegion.get_box_vertices_2d() in helper_table order.
    """
    region = NearNetRegion(
        image_corners=corners_lf_rf_rb_lb,
        frame_shape=frame_shape,
        near_dist=near_dist,
        box_height=box_height,
    )
    table_corners = helper_corners_to_table_corners(corners_lf_rf_rb_lb)
    net_zone = helper_box_vertices_to_net_points(region.get_box_vertices_2d(as_int=False))
    return polygon_to_result_dict(table_corners, net_zone)


def point_in_net_zone(px: float, py: float, net_zone_points) -> bool:
    if net_zone_points is None:
        return False

    pts = np.asarray(net_zone_points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape != (8, 2):
        return False

    # 視覺上的 8 點框外輪廓，用固定六邊形：
    # N1 -> N4 -> N3 -> N7 -> N6 -> N5
    # 1-based: 1, 4, 3, 7, 6, 5
    # 0-based: 0, 3, 2, 6, 5, 4
    poly = pts[[0, 3, 2, 6, 5, 4]].astype(np.float32)

    return cv2.pointPolygonTest(poly, (float(px), float(py)), False) >= 0


def draw_helper_box(frame, box_pts, color=(0, 255, 255), thickness=2, fill_alpha=0.0):
    """Draw helper_table Near Box with the original 8 vertices and BOX_EDGES. No point conversion."""
    if box_pts is None:
        return
    pts = np.asarray(box_pts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape != (8, 2):
        return

    pts_i = pts.astype(np.int32)
    if fill_alpha and fill_alpha > 0:
        overlay = frame.copy()
        # Fill only for readability; edges still use the original 8 helper_table points.
        hull = cv2.convexHull(pts_i)
        cv2.fillPoly(overlay, [hull], color)
        cv2.addWeighted(overlay, fill_alpha, frame, 1 - fill_alpha, 0, frame)

    for i, j in BOX_EDGES:
        cv2.line(frame, tuple(pts_i[i]), tuple(pts_i[j]), color, thickness, cv2.LINE_AA)

    for idx, p in enumerate(pts_i):
        cv2.circle(frame, tuple(p), 4, color, -1)
        cv2.putText(frame, f"N{idx + 1}", (int(p[0]) + 5, int(p[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def compute_fixed_scales(table_corners):
    """Compute fixed cm-per-pixel scales from table corners ordered as LT, RT, RB, LB."""
    p1, p2, p3, p4 = np.asarray(table_corners, dtype=np.float32)

    top_sx = TABLE_W / calc_step(p1[0], p1[1], p2[0], p2[1])
    bottom_sx = TABLE_W / calc_step(p4[0], p4[1], p3[0], p3[1])
    left_sy = TABLE_H / calc_step(p1[0], p1[1], p4[0], p4[1])
    right_sy = TABLE_H / calc_step(p2[0], p2[1], p3[0], p3[1])

    return float(max(top_sx, bottom_sx)), float(max(left_sy, right_sy))


def calc_segment_speed_basic_kmh(x1, y1, x2, y2, fps, sx, sy, dt_frames):
    scale = sx + 0.25 * (sy - sx)
    dx_cm = (x2 - x1) * scale
    dy_cm = (y2 - y1) * scale
    v_cm_s = math.hypot(dx_cm, dy_cm) / (dt_frames / fps)
    return float(v_cm_s * 0.036)


def make_speed_segment(df, i, j, fps, sx, sy, dt_frames, speed_end_frame, run_start_idx, run_end_idx):
    if i < run_start_idx or j > run_end_idx:
        return None

    r1 = df.iloc[i]
    r2 = df.iloc[j]
    f1 = int(r1["Frame"])
    f2 = int(r2["Frame"])

    if f2 - f1 != dt_frames or f2 > speed_end_frame:
        return None
    if not is_valid_point(r1) or not is_valid_point(r2):
        return None

    x1 = float(r1["X"])
    y1 = float(r1["Y"])
    x2 = float(r2["X"])
    y2 = float(r2["Y"])
    if min(x1, y1, x2, y2) <= 0:
        return None
    
    # 只保留向右的速度段
    # 你的影片中左到右擊球 = X 增加
    if x2 <= x1:
        return None

    speed = calc_segment_speed_basic_kmh(x1, y1, x2, y2, fps, sx, sy, dt_frames)
    if not np.isfinite(speed) or speed > MAX_SPEED_KMH:
        return None

    return float(speed), f1, f2


def compute_net_zone_speed_for_stroke(df: pd.DataFrame, stroke: Dict, fps: float, table_corners, net_zone_points):
    """
    Compute only net-zone max speed.
    This no longer depends on hit_frame and no longer exports avg/max speed.
    """
    if table_corners is None or net_zone_points is None:
        return None

    frame_start = int(stroke["frame_start"])
    frame_end = int(stroke["frame_end"])
    run_start_idx = int(stroke["run_start_idx"])
    run_end_idx = int(stroke["run_end_idx"])
    sx, sy = compute_fixed_scales(table_corners)

    all_segments = []

    for i in range(run_start_idx, run_end_idx + 1):
        row = df.iloc[i]
        frame_id = int(row["Frame"])

        if frame_id < frame_start or frame_id > frame_end or not is_valid_point(row):
            continue

        x = float(row["X"])
        y = float(row["Y"])
        if min(x, y) <= 0:
            continue

        speed_candidates = []
        for speed_type, start_i, end_i, dt_frames in (
            ("1f", i - 1, i, 1),
            ("2f", i, i + 2, 2),
            ("c2f", i - 1, i + 1, 2),
        ):
            seg = make_speed_segment(
                df, start_i, end_i, fps, sx, sy, dt_frames,
                speed_end_frame=frame_end,
                run_start_idx=run_start_idx,
                run_end_idx=run_end_idx,
            )
            if seg is not None:
                speed_candidates.append((speed_type, *seg))

        if not speed_candidates:
            continue

        speed_map = {speed_type: speed for speed_type, speed, _, _ in speed_candidates}
        best_type, best_speed, best_start, best_end = max(speed_candidates, key=lambda item: item[1])

        all_segments.append({
            "frame": frame_id,
            "best_type": best_type,
            "best_speed": float(best_speed),
            "best_start": int(best_start),
            "best_end": int(best_end),
            "speed_1f": speed_map.get("1f"),
            "speed_2f": speed_map.get("2f"),
            "speed_c2f": speed_map.get("c2f"),
            "in_net": point_in_net_zone(x, y, net_zone_points),
        })

    if not all_segments:
        return None

    expanded_net_idx = set()
    for idx, seg in enumerate(all_segments):
        if seg["in_net"]:
            expanded_net_idx.update(j for j in (idx - 1, idx, idx + 1) if 0 <= j < len(all_segments))

    net_segments = [all_segments[j] for j in sorted(expanded_net_idx)]
    best_net = max(net_segments, key=lambda seg: seg["best_speed"]) if net_segments else None

    return {
        "net_zone_max_speed_kmh": best_net["best_speed"] if best_net else None,
        "net_zone_max_speed_type": best_net["best_type"] if best_net else "",
        "net_zone_max_speed_start_frame": best_net["best_start"] if best_net else None,
        "net_zone_max_speed_end_frame": best_net["best_end"] if best_net else None,
        "net_zone_max_speed_1f_kmh": max((seg["speed_1f"] for seg in net_segments if seg["speed_1f"] is not None), default=None),
        "net_zone_max_speed_2f_kmh": max((seg["speed_2f"] for seg in net_segments if seg["speed_2f"] is not None), default=None),
        "net_zone_max_speed_c2f_kmh": max((seg["speed_c2f"] for seg in net_segments if seg["speed_c2f"] is not None), default=None),
        "sx_cm_per_px": sx,
        "sy_cm_per_px": sy,
    }


def zone_info_to_arrays(zone_info: Dict):
    table_corners = None
    net_zone_points = None

    try:
        if zone_info.get("table_p1_x", "") != "":
            table_corners = np.array([
                [zone_info["table_p1_x"], zone_info["table_p1_y"]],
                [zone_info["table_p2_x"], zone_info["table_p2_y"]],
                [zone_info["table_p3_x"], zone_info["table_p3_y"]],
                [zone_info["table_p4_x"], zone_info["table_p4_y"]],
            ], dtype=np.float32)

        if zone_info.get("net_p1_x", "") != "":
            net_zone_points = np.array([
                [zone_info[f"net_p{i}_x"], zone_info[f"net_p{i}_y"]] for i in range(1, 9)
            ], dtype=np.float32)
    except Exception:
        return None, None

    return table_corners, net_zone_points


def append_note(note: str, value: str) -> str:
    if note is None:
        note = ""
    try:
        if pd.isna(note):
            note = ""
    except Exception:
        pass

    note = str(note).strip()
    if note == "":
        return value
    if value in note.split(";"):
        return note
    return f"{note};{value}"


def update_net_note(df: pd.DataFrame, stroke: Dict, net_zone_points, note: str) -> str:
    if net_zone_points is None:
        return note

    # 保留 net_stop：最後一幀停在 net zone
    end_row = df[df["Frame"] == int(stroke["frame_end"])]
    if len(end_row) > 0 and int(end_row.iloc[0]["Visibility"]) == 1:
        ex = float(end_row.iloc[0]["X"])
        ey = float(end_row.iloc[0]["Y"])
        if point_in_net_zone(ex, ey, net_zone_points):
            note = append_note(note, "net_stop")

    return note


def value_or_blank(value):
    return value if value is not None else ""


def build_stroke_summary_csv(
    df: pd.DataFrame,
    strokes: List[Dict],
    fps: float,
    helper_zone_info: Dict,
) -> pd.DataFrame:
    rows = []

    if helper_zone_info is None:
        raise ValueError("helper_zone_info is required. Run helper_table.py first to create *_helper_table.json.")

    for stroke in strokes:
        zone_info = helper_zone_info.copy()
        table_corners, net_zone_points = zone_info_to_arrays(zone_info)

        valid = stroke["valid"]
        note = stroke["note"]
        speed_metrics = None

        is_no_hit = int(valid) == 0 or "no_hit" in str(note).split(";")

        if is_no_hit:
            # no_hit means the stroke did not turn into a left-to-right hit.
            # Keep the row, but do not compute any speed.
            speed_metrics = None
            valid = 0
            note = append_note(note, "no_hit")
        elif table_corners is not None and net_zone_points is not None and fps is not None and fps > 0:
            speed_metrics = compute_net_zone_speed_for_stroke(df, stroke, fps, table_corners, net_zone_points)
            if speed_metrics is None:
                valid = 0
                note = append_note(note, "no_clean_speed_segment")
            elif speed_metrics["net_zone_max_speed_kmh"] is None:
                note = append_note(note, "no_ball_in_net_zone")
        else:
            note = append_note(note, "no_video_or_table_geometry")

        note = update_net_note(df, stroke, net_zone_points, note)

        row_out = {
            "stroke_id": stroke["stroke_id"],
            "frame_start": safe_int(stroke["frame_start"]),
            "frame_end": safe_int(stroke["frame_end"]),
            "bounce_frame": int(stroke.get("bounce_frame", 0) or 0),
            "net_zone_max_speed_kmh": value_or_blank(speed_metrics["net_zone_max_speed_kmh"] if speed_metrics else None),
            "net_zone_max_speed_1f_kmh": value_or_blank(speed_metrics["net_zone_max_speed_1f_kmh"] if speed_metrics else None),
            "net_zone_max_speed_2f_kmh": value_or_blank(speed_metrics["net_zone_max_speed_2f_kmh"] if speed_metrics else None),
            "net_zone_max_speed_c2f_kmh": value_or_blank(speed_metrics["net_zone_max_speed_c2f_kmh"] if speed_metrics else None),
            "net_zone_max_speed_type": speed_metrics["net_zone_max_speed_type"] if speed_metrics else "",
            "net_zone_max_speed_start_frame": value_or_blank(speed_metrics["net_zone_max_speed_start_frame"] if speed_metrics else None),
            "net_zone_max_speed_end_frame": value_or_blank(speed_metrics["net_zone_max_speed_end_frame"] if speed_metrics else None),
            "sx_cm_per_px": value_or_blank(speed_metrics["sx_cm_per_px"] if speed_metrics else None),
            "sy_cm_per_px": value_or_blank(speed_metrics["sy_cm_per_px"] if speed_metrics else None),
            "valid": valid,
            "note": note,
        }
        row_out.update(zone_info)
        rows.append(row_out)

    return pd.DataFrame(rows)


def draw_visual_video(
    video_path: str,
    df: pd.DataFrame,
    strokes: List[Dict],
    summary_df: pd.DataFrame,
    output_video_path: str,
    video_codec: str = "h264_nvenc",
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    try:
        out = FFmpegWriter(output_video_path, width, height, fps, codec=video_codec)
        used_codec = video_codec
    except Exception as e:
        print(f"[draw_visual_video] {video_codec} init failed ({e}), fallback to libx264")
        out = FFmpegWriter(output_video_path, width, height, fps, codec="libx264")
        used_codec = "libx264"

    print(f"FFmpegWriter opened: codec={used_codec}, {width}x{height}@{fps:.2f}fps, save={output_video_path}")

    frame_to_stroke = build_frame_to_stroke_map(strokes)
    summary_map = {int(row["stroke_id"]): row for _, row in summary_df.iterrows()}
    frame_to_row = {int(df.iloc[idx]["Frame"]): idx for idx in range(len(df))}

    default_summary_row = summary_df.iloc[0] if summary_df is not None and len(summary_df) > 0 else None

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Always draw the helper_table geometry, so the visualize video clearly shows
        # the fixed table and near-net zone read from *_helper_table.json.
        if default_summary_row is not None:
            table_pts = extract_zone_points(default_summary_row, "table", 4)
            net_pts = extract_zone_points(default_summary_row, "net", 8)
            if table_pts is not None:
                draw_polygon(frame, table_pts, (255, 0, 0), thickness=2, fill=False)
            if net_pts is not None:
                draw_helper_box(frame, net_pts, color=(0, 255, 255), thickness=2, fill_alpha=0.12)

        if frame_id in frame_to_stroke and frame_id in frame_to_row:
            stroke = frame_to_stroke[frame_id]
            row = df.iloc[frame_to_row[frame_id]]
            if int(row["Visibility"]) == 1:
                draw_stroke_overlay(frame, df, frame_to_row, frame_id, stroke, summary_map)

        # Draw current frame number on every frame
        cv2.putText(frame,f"Frame: {frame_id}", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3, cv2.LINE_AA,)

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()


def draw_stroke_overlay(frame, df, frame_to_row, frame_id: int, stroke: Dict, summary_map: Dict):
    x = int(df.iloc[frame_to_row[frame_id]]["X"])
    y = int(df.iloc[frame_to_row[frame_id]]["Y"])
    color = (0, 255, 0) if int(stroke["valid"]) == 1 else (0, 0, 255)

    cv2.circle(frame, (x, y), 6, color, -1)

    sid = int(stroke["stroke_id"])
    f_start = int(stroke["frame_start"])
    f_end = int(stroke["frame_end"])
    bounce_frame = stroke.get("bounce_frame", 0)

    points = []
    for f in range(max(f_start, frame_id - 25), min(f_end, frame_id) + 1):
        if f not in frame_to_row:
            continue
        r = df.iloc[frame_to_row[f]]
        if int(r["Visibility"]) == 1:
            points.append((int(r["X"]), int(r["Y"])))

    for p in points:
        cv2.circle(frame, p, 3, color, -1)
    for k in range(1, len(points)):
        cv2.line(frame, points[k - 1], points[k], color, 2)

    note = stroke.get("note", "")
    text = f"stroke={sid} start={f_start} end={f_end} bounce={bounce_frame} valid={stroke['valid']} {note}"
    cv2.putText(frame, text, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    # Table / net zone is drawn once per frame in draw_visual_video().
    # Do not draw it again here, otherwise stroke frames get a double-filled zone.

    if frame_id == f_start:
        cv2.putText(frame, "START", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    if bounce_frame and frame_id == int(bounce_frame):
        cv2.putText(frame, "BOUNCE", (x + 10, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2, cv2.LINE_AA)
    if frame_id == f_end:
        cv2.putText(frame, "END", (x + 10, y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)


def build_export_stroke_csv(summary_df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "stroke_id",
        "frame_start",
        "frame_end",
        "bounce_frame",
        "net_zone_max_speed_kmh",
        "net_zone_max_speed_type",
        "net_zone_max_speed_start_frame",
        "net_zone_max_speed_end_frame",
        "zone_label",
        "in_table",
        "valid",
        "note",
    ]
    return keep_columns(summary_df, keep_cols)


def build_export_zone_detail_csv(summary_df_full: pd.DataFrame) -> pd.DataFrame:
    return keep_columns(summary_df_full, ["stroke_id"] + ZONE_OUT_COLS)


def build_export_speed_detail_csv(summary_df_full: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "stroke_id", "frame_start", "frame_end", "bounce_frame", "valid", "note",
        "net_zone_max_speed_kmh", "net_zone_max_speed_type",
        "net_zone_max_speed_start_frame", "net_zone_max_speed_end_frame",
        "net_zone_max_speed_1f_kmh", "net_zone_max_speed_2f_kmh", "net_zone_max_speed_c2f_kmh",
        "sx_cm_per_px", "sy_cm_per_px",
    ]
    return keep_columns(summary_df_full, keep_cols)


def keep_columns(df: pd.DataFrame, keep_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in keep_cols:
        if col not in out.columns:
            out[col] = ""
    return out[keep_cols]



LANDING_MERGE_COLS = [
    "bounce_frame",
    "ball_px",
    "ball_py",
    "x_cm",
    "y_cm",
    "zone_col",
    "zone_row",
    "zone_label",
    "in_table",
    "in_table_strict",
    "in_table_relaxed",
    "edge_bounce",
    "right_half_landing",
    "bounce_type",
    "ball_py_smooth",
    "drop_before_px",
    "rise_after_px",
    "pre_slope_px_per_frame",
    "post_slope_px_per_frame",
    "piecewise_rmse_px",
    "single_rmse_px",
    "model_improvement_px",
    "bounce_score",
]


def merge_landing_results(summary_df_full: pd.DataFrame, df_land: pd.DataFrame) -> pd.DataFrame:
    """Merge piecewise landing results back to the stroke summary."""
    out = summary_df_full.copy()

    if df_land is None or df_land.empty:
        for col in LANDING_MERGE_COLS:
            if col not in out.columns:
                out[col] = ""
        return out

    merge_cols = ["stroke_id"] + [c for c in LANDING_MERGE_COLS if c in df_land.columns]
    out = out.merge(df_land[merge_cols], on="stroke_id", how="left", suffixes=("", "_landing"))

    if "bounce_frame_landing" in out.columns:
        out["bounce_frame"] = out["bounce_frame_landing"].fillna(out["bounce_frame"]).fillna(0).astype(int)
        out.drop(columns=["bounce_frame_landing"], inplace=True)

    for col in LANDING_MERGE_COLS:
        landing_col = f"{col}_landing"
        if landing_col in out.columns:
            if col in out.columns:
                out[col] = out[landing_col].combine_first(out[col])
            else:
                out[col] = out[landing_col]
            out.drop(columns=[landing_col], inplace=True)
        elif col not in out.columns:
            out[col] = ""

    return out


def sync_bounce_frames_to_strokes(strokes: List[Dict], summary_df_full: pd.DataFrame) -> None:
    """Update strokes list in-place so visual video can draw updated fields."""
    if summary_df_full is None or summary_df_full.empty:
        return

    row_map = {}
    for _, row in summary_df_full.iterrows():
        try:
            row_map[int(row["stroke_id"])] = row
        except Exception:
            continue

    for stroke in strokes:
        sid = int(stroke.get("stroke_id", -1))
        if sid not in row_map:
            continue

        row = row_map[sid]

        if "bounce_frame" in row:
            try:
                stroke["bounce_frame"] = int(row.get("bounce_frame", 0) or 0)
            except Exception:
                pass

        if "valid" in row:
            try:
                stroke["valid"] = int(row.get("valid", stroke.get("valid", 1)) or 0)
            except Exception:
                pass

        if "note" in row:
            try:
                note_value = row.get("note", "")
                stroke["note"] = "" if pd.isna(note_value) else str(note_value)
            except Exception:
                pass

def process_single_video(
    video_file,
    ball_csv,
    save_dir,
    min_left_segments=5,
    min_candidate_frames=50,
    min_no_hit_candidate_frames=20,
    max_step_th=300.0,
    max_abs_dy_th=45.0,
    left_half_ratio=0.35,
    right_side_ratio=0.6,
    fps=120.0,
    frame_w=1920,
    frame_h=1080,
    save_video=False,
    video_codec="h264_nvenc",
    helper_table_json=None,
    near_dist=NEAR_NET_DIST,
    box_height=BOX_HEIGHT,
):
    ensure_dir(save_dir)
    df = pd.read_csv(ball_csv)

    for col in ["Frame", "Visibility", "X", "Y"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    has_video = video_file is not None and os.path.exists(video_file)
    frame_reader = FrameReader(video_file) if has_video else CsvFrameInfo(df, fps=fps, frame_w=frame_w, frame_h=frame_h)
    if not has_video:
        print(f"[WARN] no original/predict mp4 found for csv, run csv-only mode: {ball_csv}")

    if helper_table_json is None:
        raise ValueError("helper_table_json is required. Run helper_table.py first to create *_helper_table.json.")
    if not os.path.exists(helper_table_json):
        raise FileNotFoundError(f"helper_table_json not found: {helper_table_json}")

    corners_lf_rf_rb_lb = load_table_corners(helper_table_json)
    frame_shape = (frame_reader.height, frame_reader.width, 3)
    helper_zone_info = build_zone_info_from_helper_table(
        corners_lf_rf_rb_lb=corners_lf_rf_rb_lb,
        frame_shape=frame_shape,
        near_dist=near_dist,
        box_height=box_height,
    )
    print(f"[INFO] use helper_table geometry: {helper_table_json}")
    table_dbg, net_dbg = zone_info_to_arrays(helper_zone_info)
    print("[INFO] table order for cm mapping: p1=NL, p2=FL, p3=FR, p4=NR")
    print(f"[INFO] helper table_p1~p4:\n{table_dbg}")

    try:
        strokes = detect_strokes_from_runs(
            df=df,
            frame_w=frame_reader.width,
            min_left_segments=min_left_segments,
            min_candidate_frames=min_candidate_frames,
            min_no_hit_candidate_frames=min_no_hit_candidate_frames,
            max_step_th=max_step_th,
            max_abs_dy_th=max_abs_dy_th,
            left_half_ratio=left_half_ratio,
            right_side_ratio=right_side_ratio,
        )

        summary_df_full = build_stroke_summary_csv(
            df=df,
            strokes=strokes,
            fps=frame_reader.fps,
            helper_zone_info=helper_zone_info,
        )

        # Landing module detects bounce_frame with the current piecewise trajectory method.
        # It writes landing_detail.csv, landing_heatmap.png, landing_zones.png, and zone_stats.csv.
        landing_input_df = summary_df_full.copy()
        if "note" in landing_input_df.columns:
            landing_input_df = landing_input_df[
                ~landing_input_df["note"].fillna("").astype(str).str.split(";").apply(lambda parts: "no_hit" in parts)
            ].copy()

        df_land = landing.compute_landings_with_bounce(landing_input_df, df, save_dir=save_dir)
        summary_df_full = merge_landing_results(summary_df_full, df_land)
        sync_bounce_frames_to_strokes(strokes, summary_df_full)

        base = os.path.splitext(os.path.basename(video_file))[0] if has_video else strip_csv_suffix(ball_csv)
        csv_path = os.path.join(save_dir, f"{base}_stroke_zone.csv")
        zone_detail_csv_path = os.path.join(save_dir, f"{base}_zone_detail.csv")
        speed_detail_csv_path = os.path.join(save_dir, f"{base}_net_zone_speed_detail.csv")
        video_path = os.path.join(save_dir, f"{base}_stroke_zone_visualize.mp4")

        build_export_stroke_csv(summary_df_full).to_csv(csv_path, index=False, encoding="utf-8-sig")
        build_export_zone_detail_csv(summary_df_full).to_csv(zone_detail_csv_path, index=False, encoding="utf-8-sig")
        build_export_speed_detail_csv(summary_df_full).to_csv(speed_detail_csv_path, index=False, encoding="utf-8-sig")

        if save_video and has_video:
            draw_visual_video(video_file, df, strokes, summary_df_full, video_path, video_codec=video_codec)
        elif save_video and not has_video:
            print("[WARN] --save_video was set, but no mp4 was found. Skip visual video output.")
    finally:
        frame_reader.release()

    print(f"saved csv   : {csv_path}")
    print(f"saved zone  : {zone_detail_csv_path}")
    print(f"saved speed : {speed_detail_csv_path}")
    if save_video and has_video:
        print(f"saved video : {video_path}")
    elif save_video and not has_video:
        print("saved video : skipped (no matching mp4)")
    else:
        print("saved video : skipped (--save_video not set)")
    print(f"num strokes : {len(summary_df_full)}")


def process_video_root(
    video_root,
    save_root=None,
    csv_suffixes=("_ball.csv", "_bass.csv"),
    min_left_segments=5,
    min_candidate_frames=50,
    min_no_hit_candidate_frames=20,
    max_step_th=300.0,
    max_abs_dy_th=45.0,
    left_half_ratio=0.35,
    right_side_ratio=0.5,
    fps=120.0,
    frame_w=1920,
    frame_h=1080,
    save_video=False,
    video_codec="h264_nvenc",
    near_dist=NEAR_NET_DIST,
    box_height=BOX_HEIGHT,
):
    search_root = save_root if save_root is not None else video_root
    ball_csv_files = collect_ball_csvs(search_root, csv_suffixes=csv_suffixes)
    if not ball_csv_files:
        raise RuntimeError(f"No ball csv files found under: {search_root}")

    print(f"[INFO] video_root : {video_root}")
    print(f"[INFO] search_root: {search_root}")
    print(f"[INFO] found {len(ball_csv_files)} csv files under {search_root}")

    for i, ball_csv in enumerate(ball_csv_files, 1):
        video_file = find_video_for_ball_csv(ball_csv, video_root=video_root, csv_suffixes=csv_suffixes)
        save_dir = os.path.dirname(ball_csv)
        print("=" * 80)
        print(f"[BATCH] ({i}/{len(ball_csv_files)})")
        print(f"[BATCH] video    : {video_file if video_file else 'None (csv-only)'}")
        print(f"[BATCH] ball csv : {ball_csv}")
        print(f"[BATCH] save dir : {save_dir}")

        base_stem = strip_csv_suffix(ball_csv, csv_suffixes)
        helper_table_json = os.path.join(os.path.dirname(ball_csv), f"{base_stem}_helper_table.json")
        print(f"[BATCH] helper  : {helper_table_json}")

        try:
            if not os.path.exists(helper_table_json):
                raise FileNotFoundError(
                    f"helper_table_json not found: {helper_table_json}. "
                    "Run helper_table.py first to create this json."
                )
            process_single_video(
                video_file=video_file,
                ball_csv=ball_csv,
                save_dir=save_dir,
                min_left_segments=min_left_segments,
                min_candidate_frames=min_candidate_frames,
                min_no_hit_candidate_frames=min_no_hit_candidate_frames,
                max_step_th=max_step_th,
                max_abs_dy_th=max_abs_dy_th,
                left_half_ratio=left_half_ratio,
                right_side_ratio=right_side_ratio,
                fps=fps,
                frame_w=frame_w,
                frame_h=frame_h,
                save_video=save_video,
                video_codec=video_codec,
                helper_table_json=helper_table_json,
                near_dist=near_dist,
                box_height=box_height,
            )
        except Exception as e:
            print(f"[ERROR] failed on {video_file}")
            print(f"[ERROR] {e}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_file", type=str, default=None)
    parser.add_argument("--ball_csv", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--video_root", type=str, default=None)
    parser.add_argument("--save_root", type=str, default=None)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--video_codec", type=str, default="h264_nvenc", choices=["h264_nvenc", "libx264"])
    parser.add_argument("--csv_suffixes", type=str, nargs="+", default=["_ball.csv", "_bass.csv"])
    parser.add_argument("--min_left_segments", type=int, default=5)
    parser.add_argument("--min_candidate_frames", type=int, default=50)
    parser.add_argument("--min_no_hit_candidate_frames", type=int, default=20)
    parser.add_argument("--max_step_th", type=float, default=300.0)
    parser.add_argument("--max_abs_dy_th", type=float, default=45.0)
    parser.add_argument("--left_half_ratio", type=float, default=0.35)
    parser.add_argument("--right_side_ratio", type=float, default=0.5)
    parser.add_argument("--helper_table_json", type=str, default=None)
    parser.add_argument("--near_dist", type=float, default=NEAR_NET_DIST)
    parser.add_argument("--box_height", type=float, default=BOX_HEIGHT)
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--frame_w", type=int, default=1920)
    parser.add_argument("--frame_h", type=int, default=1080)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.video_root is not None:
        process_video_root(
            video_root=args.video_root,
            save_root=args.save_root,
            csv_suffixes=tuple(args.csv_suffixes),
            min_left_segments=args.min_left_segments,
            min_candidate_frames=args.min_candidate_frames,
            min_no_hit_candidate_frames=args.min_no_hit_candidate_frames,
            max_step_th=args.max_step_th,
            max_abs_dy_th=args.max_abs_dy_th,
            left_half_ratio=args.left_half_ratio,
            right_side_ratio=args.right_side_ratio,
            fps=args.fps,
            frame_w=args.frame_w,
            frame_h=args.frame_h,
            save_video=args.save_video,
            video_codec=args.video_codec,
            near_dist=args.near_dist,
            box_height=args.box_height,
        )
        return

    if args.video_file is not None or args.ball_csv is not None:
        if args.ball_csv is None or args.save_dir is None:
            raise ValueError("single mode requires --ball_csv --save_dir. --video_file is optional.")

        process_single_video(
            video_file=args.video_file,
            ball_csv=args.ball_csv,
            save_dir=args.save_dir,
            min_left_segments=args.min_left_segments,
            min_candidate_frames=args.min_candidate_frames,
            min_no_hit_candidate_frames=args.min_no_hit_candidate_frames,
            max_step_th=args.max_step_th,
            max_abs_dy_th=args.max_abs_dy_th,
            left_half_ratio=args.left_half_ratio,
            right_side_ratio=args.right_side_ratio,
            fps=args.fps,
            frame_w=args.frame_w,
            frame_h=args.frame_h,
            save_video=args.save_video,
            video_codec=args.video_codec,
            helper_table_json=args.helper_table_json,
            near_dist=args.near_dist,
            box_height=args.box_height,
        )
        return

    raise ValueError("Please provide either --video_root or --ball_csv")


if __name__ == "__main__":
    main()
