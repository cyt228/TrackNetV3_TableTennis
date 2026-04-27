import argparse
import math
import os
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd


# Basic helpers

def is_valid_point(row) -> bool:
    return int(row["Visibility"]) == 1


def safe_int(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return int(value)


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def calc_step(x1, y1, x2, y2) -> float:
    return math.hypot(float(x2) - float(x1), float(y2) - float(y1))


# Stroke detection

def collect_valid_runs(df: pd.DataFrame) -> List[Dict]:
    """Collect consecutive visible frame runs."""
    runs = []
    i = 0
    n = len(df)

    while i < n:
        if not is_valid_point(df.iloc[i]):
            i += 1
            continue

        start = i
        end = i

        while end + 1 < n:
            cur_frame = int(df.iloc[end]["Frame"])
            next_row = df.iloc[end + 1]
            next_frame = int(next_row["Frame"])

            if not is_valid_point(next_row) or next_frame != cur_frame + 1:
                break
            end += 1

        if end > start:
            runs.append({"start_idx": start, "end_idx": end})
        i = end + 1

    return runs


def find_bounce_frame(df: pd.DataFrame, hit_idx: int, end_idx: int, frame_w: int) -> int:
    """Find the first right-side local y peak after hit."""
    if end_idx - hit_idx < 2:
        return 0

    right_x_th = frame_w * 0.65

    for i in range(hit_idx + 1, end_idx):
        y_prev = float(df.iloc[i - 1]["Y"])
        y_cur = float(df.iloc[i]["Y"])
        y_next = float(df.iloc[i + 1]["Y"])
        x_cur = float(df.iloc[i]["X"])

        dy1 = y_cur - y_prev
        dy2 = y_next - y_cur
        is_peak = False

        if dy1 > 0 and dy2 < 0:
            is_peak = True
        elif dy1 > 0 and dy2 == 0 and i + 2 <= end_idx:
            is_peak = float(df.iloc[i + 2]["Y"]) < y_next
        elif dy1 == 0 and dy2 < 0 and i - 2 >= hit_idx:
            is_peak = y_prev > float(df.iloc[i - 2]["Y"])

        if is_peak and x_cur >= right_x_th:
            return int(df.iloc[i]["Frame"])

    return 0


def find_relaxed_end_idx(
    df: pd.DataFrame,
    hit_idx: int,
    run_end_idx: int,
    max_backward_tol: float = 12.0,
    max_nonforward_count: int = 3,
) -> int:
    """Find stroke end after hit, allowing small temporary non-forward movement."""
    end_idx = hit_idx + 1
    nonforward_count = 0

    for j in range(hit_idx + 1, run_end_idx):
        pdx = float(df.iloc[j + 1]["X"]) - float(df.iloc[j]["X"])

        if pdx > 0:
            end_idx = j + 1
            nonforward_count = 0
            continue

        if pdx >= -max_backward_tol:
            nonforward_count += 1
            if nonforward_count <= max_nonforward_count:
                end_idx = j + 1
                continue

        break

    return end_idx


def find_left_start_idx(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    min_left_segments: int,
    max_step_th: float,
    max_abs_dy_th: float,
) -> Optional[int]:
    """Find the start of a stable left-moving segment."""
    count = 0
    frame_start_idx = None

    for i in range(start_idx, end_idx):
        x1 = float(df.iloc[i]["X"])
        y1 = float(df.iloc[i]["Y"])
        x2 = float(df.iloc[i + 1]["X"])
        y2 = float(df.iloc[i + 1]["Y"])
        dx = x2 - x1
        dy = y2 - y1
        step = calc_step(x1, y1, x2, y2)

        if step <= max_step_th and abs(dy) <= max_abs_dy_th and dx < 0:
            if count == 0:
                frame_start_idx = i
            count += 1
            if count >= min_left_segments:
                return frame_start_idx
        else:
            count = 0
            frame_start_idx = None

    return None


def find_best_hit_candidate(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    frame_w: int,
    max_step_th: float,
    left_half_ratio: float,
) -> Optional[Dict]:
    """Choose the best left-to-right turning point as hit."""
    hit_x_limit = frame_w * float(left_half_ratio)
    local_window = 3
    future_window = 8
    min_rise_px = 80.0
    min_net_right = 3
    candidates = []

    for i in range(start_idx + 1, end_idx):
        x_im1 = float(df.iloc[i - 1]["X"])
        x_i = float(df.iloc[i]["X"])
        x_ip1 = float(df.iloc[i + 1]["X"])

        prev_dx = x_i - x_im1
        curr_dx = x_ip1 - x_i
        is_turn = prev_dx < 0 and curr_dx > 0

        if not is_turn and prev_dx == 0 and curr_dx > 0 and i - 2 >= start_idx:
            x_im2 = float(df.iloc[i - 2]["X"])
            is_turn = x_im1 - x_im2 < 0

        if not is_turn:
            continue

        if not (x_i <= hit_x_limit and x_ip1 <= hit_x_limit):
            continue

        left_bound = max(start_idx, i - local_window)
        right_bound = min(end_idx, i + local_window)
        local_x = [float(df.iloc[k]["X"]) for k in range(left_bound, right_bound + 1)]
        if x_i > min(local_x) + 8:
            continue

        check_end = min(end_idx, i + future_window)
        max_future_x = x_i
        right_count = 0
        left_count = 0

        for j in range(i, check_end):
            pdx = float(df.iloc[j + 1]["X"]) - float(df.iloc[j]["X"])
            if pdx > 0:
                right_count += 1
            elif pdx < 0:
                left_count += 1
            max_future_x = max(max_future_x, float(df.iloc[j + 1]["X"]))

        rise_px = max_future_x - x_i
        net_right = right_count - left_count
        if rise_px < min_rise_px or net_right < min_net_right:
            continue

        abnormal_jump_count = 0
        for j in range(i, min(end_idx, i + 6)):
            x_a = float(df.iloc[j]["X"])
            y_a = float(df.iloc[j]["Y"])
            x_b = float(df.iloc[j + 1]["X"])
            y_b = float(df.iloc[j + 1]["Y"])
            dx = x_b - x_a
            dy = y_b - y_a
            step = calc_step(x_a, y_a, x_b, y_b)

            if step > max_step_th * 1.2 or abs(dx) > 140 or abs(dy) > 60:
                abnormal_jump_count += 1
                if abnormal_jump_count > 1:
                    break

        if abnormal_jump_count > 1:
            continue

        stroke_end_idx = find_relaxed_end_idx(df, hit_idx=i, run_end_idx=end_idx)
        forward_count = 0
        for j in range(i, stroke_end_idx):
            if float(df.iloc[j + 1]["X"]) - float(df.iloc[j]["X"]) > 0:
                forward_count += 1

        candidates.append({
            "hit_idx": i,
            "end_idx": stroke_end_idx,
            "forward_count": forward_count,
            "post_hit_frames": int(df.iloc[stroke_end_idx]["Frame"]) - int(df.iloc[i]["Frame"]) + 1,
            "rise_px": rise_px,
            "net_right": net_right,
            "end_x": float(df.iloc[stroke_end_idx]["X"]),
        })

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda c: (c["end_x"], c["post_hit_frames"], c["forward_count"], c["rise_px"], -c["hit_idx"]),
    )


def make_stroke(
    stroke_id: int,
    run_start_idx: int,
    run_end_idx: int,
    frame_start: int,
    hit_frame,
    frame_end: int,
    stroke_end_idx: int,
    bounce_frame: int,
    valid: int,
    note: str,
) -> Dict:
    return {
        "stroke_id": stroke_id,
        "run_start_idx": run_start_idx,
        "run_end_idx": run_end_idx,
        "frame_start": frame_start,
        "hit_frame": hit_frame,
        "frame_end": frame_end,
        "stroke_end_idx": stroke_end_idx,
        "bounce_frame": bounce_frame,
        "valid": valid,
        "note": note,
    }


def detect_strokes_from_runs(
    df: pd.DataFrame,
    frame_w: int,
    min_left_segments: int = 5,
    min_candidate_frames: int = 35,
    max_step_th: float = 130.0,
    max_abs_dy_th: float = 45.0,
    left_half_ratio: float = 0.5,
    right_side_ratio: float = 0.5,
) -> List[Dict]:
    """Detect strokes using left-moving start and left-to-right hit turning point."""
    strokes = []
    stroke_id = 1

    for run in collect_valid_runs(df):
        run_s = run["start_idx"]
        run_e = run["end_idx"]
        s = run_s

        while s < run_e:
            if run_e - s < 1:
                break

            frame_start_idx = find_left_start_idx(df, s, run_e, min_left_segments, max_step_th, max_abs_dy_th)
            if frame_start_idx is None:
                break

            frame_start = int(df.iloc[frame_start_idx]["Frame"])
            best = find_best_hit_candidate(df, s, run_e, frame_w, max_step_th, left_half_ratio)

            if best is not None:
                frame_end = int(df.iloc[best["end_idx"]]["Frame"])
                if frame_end - frame_start + 1 >= min_candidate_frames:
                    end_x = float(df.iloc[best["end_idx"]]["X"])
                    right_x_limit = frame_w * float(right_side_ratio)

                    if end_x >= right_x_limit:
                        bounce_frame = find_bounce_frame(df, best["hit_idx"], best["end_idx"], frame_w)
                        strokes.append(make_stroke(
                            stroke_id, s, run_e, frame_start, int(df.iloc[best["hit_idx"]]["Frame"]),
                            frame_end, best["end_idx"], bounce_frame, 1, ""
                        ))
                    else:
                        strokes.append(make_stroke(stroke_id, s, run_e, frame_start, None, frame_end, best["end_idx"], 0, 0, "no_hit"))
                    stroke_id += 1

                s = best["end_idx"] + 1
                continue

            frame_end = int(df.iloc[run_e]["Frame"])
            if frame_end - frame_start + 1 >= min_candidate_frames:
                strokes.append(make_stroke(stroke_id, s, run_e, frame_start, None, frame_end, run_e, 0, 0, "no_hit"))
                stroke_id += 1
            break

    return strokes


# Debug drawing helpers

def build_frame_to_stroke_map(strokes: List[Dict]) -> Dict[int, Dict]:
    frame_map = {}
    for stroke in strokes:
        frame_start = stroke.get("frame_start")
        frame_end = stroke.get("frame_end")
        if frame_start is None or frame_end is None:
            continue
        for frame_id in range(int(frame_start), int(frame_end) + 1):
            frame_map[frame_id] = stroke
    return frame_map


def extract_zone_points(row, prefix: str, point_count: int):
    points = []
    for i in range(1, point_count + 1):
        x = row[f"{prefix}_p{i}_x"]
        y = row[f"{prefix}_p{i}_y"]
        if x == "" or y == "" or pd.isna(x) or pd.isna(y):
            return None
        points.append((int(round(float(x))), int(round(float(y)))))
    return points


def draw_polygon(frame, pts, color, thickness=2, fill=False, alpha=0.18):
    pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    if fill:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts_np], color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [pts_np], True, color, thickness, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ball_csv", type=str, required=True)
    parser.add_argument("--frame_w", type=int, required=True)
    parser.add_argument("--min_left_segments", type=int, default=5)
    parser.add_argument("--min_candidate_frames", type=int, default=35)
    parser.add_argument("--max_step_th", type=float, default=130.0)
    parser.add_argument("--max_abs_dy_th", type=float, default=45.0)
    parser.add_argument("--left_half_ratio", type=float, default=0.5)
    parser.add_argument("--right_side_ratio", type=float, default=0.5)
    args = parser.parse_args()

    df = pd.read_csv(args.ball_csv)
    strokes = detect_strokes_from_runs(
        df=df,
        frame_w=args.frame_w,
        min_left_segments=args.min_left_segments,
        min_candidate_frames=args.min_candidate_frames,
        max_step_th=args.max_step_th,
        max_abs_dy_th=args.max_abs_dy_th,
        left_half_ratio=args.left_half_ratio,
        right_side_ratio=args.right_side_ratio,
    )
    print(pd.DataFrame(strokes).to_string(index=False))


if __name__ == "__main__":
    main()
