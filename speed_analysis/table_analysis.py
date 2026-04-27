# python speed_analysis/table_analysis.py --video_path pred_result_NO3/044/C0086_predict.mp4

# python speed_analysis/table_analysis.py --video_root pred_result_NO3
import argparse
import csv
import os

import cv2

from table_tracker import (
    build_net_front_zone,
    build_table_from_lines,
    corners_to_list,
    detect_table,
    draw_full_overlay,
    polygon_to_flat_list,
)


ZONE_HEADER = [
    "frame",
    "lf_x", "lf_y",
    "rf_x", "rf_y",
    "rn_x", "rn_y",
    "ln_x", "ln_y",
    "zone_p1_x", "zone_p1_y",
    "zone_p2_x", "zone_p2_y",
    "zone_p3_x", "zone_p3_y",
    "zone_p4_x", "zone_p4_y",
    "zone_p5_x", "zone_p5_y",
    "zone_p6_x", "zone_p6_y",
]


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def collect_predict_videos(video_root: str):
    video_files = []
    for root, _, files in os.walk(video_root):
        for fname in files:
            if fname.endswith("_predict.mp4"):
                video_files.append(os.path.join(root, fname))
    return sorted(video_files)


def build_output_paths_for_video(video_path: str):
    video_dir = os.path.dirname(video_path)
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    return (
        os.path.join(video_dir, f"{video_stem}_zone.csv"),
        os.path.join(video_dir, f"{video_stem}_zone.mp4"),
    )


def should_refresh_geometry(frame_id: int, refresh_interval: int = 200, freeze_after_frame: int = 1400):
    if frame_id == 0:
        return True, "init"
    if frame_id <= freeze_after_frame and refresh_interval > 0 and frame_id % refresh_interval == 0:
        return True, "interval"
    return False, "reuse"


def process_video(
    video_path: str,
    csv_path: str,
    overlay_video_path: str,
    up_px: int = 120,
    left_shift_px: int = 150,
    refresh_interval: int = 200,
    freeze_after_frame: int = 1400,
    save_overlay_video: bool = True,
    debug: bool = False,
):
    ensure_dir(os.path.dirname(csv_path))
    if save_overlay_video:
        ensure_dir(os.path.dirname(overlay_video_path))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_writer = None

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if save_overlay_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(overlay_video_path, fourcc, fps, (width, height))
            if not video_writer.isOpened():
                raise RuntimeError(f"Cannot open VideoWriter: {overlay_video_path}")

        print(f"[INFO] video_path        : {video_path}")
        print(f"[INFO] total_frames      : {total_frames}")
        print(f"[INFO] csv_path          : {csv_path}")
        print(f"[INFO] overlay_video     : {overlay_video_path if save_overlay_video else 'disabled'}")
        print(f"[INFO] refresh_interval  : {refresh_interval}")
        print(f"[INFO] freeze_after_frame: {freeze_after_frame}")

        last_corners = None
        last_net_line = None
        last_net_zone = None

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(ZONE_HEADER)

            frame_id = 0
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                need_refresh, refresh_reason = should_refresh_geometry(frame_id, refresh_interval, freeze_after_frame)

                if need_refresh:
                    horiz_lines, net_lines, edge_lines = detect_table(frame, debug=False)
                    corners, net_line, top_line, bottom_line = build_table_from_lines(
                        horiz_lines, edge_lines, net_lines, frame, debug=False
                    )
                    net_zone = build_net_front_zone(net_line, top_line, bottom_line, up_px=up_px, left_shift_px=left_shift_px)

                    if corners is not None:
                        last_corners = corners
                    if net_line is not None:
                        last_net_line = net_line
                    if net_zone is not None:
                        last_net_zone = net_zone

                table_found = last_corners is not None
                net_found = last_net_line is not None
                net_zone_found = last_net_zone is not None

                writer.writerow([frame_id] + corners_to_list(last_corners) + polygon_to_flat_list(last_net_zone))

                if video_writer is not None:
                    overlay = draw_full_overlay(frame, corners=last_corners, net_line=last_net_line, net_zone=last_net_zone)
                    cv2.putText(overlay, f"refresh={int(need_refresh)} reason={refresh_reason}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(overlay, f"table={int(table_found)} net={int(net_found)} zone={int(net_zone_found)}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    video_writer.write(overlay)

                if debug and frame_id % 300 == 0:
                    print(
                        f"[progress] frame={frame_id}/{total_frames} "
                        f"refresh={int(need_refresh)} reason={refresh_reason} "
                        f"table={int(table_found)} net={int(net_found)} zone={int(net_zone_found)}"
                    )

                frame_id += 1

        print(f"[DONE] CSV saved to: {csv_path}")
        if save_overlay_video:
            print(f"[DONE] Overlay video saved to: {overlay_video_path}")

    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()


def process_video_root(
    video_root: str,
    up_px: int = 120,
    left_shift_px: int = 150,
    refresh_interval: int = 200,
    freeze_after_frame: int = 1400,
    save_overlay_video: bool = True,
    debug: bool = False,
):
    video_files = collect_predict_videos(video_root)
    if not video_files:
        raise RuntimeError(f"No *_predict.mp4 files found under: {video_root}")

    print(f"[INFO] found {len(video_files)} videos under {video_root}")

    for i, video_path in enumerate(video_files, 1):
        csv_path, overlay_video_path = build_output_paths_for_video(video_path)
        print("=" * 80)
        print(f"[BATCH] ({i}/{len(video_files)})")
        print(f"[BATCH] video : {video_path}")
        print(f"[BATCH] csv   : {csv_path}")
        print(f"[BATCH] mp4   : {overlay_video_path if save_overlay_video else 'disabled'}")

        process_video(
            video_path=video_path,
            csv_path=csv_path,
            overlay_video_path=overlay_video_path,
            up_px=up_px,
            left_shift_px=left_shift_px,
            refresh_interval=refresh_interval,
            freeze_after_frame=freeze_after_frame,
            save_overlay_video=save_overlay_video,
            debug=debug,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--video_root", type=str, default=None)
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--overlay_video_path", type=str, default=None)
    parser.add_argument("--up_px", type=int, default=120)
    parser.add_argument("--left_shift_px", type=int, default=150)
    parser.add_argument("--refresh_interval", type=int, default=200)
    parser.add_argument("--freeze_after_frame", type=int, default=1400)
    parser.add_argument("--no_overlay_video", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.video_path is not None:
        auto_csv_path, auto_overlay_video_path = build_output_paths_for_video(args.video_path)
        process_video(
            video_path=args.video_path,
            csv_path=args.csv_path or auto_csv_path,
            overlay_video_path=args.overlay_video_path or auto_overlay_video_path,
            up_px=args.up_px,
            left_shift_px=args.left_shift_px,
            refresh_interval=args.refresh_interval,
            freeze_after_frame=args.freeze_after_frame,
            save_overlay_video=not args.no_overlay_video,
            debug=args.debug,
        )
        return

    if args.video_root is not None:
        process_video_root(
            video_root=args.video_root,
            up_px=args.up_px,
            left_shift_px=args.left_shift_px,
            refresh_interval=args.refresh_interval,
            freeze_after_frame=args.freeze_after_frame,
            save_overlay_video=not args.no_overlay_video,
            debug=args.debug,
        )
        return

    raise ValueError("Please provide either --video_path or --video_root")


if __name__ == "__main__":
    main()
