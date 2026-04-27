import cv2
import numpy as np


TABLE_POINT_NAMES = ["LF", "RF", "RN", "LN"]


def order_table_corners(pts):
    """Sort table corners as LF, RF, RN, LN."""
    pts = np.asarray(pts, dtype=np.float32)

    idx = np.argsort(pts[:, 1])
    top2 = pts[idx[:2]]
    bottom2 = pts[idx[2:]]

    top2 = top2[np.argsort(top2[:, 0])]
    bottom2 = bottom2[np.argsort(bottom2[:, 0])]

    lf, rf = top2[0], top2[1]
    ln, rn = bottom2[0], bottom2[1]
    return np.array([lf, rf, rn, ln], dtype=np.float32)


def line_length(x1, y1, x2, y2):
    return float(np.hypot(x2 - x1, y2 - y1))


def line_angle_deg(x1, y1, x2, y2):
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))


def line_to_abc(x1, y1, x2, y2):
    """Convert two-point line to ax + by + c = 0."""
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    return np.array([a, b, c], dtype=np.float64)


def intersect_lines(line1, line2):
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    det = a1 * b2 - a2 * b1

    if abs(det) < 1e-8:
        return None

    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    return np.array([x, y], dtype=np.float32)


def extend_line_to_top_bottom(line, top_line, bottom_line, frame):
    if line is None or top_line is None or bottom_line is None:
        return None

    h, w = frame.shape[:2]
    line_abc = line_to_abc(*line)
    top_abc = line_to_abc(*top_line)
    bottom_abc = line_to_abc(*bottom_line)

    p_top = intersect_lines(line_abc, top_abc)
    p_bottom = intersect_lines(line_abc, bottom_abc)
    if p_top is None or p_bottom is None:
        return line

    p_top[0] = np.clip(p_top[0], 0, w - 1)
    p_top[1] = np.clip(p_top[1], 0, h - 1)
    p_bottom[0] = np.clip(p_bottom[0], 0, w - 1)
    p_bottom[1] = np.clip(p_bottom[1], 0, h - 1)

    if p_top[1] <= p_bottom[1]:
        return float(p_top[0]), float(p_top[1]), float(p_bottom[0]), float(p_bottom[1])
    return float(p_bottom[0]), float(p_bottom[1]), float(p_top[0]), float(p_top[1])


def detect_table(frame, debug=False):
    h, w = frame.shape[:2]

    roi_y1 = int(h * 0.45)
    roi_y2 = int(h * 0.88)
    roi_x1 = int(w * 0.22)
    roi_x2 = int(w * 0.96)
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    lower_blue = np.array([75, 40, 40], dtype=np.uint8)
    upper_blue = np.array([125, 255, 255], dtype=np.uint8)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    edges = cv2.Canny(gray, 50, 150)
    blue_dilate = cv2.dilate(blue_mask, np.ones((5, 5), np.uint8), iterations=1)
    masked_edges = cv2.bitwise_and(edges, blue_dilate)

    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi / 180, threshold=30, minLineLength=90, maxLineGap=80)
    if lines is None:
        if debug:
            print("[detect_table] no lines found")
        return [], [], []

    horiz_lines = []
    net_lines = []
    edge_lines = []

    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = line.astype(float)
        angle_abs = abs(line_angle_deg(x1, y1, x2, y2))
        length = line_length(x1, y1, x2, y2)

        gx1 = x1 + roi_x1
        gy1 = y1 + roi_y1
        gx2 = x2 + roi_x1
        gy2 = y2 + roi_y1
        gcx = (gx1 + gx2) / 2.0
        gcy = (gy1 + gy2) / 2.0

        if angle_abs < 20 or angle_abs > 160:
            if 420 <= gcy <= 880 and length >= 280:
                horiz_lines.append((gx1, gy1, gx2, gy2))
            continue

        if 30 < angle_abs < 85 or 95 < angle_abs < 150:
            if not (420 <= gcy <= 930 and length >= 220):
                continue
            if 980 <= gcx <= 1280:
                net_lines.append((gx1, gy1, gx2, gy2))
            elif gcx < 900 or gcx > 1360:
                edge_lines.append((gx1, gy1, gx2, gy2))

    if debug:
        print(f"[detect_table] horiz={len(horiz_lines)}, net={len(net_lines)}, edge={len(edge_lines)}")

    return horiz_lines, net_lines, edge_lines


def polygon_area(pts):
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def is_convex_quad(pts):
    signs = []
    for i in range(4):
        p0 = pts[i]
        p1 = pts[(i + 1) % 4]
        p2 = pts[(i + 2) % 4]
        v1 = p1 - p0
        v2 = p2 - p1
        signs.append(v1[0] * v2[1] - v1[1] * v2[0])

    signs = np.asarray(signs, dtype=np.float32)
    return np.all(signs > 0) or np.all(signs < 0)


def make_table_corners(top_line, bottom_line, left_line, right_line, frame):
    h, w = frame.shape[:2]

    top_abc = line_to_abc(*top_line)
    bottom_abc = line_to_abc(*bottom_line)
    left_abc = line_to_abc(*left_line)
    right_abc = line_to_abc(*right_line)

    lf = intersect_lines(top_abc, left_abc)
    rf = intersect_lines(top_abc, right_abc)
    rn = intersect_lines(bottom_abc, right_abc)
    ln = intersect_lines(bottom_abc, left_abc)

    if any(p is None for p in [lf, rf, rn, ln]):
        return None

    corners = np.array([lf, rf, rn, ln], dtype=np.float32)
    corners[:, 0] = np.clip(corners[:, 0], 0, w - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, h - 1)
    corners = order_table_corners(corners)

    lf, rf, rn, ln = corners
    top_width = np.linalg.norm(rf - lf)
    bottom_width = np.linalg.norm(rn - ln)
    left_height = np.linalg.norm(ln - lf)
    right_height = np.linalg.norm(rn - rf)
    height_ratio = max(left_height, right_height) / max(1.0, min(left_height, right_height))

    if top_width < 250 or bottom_width < 500:
        return None
    if left_height < 120 or right_height < 120:
        return None
    if polygon_area(corners) < 120000:
        return None
    if not (lf[1] < ln[1] and rf[1] < rn[1]):
        return None
    if top_width >= bottom_width:
        return None
    if height_ratio > 2.2:
        return None
    if not (lf[0] < rf[0] and ln[0] < rn[0]):
        return None
    if not is_convex_quad(corners):
        return None

    return corners


def build_table_from_lines(horiz_lines, edge_lines, net_lines, frame, debug=False):
    h, w = frame.shape[:2]

    if len(horiz_lines) < 2 or len(edge_lines) < 2:
        if debug:
            print("[build_table_from_lines] not enough lines")
        return None, None, None, None

    def mid_x(line):
        return (line[0] + line[2]) / 2.0

    def mid_y(line):
        return (line[1] + line[3]) / 2.0

    def x_span(line):
        return abs(line[2] - line[0])

    horiz_sorted_top = sorted(horiz_lines, key=lambda line: (mid_y(line), -x_span(line)))
    horiz_sorted_bottom = sorted(horiz_lines, key=lambda line: (-mid_y(line), -x_span(line)))
    edge_sorted_left = sorted(edge_lines, key=lambda line: (mid_x(line), -line_length(*line)))
    edge_sorted_right = sorted(edge_lines, key=lambda line: (-mid_x(line), -line_length(*line)))

    best_corners = None
    top_line = None
    bottom_line = None
    left_line = None
    right_line = None

    for top in horiz_sorted_top:
        for bottom in horiz_sorted_bottom:
            if top == bottom:
                continue
            if mid_y(bottom) <= mid_y(top) + 20:
                continue
            if line_length(*bottom) < 360 or x_span(bottom) < 300:
                continue

            for left in edge_sorted_left:
                for right in edge_sorted_right:
                    if left == right or mid_x(left) >= mid_x(right):
                        continue

                    corners = make_table_corners(top, bottom, left, right, frame)
                    if corners is None:
                        continue

                    best_corners = corners
                    top_line = top
                    bottom_line = bottom
                    left_line = left
                    right_line = right
                    break

                if best_corners is not None:
                    break

            if best_corners is not None:
                break

        if best_corners is not None:
            break

    if best_corners is None:
        if debug:
            print("[build_table_from_lines] no valid table combination found")
        return None, None, None, None

    net_line = select_net_line(net_lines, top_line, bottom_line, frame, frame_width=w)

    if debug:
        print("[build_table_from_lines] top_line:", top_line)
        print("[build_table_from_lines] bottom_line:", bottom_line)
        print("[build_table_from_lines] left_line:", left_line)
        print("[build_table_from_lines] right_line:", right_line)
        print("[build_table_from_lines] net_line:", net_line)
        print("[build_table_from_lines] corners:", best_corners)

    return best_corners, net_line, top_line, bottom_line


def select_net_line(net_lines, top_line, bottom_line, frame, frame_width):
    if not net_lines:
        return None

    def mid_x(line):
        return (line[0] + line[2]) / 2.0

    target_lines = [line for line in net_lines if 0.40 * frame_width <= mid_x(line) <= 0.70 * frame_width]
    if not target_lines:
        target_lines = list(net_lines)

    target_lines = sorted(target_lines, key=mid_x)
    clusters = []
    current_cluster = [target_lines[0]]

    for line in target_lines[1:]:
        if abs(mid_x(line) - mid_x(current_cluster[-1])) <= 35:
            current_cluster.append(line)
        else:
            clusters.append(current_cluster)
            current_cluster = [line]
    clusters.append(current_cluster)

    best_cluster = max(clusters, key=lambda group: (len(group), sum(line_length(*line) for line in group)))
    cluster_mean_x = np.mean([mid_x(line) for line in best_cluster])
    base_net_line = min(best_cluster, key=lambda line: (abs(mid_x(line) - cluster_mean_x), -line_length(*line)))

    return extend_line_to_top_bottom(base_net_line, top_line, bottom_line, frame)


def build_net_front_zone(net_line, top_line, bottom_line, up_px=60, left_shift_px=120):
    if net_line is None or top_line is None or bottom_line is None:
        return None

    x1, y1, x2, y2 = map(float, net_line)
    p1 = np.array([x1, y1], dtype=np.float32)
    p2 = np.array([x2, y2], dtype=np.float32)
    net_top, net_bottom = (p1, p2) if p1[1] <= p2[1] else (p2, p1)

    tx1, ty1, tx2, ty2 = map(float, top_line)
    top_v = np.array([tx2 - tx1, ty2 - ty1], dtype=np.float32)
    top_norm = np.linalg.norm(top_v)
    if top_norm < 1e-6:
        return None

    top_u = top_v / top_norm
    up = np.array([0, -up_px], dtype=np.float32)

    p1_zone = net_bottom + up
    p2_zone = net_top + up
    p3_zone = p2_zone - top_u * left_shift_px

    top_abc = line_to_abc(*top_line)
    bottom_abc = line_to_abc(*bottom_line)

    vertical_from_p3 = line_to_abc(p3_zone[0], p3_zone[1], p3_zone[0], p3_zone[1] + 3000)
    p4_zone = intersect_lines(vertical_from_p3, top_abc)
    if p4_zone is None:
        return None

    p12_v = p2_zone - p1_zone
    p12_norm = np.linalg.norm(p12_v)
    if p12_norm < 1e-6:
        return None

    p12_u = p12_v / p12_norm
    line_from_p4 = line_to_abc(
        p4_zone[0], p4_zone[1],
        p4_zone[0] + p12_u[0] * 3000,
        p4_zone[1] + p12_u[1] * 3000,
    )
    p5_zone = intersect_lines(line_from_p4, bottom_abc)
    if p5_zone is None:
        return None

    vertical_from_p1 = line_to_abc(p1_zone[0], p1_zone[1], p1_zone[0], p1_zone[1] + 3000)
    p6_zone = intersect_lines(vertical_from_p1, bottom_abc)
    if p6_zone is None:
        return None

    return np.array([p1_zone, p2_zone, p3_zone, p4_zone, p5_zone, p6_zone], dtype=np.float32)


def draw_full_overlay(frame, corners=None, net_line=None, net_zone=None):
    out = frame.copy()

    if corners is not None:
        corners_i = np.asarray(corners, dtype=np.float32).astype(int)
        for i in range(4):
            p1 = tuple(corners_i[i])
            p2 = tuple(corners_i[(i + 1) % 4])
            cv2.line(out, p1, p2, (0, 255, 0), 3)

        for name, (x, y) in zip(TABLE_POINT_NAMES, corners_i):
            cv2.circle(out, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(out, name, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    if net_line is not None:
        x1, y1, x2, y2 = net_line
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 3)

    if net_zone is not None:
        pts = np.asarray(net_zone, dtype=np.float32).astype(np.int32).reshape((-1, 1, 2))
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], (255, 0, 0))
        out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)
        cv2.polylines(out, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

    return out


def corners_to_list(corners):
    if corners is None:
        return [None] * 8

    corners = np.asarray(corners, dtype=np.float32)
    return [float(v) for point in corners[:4] for v in point]


def polygon_to_flat_list(polygon):
    if polygon is None:
        return [None] * 12

    polygon = np.asarray(polygon, dtype=np.float32)
    return [float(v) for point in polygon[:6] for v in point]
