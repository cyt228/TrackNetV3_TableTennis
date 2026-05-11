"""
helper_table.py (單一 Near Box 版)

座標定義：
  x: 桌子長邊方向，左 -> 右，範圍 [0, L]，L = 2.74m
  y: 桌子短邊方向，後 -> 前，範圍 [0, W]，W = 1.525m
  z: 高度方向

只偵測靠近網子的左半桌長方體:
  x ∈ [L/2 - near_dist, L/2]  左半桌靠近網前 near_dist 公尺
  y ∈ [0, W]                 整個桌寬
  z ∈ [0, box_height]        網高的 3 倍 ≈ 45.75cm

標註順序 (LF -> RF -> RB -> LB)
"""

import json

import cv2
import numpy as np


# ============ 桌球標準尺寸 ============
TABLE_LENGTH = 2.74
TABLE_WIDTH = 1.525
NET_HEIGHT = 0.1525

# ============ 偵測盒參數 ============
NEAR_NET_DIST = 0.40
BOX_HEIGHT_FACTOR = 3
BOX_HEIGHT = BOX_HEIGHT_FACTOR * NET_HEIGHT  # ≈ 0.4575m



# ============ 桌角標註 (順序: LF -> RF -> RB -> LB) ============
class TableCornerSelector:
    CORNER_NAMES = [
        "1. Left-Front  (左前, x=0, y=W)",
        "2. Right-Front (右前, x=L, y=W)",
        "3. Right-Back  (右後, x=L, y=0)",
        "4. Left-Back   (左後, x=0, y=0)",
    ]
    COLORS = [(0, 255, 255), (0, 165, 255), (255, 0, 255), (0, 255, 0)]
    # 使用者實際點選順序：LF -> RF -> RB -> LB
    # 這裡把 x 定義為桌子長邊方向（左 -> 右，274cm），y 定義為桌子短邊方向（後 -> 前，152.5cm）。
    # 注意：影像座標 (0,0) 在左上角；這裡的 world/table 座標是桌面座標：
    #   LB 左後 = (0, 0), RB 右後 = (L, 0), LF 左前 = (0, W), RF 右前 = (L, W)。
    WORLD_COORDS = [
        (0, TABLE_WIDTH, 0),              # LF 左前 = (0, y)
        (TABLE_LENGTH, TABLE_WIDTH, 0),   # RF 右前 = (x, y)
        (TABLE_LENGTH, 0, 0),             # RB 右後 = (x, 0)
        (0, 0, 0),                        # LB 左後 = (0, 0)
    ]

    def __init__(self, frame):
        self.frame = frame.copy()
        self.display = frame.copy()
        self.corners = []
        self.window_name = "Side-View: LF -> RF -> RB -> LB"

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.corners) < 4:
            self.corners.append((x, y))
            self._redraw()
        elif event == cv2.EVENT_MOUSEMOVE and len(self.corners) < 4:
            self._redraw(cursor=(x, y))

    def _redraw(self, cursor=None):
        self.display = self.frame.copy()
        if cursor is not None:
            h, w = self.display.shape[:2]
            cv2.line(self.display, (cursor[0], 0), (cursor[0], h), (200, 200, 200), 1)
            cv2.line(self.display, (0, cursor[1]), (w, cursor[1]), (200, 200, 200), 1)
        for i, pt in enumerate(self.corners):
            cv2.circle(self.display, pt, 8, self.COLORS[i], -1)
            cv2.putText(self.display, f"{i+1}", (pt[0] + 12, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.COLORS[i], 2)
        if len(self.corners) > 1:
            for i in range(len(self.corners) - 1):
                cv2.line(self.display, self.corners[i], self.corners[i + 1],
                         (0, 255, 255), 2)
        if len(self.corners) == 4:
            cv2.line(self.display, self.corners[3], self.corners[0],
                     (0, 255, 255), 2)
        hint = (f"Click: {self.CORNER_NAMES[len(self.corners)]}"
                if len(self.corners) < 4
                else "Done! ENTER=confirm, r=reset, ESC=cancel")
        hint_color = (self.COLORS[len(self.corners)] if len(self.corners) < 4
                      else (0, 255, 255))
        (tw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(self.display, (10, 15), (20 + tw, 50), (0, 0, 0), -1)
        cv2.putText(self.display, hint, (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, hint_color, 2)

    def select(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        self._redraw()
        while True:
            cv2.imshow(self.window_name, self.display)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('r'):
                self.corners = []; self._redraw()
            elif key == 13 and len(self.corners) == 4: break
            elif key == 27:
                self.corners = []; break
        cv2.destroyWindow(self.window_name)
        return self.corners


def get_frame_from_video(video_path, frame_index=0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"無法開啟影片: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = min(frame_index, max(total - 1, 0))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise IOError("無法讀取影片畫面")
    return frame


# ============ 相機姿態 & Homography ============
def estimate_camera_pose(image_corners, frame_shape,
                         camera_matrix=None, dist_coeffs=None):
    h, w = frame_shape[:2]
    if camera_matrix is None:
        f = max(w, h)
        camera_matrix = np.array([[f, 0, w/2.0], [0, f, h/2.0], [0, 0, 1]],
                                 dtype=np.float64)
    if dist_coeffs is None:
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    object_pts = np.array(TableCornerSelector.WORLD_COORDS, dtype=np.float64)
    image_pts = np.array(image_corners, dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(object_pts, image_pts, camera_matrix, dist_coeffs,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP 失敗")
    return rvec, tvec, camera_matrix, dist_coeffs


def project_3d_to_image(points_3d, rvec, tvec, K, D):
    pts = np.array(points_3d, dtype=np.float64).reshape(-1, 1, 3)
    img_pts, _ = cv2.projectPoints(pts, rvec, tvec, K, D)
    return img_pts.reshape(-1, 2)


def compute_homography(image_corners):
    world_pts = np.array(
        [[c[0], c[1]] for c in TableCornerSelector.WORLD_COORDS], dtype=np.float32)
    image_pts = np.array(image_corners, dtype=np.float32)
    H_img2world, _ = cv2.findHomography(image_pts, world_pts)
    H_world2img, _ = cv2.findHomography(world_pts, image_pts)
    return H_img2world, H_world2img


def image_point_to_table_xy(image_xy, H_img2world):
    pt = np.array([[image_xy]], dtype=np.float32)
    world = cv2.perspectiveTransform(pt, H_img2world)
    return float(world[0, 0, 0]), float(world[0, 0, 1])


# ============ 單一 Near Box ============
def build_near_box(near_dist=NEAR_NET_DIST, box_height=BOX_HEIGHT):
    """
    近端網前長方體, 8 個頂點 + bounds。

    座標定義配合使用者點選順序 LF -> RF -> RB -> LB：
      x: 桌子長邊方向，左 -> 右，範圍 [0, TABLE_LENGTH]
      y: 桌子短邊方向，後 -> 前，範圍 [0, TABLE_WIDTH]
      z: 高度

    網線位於 x = TABLE_LENGTH / 2。
    Near Box 是「左半邊靠近網前」的區域：
      x ∈ [L/2 - near_dist, L/2]
      y ∈ [0, W]
      z ∈ [0, box_height]

    頂點順序：
      0~3: 底面 z=0
      4~7: 頂面 z=box_height
    """
    L = TABLE_LENGTH
    W = TABLE_WIDTH
    mid_x = L / 2.0
    x_min = mid_x - near_dist

    verts = np.array([
        [x_min, 0, 0],          [mid_x, 0, 0],
        [mid_x, W, 0],          [x_min, W, 0],
        [x_min, 0, box_height], [mid_x, 0, box_height],
        [mid_x, W, box_height], [x_min, W, box_height],
    ], dtype=np.float64)

    bounds = (x_min, mid_x, 0.0, W, 0.0, box_height)
    return verts, bounds

BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # 底面
    (4, 5), (5, 6), (6, 7), (7, 4),   # 頂面
    (0, 4), (1, 5), (2, 6), (3, 7),   # 垂直邊
]


# ============ 核心區域邏輯 ============
class NearNetRegion:
    def __init__(self, image_corners, frame_shape,
                 near_dist=NEAR_NET_DIST, box_height=BOX_HEIGHT,
                 camera_matrix=None, dist_coeffs=None):
        self.near_dist = near_dist
        self.box_height = box_height
        # x 軸為 TABLE_LENGTH，y 軸為 TABLE_WIDTH
        self.L = TABLE_LENGTH
        self.W = TABLE_WIDTH
        self.mid_x = self.L / 2.0

        self.rvec, self.tvec, self.K, self.D = estimate_camera_pose(
            image_corners, frame_shape, camera_matrix, dist_coeffs)
        self.H_img2world, self.H_world2img = compute_homography(image_corners)

        self.box_3d, self.box_bounds = build_near_box(near_dist, box_height)

        # Project the 3D near box. With only four coplanar table corners, solvePnP may
        # choose the opposite plane normal, making positive z appear downward in the
        # image. Keep the table x-y position fixed, but store the chosen z direction so
        # the box, net, reference line, printed vertices, and bounds stay consistent.
        self.z_sign = 1.0
        self.box_2d = project_3d_to_image(
            self.box_3d, self.rvec, self.tvec, self.K, self.D)
        bottom_y = float(np.mean(self.box_2d[:4, 1]))
        top_y = float(np.mean(self.box_2d[4:, 1]))
        if top_y > bottom_y:
            self.z_sign = -1.0
            self.box_3d = self.box_3d.copy()
            self.box_3d[4:, 2] = self.z_sign * np.abs(self.box_3d[4:, 2])
            self.box_bounds = (
                self.box_bounds[0], self.box_bounds[1],
                self.box_bounds[2], self.box_bounds[3],
                min(0.0, self.z_sign * self.box_height),
                max(0.0, self.z_sign * self.box_height),
            )
            self.box_2d = project_3d_to_image(
                self.box_3d, self.rvec, self.tvec, self.K, self.D)

        self.box_hull = cv2.convexHull(
            self.box_2d.astype(np.int32)).reshape(-1, 2)

        self.table_corners_2d = np.array(image_corners, dtype=np.int32)
    VERTEX_NAMES = [
        "near_back_bottom",   # 0: x=L/2-near, y=0, z=0
        "net_back_bottom",    # 1: x=L/2,      y=0, z=0
        "net_front_bottom",   # 2: x=L/2,      y=W, z=0
        "near_front_bottom",  # 3: x=L/2-near, y=W, z=0
        "near_back_top",      # 4
        "net_back_top",       # 5
        "net_front_top",      # 6
        "near_front_top",     # 7
    ]
    def get_box_vertices_3d(self):
        """回傳 8 個頂點的 3D 世界座標 (單位: 公尺)
        
        Returns:
            ndarray shape (8, 3), dtype=float64
            順序依照 VERTEX_NAMES
        """
        return self.box_3d.copy()
    def get_box_vertices_2d(self, as_int=False):
        """回傳 8 個頂點投影到影像上的 2D 座標 (單位: pixel)
        
        Args:
            as_int: True 則回傳 int (方便 cv2 畫圖), False 回傳 float
        
        Returns:
            ndarray shape (8, 2)
        """
        if as_int:
            return self.box_2d.astype(np.int32).copy()
        return self.box_2d.copy()
    def get_box_vertices_dict(self, as_int=False):
        """回傳 dict 格式, 含 3D 與 2D 座標, 以名稱為 key
        
        Returns:
            {
                "near_bottom_left":  {"world": (x,y,z), "image": (u,v)},
                ...
            }
        """
        pts2d = self.get_box_vertices_2d(as_int=as_int)
        result = {}
        for i, name in enumerate(self.VERTEX_NAMES):
            result[name] = {
                "world": tuple(self.box_3d[i].tolist()),
                "image": tuple(pts2d[i].tolist()),
            }
        return result
    def get_box_bounds(self):
        """回傳 box 的 AABB 範圍
        
        Returns:
            dict: {"x": (min, max), "y": (min, max), "z": (min, max)}
        """
        b = self.box_bounds
        return {
            "x": (b[0], b[1]),
            "y": (b[2], b[3]),
            "z": (b[4], b[5]),
        }
    def get_box_image_polygon(self):
        """回傳 box 在影像上的外輪廓 (convex hull) 頂點
        
        可直接用於 cv2.pointPolygonTest / cv2.fillPoly
        
        Returns:
            ndarray shape (N, 2), dtype=int32, N <= 8
        """
        return self.box_hull.copy()
    def get_box_faces_3d(self):
        """回傳 6 個面的頂點索引 (可用於計算法向量 / 繪製各面)
        
        Returns:
            dict: 面名稱 -> 4 個頂點索引 (逆時針, 朝外為正)
        """
        return {
            "bottom":     [0, 1, 2, 3],    # z=0
            "top":        [4, 5, 6, 7],    # z=box_height
            "near_side":  [0, 3, 7, 4],    # x=L/2-near
            "net_side":   [1, 2, 6, 5],    # x=L/2
            "back":       [0, 1, 5, 4],    # y=0
            "front":      [3, 2, 6, 7],    # y=W
        }

# ============ 視覺化 ============
def draw_box(img, pts_2d, edges, color, fill_alpha=0.2, thickness=2):
    pts = pts_2d.astype(np.int32)
    if fill_alpha > 0:
        overlay = img.copy()
        cv2.fillPoly(overlay, [cv2.convexHull(pts)], color)
        cv2.addWeighted(overlay, fill_alpha, img, 1 - fill_alpha, 0, img)
    for i, j in edges:
        cv2.line(img, tuple(pts[i]), tuple(pts[j]), color, thickness, cv2.LINE_AA)


BOX_COLOR = (80, 180, 255)   # 橘色


def visualize_result(frame, corners, region: NearNetRegion, save_path=None):
    vis = frame.copy()

    # 桌面輪廓
    pts = np.array(corners, np.int32)
    cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
    labels = ["1.LF", "2.RF", "3.RB", "4.LB"]
    for p, lbl in zip(corners, labels):
        cv2.circle(vis, p, 6, (0, 255, 0), -1)
        cv2.putText(vis, lbl, (p[0] + 8, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # Near Box
    draw_box(vis, region.box_2d, BOX_EDGES, BOX_COLOR,
             fill_alpha=0.2, thickness=2)
    center_3d = region.box_3d.mean(axis=0, keepdims=True)
    center_2d = project_3d_to_image(center_3d, region.rvec, region.tvec,
                                     region.K, region.D)[0].astype(int)
    cv2.putText(vis, "NEAR BOX", tuple(center_2d - np.array([40, 0])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, BOX_COLOR, 2)

    # 網 (實體 h=15.25cm)：x = L/2，沿 y=0..W
    # z 要跟 Near Box 使用相同方向，否則 box 翻轉後網線會和 box 對不起來。
    L, W = region.L, region.W
    z_net = region.z_sign * NET_HEIGHT
    z_box = region.z_sign * region.box_height
    net_3d = np.array([
        [L/2, 0, 0], [L/2, W, 0],
        [L/2, W, z_net], [L/2, 0, z_net],
    ])
    net_2d = project_3d_to_image(net_3d, region.rvec, region.tvec,
                                 region.K, region.D).astype(np.int32)
    cv2.polylines(vis, [net_2d], True, (0, 255, 255), 2)

    # 盒子頂部的高度參考線
    top_line_3d = np.array([
        [L/2, 0, z_box], [L/2, W, z_box],
    ])
    top_line_2d = project_3d_to_image(top_line_3d, region.rvec, region.tvec,
                                       region.K, region.D).astype(np.int32)
    cv2.line(vis, tuple(top_line_2d[0]), tuple(top_line_2d[1]),
             (0, 200, 255), 1, cv2.LINE_AA)

    # 資訊文字
    info = (f"Near Box: depth={region.near_dist*100:.0f}cm, "
            f"height={region.box_height*100:.1f}cm "
            f"({region.box_height/NET_HEIGHT:.1f}x net)")
    cv2.putText(vis, info, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if save_path:
        cv2.imwrite(save_path, vis)
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Result", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return vis


# ============ 桌角 JSON 存取 ============
def save_table_corners(corners, save_path):
    """Save manually selected table corners. Corner order is LF -> RF -> RB -> LB."""
    data = {
        "corner_order": "LF_RF_RB_LB",
        "corners": [[float(x), float(y)] for x, y in corners],
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_table_corners(json_path):
    """Load table corners saved by save_table_corners().

    Current expected order is LF -> RF -> RB -> LB.
    Older json files may still say NR_FR_FL_NL because the label text was wrong;
    this loader accepts them and treats the stored corner list as the user's actual
    clicked order: LF -> RF -> RB -> LB.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    corner_order = data.get("corner_order")
    if corner_order not in ("LF_RF_RB_LB", "NR_FR_FL_NL", None):
        raise ValueError(
            "helper_table json corner_order must be LF_RF_RB_LB. "
            f"Got: {corner_order}"
        )

    corners = data.get("corners", [])
    if len(corners) != 4:
        raise ValueError("helper_table json must contain 4 corners")

    return [(float(x), float(y)) for x, y in corners]


# ============ 主流程 ============
def process_video(video_path, frame_index=0,
                  near_dist=NEAR_NET_DIST, box_height=BOX_HEIGHT,
                  visualize=True, save_vis_path=None,
                  save_corners_path=None,
                  camera_matrix=None, dist_coeffs=None,
                  print_vertices=True):
    frame = get_frame_from_video(video_path, frame_index)
    selector = TableCornerSelector(frame)
    corners = selector.select()
    if len(corners) != 4:
        raise RuntimeError("未完成 4 個角落的標註")
    if save_corners_path is not None:
        save_table_corners(corners, save_corners_path)
        print(f"[helper_table] saved corners json: {save_corners_path}")
    region = NearNetRegion(corners, frame.shape, near_dist, box_height,
                           camera_matrix, dist_coeffs)
    print(f"\n=== 標註順序 (LF -> RF -> RB -> LB) ===")
    for i, (pt, name) in enumerate(zip(corners, TableCornerSelector.CORNER_NAMES)):
        wx, wy, wz = TableCornerSelector.WORLD_COORDS[i]
        print(f"  {name}: image={pt} -> world=({wx:.3f}, {wy:.3f}, {wz:.3f})")
    b = region.box_bounds
    print(f"\n=== Near Box 參數 ===")
    print(f"  depth (x):  {region.near_dist*100:.1f} cm")
    print(f"  height (z): {region.box_height*100:.1f} cm "
          f"({region.box_height/NET_HEIGHT:.2f}x 網高)")
    print(f"  z direction used for drawing: {'positive' if region.z_sign > 0 else 'negative'}")
    print(f"  bounds: x∈[{b[0]:.3f}, {b[1]:.3f}]  "
          f"y∈[{b[2]:.3f}, {b[3]:.3f}]  z∈[{b[4]:.3f}, {b[5]:.3f}]")
    # ============ 端點座標輸出 ============
    if print_vertices:
        verts_dict = region.get_box_vertices_dict(as_int=True)
        print(f"\n=== Near Box 8 個端點座標 ===")
        print(f"{'idx':>3} {'name':<20} "
              f"{'world (x,y,z) [m]':<30} {'image (u,v) [px]':<20}")
        print("-" * 78)
        for i, name in enumerate(region.VERTEX_NAMES):
            v = verts_dict[name]
            w_str = f"({v['world'][0]:.3f},{v['world'][1]:.3f},{v['world'][2]:.3f})"
            i_str = f"({v['image'][0]},{v['image'][1]})"
            print(f"{i:>3} {name:<20} {w_str:<30} {i_str:<20}")
    if visualize:
        visualize_result(frame, corners, region, save_vis_path)
    return corners, region


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--near", type=float, default=NEAR_NET_DIST,
                        help="網前距離 (m), 預設 0.40")
    parser.add_argument("--box_h", type=float, default=BOX_HEIGHT,
                        help="盒子高度 (m), 預設 3x 網高 ≈ 0.4575")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--save_corners", type=str, default=None)
    args = parser.parse_args()
    process_video(args.video, args.frame,
                  near_dist=args.near, box_height=args.box_h,
                  visualize=True, save_vis_path=args.save,
                  save_corners_path=args.save_corners)
