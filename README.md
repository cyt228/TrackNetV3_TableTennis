# TrackNetV3 Table Tennis

本專案是針對桌球影片改寫的 TrackNetV3 pipeline，主要目標是從影片中偵測桌球位置，輸出球軌跡 CSV，並進一步分析每一球的 stroke、網前球速、落點區域與視覺化結果。

整體流程分成兩個部分：

1. **Ball Tracking**：使用 TrackNetV3 / InpaintNet 從影片產生逐 frame 的球座標。
2. **Speed Analysis**：使用球桌四點、near-net box、球軌跡與影片 FPS，計算每一球的網前速度與落點資訊。

---

## 1. 專案流程總覽

```text
原始影片 MP4
    ↓
helper_table.py
手動標記球桌四點，產生 helper_table.json
    ↓
predict.py
TrackNetV3 偵測球點，InpaintNet 補短暫漏點
    ↓
stroke_zone_analysis.py
切分 stroke，計算 net-zone speed，分析 bounce / landing
    ↓
plot_speed_bounce.py 或 plot_speed.py
輸出速度折線圖與統計圖
```

目前建議使用根目錄的 shell script 執行完整流程，例如 `R3.sh`、`R4.sh`、`L3.sh` 等。這些腳本會依序執行 table 標記、TrackNet inference、stroke / speed analysis 與速度圖輸出。

---

## 2. 環境安裝

### Clone repository

```bash
git clone https://github.com/wasn-lab/TrackNetV3_TableTennis.git 
cd TrackNetV3_TableTennis
```

### 建立 conda 環境

```bash
conda create -n tracknetV3 python=3.11 
conda activate tracknetV3
```

### 安裝套件

```bash
pip install -r requirements.txt
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

---

## 3. 模型權重

本專案需要兩個模型權重：

| 權重 | 用途 |
|---|---|
| `TrackNet_best.pt` | 偵測每個 frame 的球位置 |
| `InpaintNet_best.pt` | 對短暫漏偵測的軌跡做補點 |

預設範例路徑如下：

```text
exp/TrackNet_best.pt
exp/InpaintNet_best.pt
```

模型下載連結：

[模型下載](https://1drv.ms/u/c/ab3b33d5410e04f3/IQCwzwpuGP6pSpgw0VyyRSCzAa4jTyVFYiFWUgSd8gPeCf0?e=hWQh7G)

下載後請將權重放到 `exp/` 資料夾，或在執行指令中自行指定權重路徑。

---

## 4. 單支影片完整流程

以下以 `C0081.MP4` 為例。

---

### Step 1：標記球桌四點

```bash
python speed_analysis/helper_table.py --video /path/to/C0081.MP4 --frame 2000 --save /path/to/C0081_helper_table.png --save_corners /path/to/C0081_helper_table.json
```

點選順序為：

```text
LF -> RF -> RB -> LB
左前 -> 右前 -> 右後 -> 左後
```

輸出：

| 檔案 | 說明 |
|---|---|
| `C0081_helper_table.png` | 標記結果確認圖 |
| `C0081_helper_table.json` | 後續速度與落點分析使用的球桌四點 |

注意：若更換相機角度、影片解析度、影片裁切方式或球桌位置，必須重新標記 helper table。

---

### Step 2：TrackNetV3 預測球軌跡

一般建議指令：

```bash
python predict.py --video_file /path/to/C0081.MP4 --tracknet_file exp/TrackNet_best.pt --inpaintnet_file exp/InpaintNet_best.pt --save_dir /path/to/output --large_video --output_video --eval_mode weight
```

如果影片輸出遇到 NVENC 錯誤，可以改用 CPU 編碼：

```bash
python predict.py --video_file /path/to/C0081.MP4 --tracknet_file exp/TrackNet_best.pt --inpaintnet_file exp/InpaintNet_best.pt --save_dir /path/to/output --large_video --output_video --eval_mode weight --video_codec libx264
```

如果只需要 CSV，不需要影片，可以拿掉 `--output_video`：

```bash
python predict.py --video_file /path/to/C0081.MP4 --tracknet_file exp/TrackNet_best.pt --inpaintnet_file exp/InpaintNet_best.pt --save_dir /path/to/output --large_video --eval_mode weight
```

輸出：

| 檔案 | 說明 |
|---|---|
| `C0081_ball.csv` | 每個 frame 的球座標 |
| `C0081_predict.mp4` | 畫上球軌跡的預測影片，只有加 `--output_video` 時才會輸出 |

`*_ball.csv` 主要欄位：

| 欄位 | 說明 |
|---|---|
| `Frame` | frame 編號 |
| `Visibility` | 是否偵測到球，`1` 代表有球，`0` 代表無球 |
| `X` | 球心 x 座標 |
| `Y` | 球心 y 座標 |
| `Inpaint_Mask` | 是否由 InpaintNet 補點 |

---

### Step 3：stroke、網前速度與落點分析

目前建議使用 height-plane correction 版本計算速度：

```bash
python speed_analysis/stroke_zone_analysis.py --video_file /path/to/C0081.MP4 --ball_csv /path/to/output/C0081_ball.csv --save_dir /path/to/output --helper_table_json /path/to/output/C0081_helper_table.json --save_video --use_height_plane_scale --plane_height_cm 26 --camera_focal_scale 1.0
```

如果影片輸出遇到 NVENC 問題，可以改用 CPU 編碼：

```bash
python speed_analysis/stroke_zone_analysis.py --video_file /path/to/C0081.MP4 --ball_csv /path/to/output/C0081_ball.csv --save_dir /path/to/output --helper_table_json /path/to/output/C0081_helper_table.json --save_video --video_codec libx264 --use_height_plane_scale --plane_height_cm 26 --camera_focal_scale 1.0
```

如果只要 CSV 與圖，不需要影片：

```bash
python speed_analysis/stroke_zone_analysis.py --ball_csv /path/to/output/C0081_ball.csv --save_dir /path/to/output --helper_table_json /path/to/output/C0081_helper_table.json --fps 120 --frame_w 1920 --frame_h 1080 --use_height_plane_scale --plane_height_cm 26 --camera_focal_scale 1.0
```

輸出：

| 檔案 | 說明 |
|---|---|
| `C0081_stroke_zone.csv` | 每個 stroke 的主要結果，包含速度、bounce、落點區域 |
| `C0081_net_zone_speed_detail.csv` | 每個 frame / segment 的速度細節 |
| `landing_detail.csv` | bounce / landing 詳細資料 |
| `zone_stats.csv` | 落點區域統計 |
| `landing_heatmap.png` | 落點熱力圖 |
| `landing_zones.png` | 落點區域散佈圖 |
| `C0081_stroke_zone_visualize.mp4` | 視覺化影片，只有加 `--save_video` 時才會輸出 |

---

### Step 4：輸出速度折線圖

依球種 / target mode 輸出速度與 bounce 圖：

```bash
python speed_analysis/plot_speed_bounce.py --input /path/to/output/C0081_stroke_zone.csv --target_mode r12
```

或使用一般速度折線圖：

```bash
python speed_analysis/plot_speed.py --input /path/to/output/C0081_stroke_zone.csv --speed net_zone_max_speed_kmh
```

---

## 5. 自動化腳本

根目錄目前主要使用 `R3.sh` ~ `R8.sh` 與 `L3.sh` ~ `L8.sh` 進行不同球種的自動化流程。  
這些腳本通常會整合影片路徑設定、TrackNet 預測、helper table、速度分析與輸出結果。

| 腳本 | 球種編號 | 球種說明 | 用途 |
|---|---:|---|---|
| `R3.sh` | 3 | 正手位斜線 | 右側 / 右手方向的正手位斜線流程 |
| `R4.sh` | 4 | 正手位直線 | 右側 / 右手方向的正手位直線流程 |
| `R5.sh` | 5 | 反手位斜線 | 右側 / 右手方向的反手位斜線流程 |
| `R6.sh` | 6 | 反手位直線 | 右側 / 右手方向的反手位直線流程 |
| `R7.sh` | 7 | 交換斜線 | 右側 / 右手方向的交換斜線流程 |
| `R8.sh` | 8 | 交換直線 | 右側 / 右手方向的交換直線流程 |
| `L3.sh` | 3 | 正手位斜線 | 左側 / 左手方向的正手位斜線流程 |
| `L4.sh` | 4 | 正手位直線 | 左側 / 左手方向的正手位直線流程 |
| `L5.sh` | 5 | 反手位斜線 | 左側 / 左手方向的反手位斜線流程 |
| `L6.sh` | 6 | 反手位直線 | 左側 / 左手方向的反手位直線流程 |
| `L7.sh` | 7 | 交換斜線 | 左側 / 左手方向的交換斜線流程 |
| `L8.sh` | 8 | 交換直線 | 左側 / 左手方向的交換直線流程 |

球種編號對應如下：

| 編號 | 球種 |
|---:|---|
| 3 | 正手位斜線 |
| 4 | 正手位直線 |
| 5 | 反手位斜線 |
| 6 | 反手位直線 |
| 7 | 交換斜線 |
| 8 | 交換直線 |

執行前請先打開對應腳本，修改影片路徑、輸出資料夾與相關模型路徑：

```bash
VIDEO_FILE="/path/to/video.MP4" SAVE_DIR="/path/to/output"
```

例如要執行右側 / 右手方向的正手位斜線流程：

```bash
bash R3.sh
```

例如要執行左側 / 左手方向的交換直線流程：

```bash
bash L8.sh
```

> Note: 早期版本中的 `run_one.sh`、`run_all.sh`、`run_by_folder.sh` 目前已不建議使用，可能與目前流程或參數不相容。若需要批次處理，建議以目前可用的 `R3.sh` ~ `R8.sh`、`L3.sh` ~ `L8.sh` 為主，或依照新版 README 中的單行指令自行建立新的 shell script。

---

## 6. `predict.py` 主要參數

`predict.py` 是球點預測主程式，負責從影片產生 `*_ball.csv`，也可以輸出球點預測影片。

| 參數 | 用法 | 說明 |
|---|---|---|
| `--video_file` | `--video_file /path/to/video.MP4` | 單支影片路徑。 |
| `--video_dir` | `--video_dir /path/to/videos` | 整個資料夾批次預測，會遞迴搜尋 `.mp4` / `.MP4`。 |
| `--tracknet_file` | `--tracknet_file exp/TrackNet_best.pt` | TrackNet 權重檔。 |
| `--inpaintnet_file` | `--inpaintnet_file exp/InpaintNet_best.pt` | InpaintNet 權重檔。不填則只跑 TrackNet。 |
| `--save_dir` | `--save_dir /path/to/output` | 輸出資料夾。 |
| `--large_video` | `--large_video` | 大影片模式，使用串流讀取，避免一次載入整部影片。 |
| `--output_video` | `--output_video` | 輸出畫上球軌跡的影片。 |
| `--eval_mode` | `--eval_mode weight` | temporal ensemble 模式，可選 `weight`、`average`、`nonoverlap`。 |
| `--video_codec` | `--video_codec libx264` | 影片編碼器，可選 `h264_nvenc` 或 `libx264`。 |
| `--max_sample_num` | `--max_sample_num 1000` | 產生 median background 時最多取樣 frame 數。 |
| `--video_range` | `--video_range 10,20` | 指定產生 background 的秒數範圍，例如 `10,20`。 |

`eval_mode` 建議：

| 模式 | 說明 |
|---|---|
| `weight` | 最穩定，建議預設使用 |
| `average` | 速度與穩定性折衷 |
| `nonoverlap` | 較快，但穩定度可能較低 |

---

## 7. `stroke_zone_analysis.py` 主要參數

`stroke_zone_analysis.py` 是目前速度分析的主程式，負責讀取 `ball.csv`、helper table 與影片資訊，進行 stroke 切分、網前速度計算、bounce / landing 分析與視覺化輸出。

常用指令如下：

```bash
python speed_analysis/stroke_zone_analysis.py --video_file /path/to/C0081.MP4 --ball_csv /path/to/output/C0081_ball.csv --save_dir /path/to/output --helper_table_json /path/to/output/C0081_helper_table.json --save_video --use_height_plane_scale --plane_height_cm 26 --camera_focal_scale 1.0
```

### 7.1 基本輸入與輸出參數

| 參數 | 用法 | 說明 |
|---|---|---|
| `--video_file` | `--video_file /path/to/video.MP4` | 輸入影片路徑。若有提供影片，程式會從影片讀取 FPS、解析度，並可輸出 overlay video。 |
| `--ball_csv` | `--ball_csv /path/to/C0081_ball.csv` | TrackNetV3 / InpaintNet 輸出的球點 CSV。這是速度分析最主要的輸入。 |
| `--save_dir` | `--save_dir /path/to/output` | 分析結果輸出資料夾。summary CSV、detail CSV、plot、overlay video 都會存在這裡。 |
| `--helper_table_json` | `--helper_table_json /path/to/C0081_helper_table.json` | `helper_table.py` 產生的球桌四點與 near-net box 資訊。速度比例、落點分析、near-net 判斷都會用到它。 |

### 7.2 影片輸出參數

| 參數 | 用法 | 說明 |
|---|---|---|
| `--save_video` | `--save_video` | 輸出速度分析 overlay video。會畫出球桌、near-net box、stroke、最大速度段與 bounce / landing 資訊。 |
| `--video_codec` | `--video_codec libx264` | 指定輸出影片編碼器。預設可能使用 `h264_nvenc`，若 GPU 編碼失敗可改成 `libx264`。 |

如果遇到 NVENC 錯誤，可以改用：

```bash
python speed_analysis/stroke_zone_analysis.py --video_file /path/to/C0081.MP4 --ball_csv /path/to/output/C0081_ball.csv --save_dir /path/to/output --helper_table_json /path/to/output/C0081_helper_table.json --save_video --video_codec libx264 --use_height_plane_scale --plane_height_cm 26 --camera_focal_scale 1.0
```

若只需要 CSV 與圖，不需要 overlay video，可以不要加 `--save_video`。

### 7.3 CSV-only 模式參數

如果沒有提供 `--video_file`，也可以只用 `ball.csv` 跑速度分析。  
但這時程式無法從影片讀取 FPS 與解析度，因此需要手動指定：

| 參數 | 用法 | 說明 |
|---|---|---|
| `--fps` | `--fps 120` | 指定影片 FPS。速度計算會直接使用這個值。FPS 錯誤會造成速度整體錯誤。 |
| `--frame_w` | `--frame_w 1920` | 指定影片寬度。沒有影片時用於幾何與視覺化相關計算。 |
| `--frame_h` | `--frame_h 1080` | 指定影片高度。沒有影片時用於幾何與視覺化相關計算。 |

CSV-only 範例：

```bash
python speed_analysis/stroke_zone_analysis.py --ball_csv /path/to/output/C0081_ball.csv --save_dir /path/to/output --helper_table_json /path/to/output/C0081_helper_table.json --fps 120 --frame_w 1920 --frame_h 1080 --use_height_plane_scale --plane_height_cm 26 --camera_focal_scale 1.0
```

注意：CSV-only 模式無法輸出 overlay video，因為沒有原始影片可以畫圖。

### 7.4 球高平面補償參數

目前建議開啟球高平面補償：

| 參數 | 用法 | 說明 |
|---|---|---|
| `--use_height_plane_scale` | `--use_height_plane_scale` | 啟用球高平面補償。系統會用簡化 camera geometry 估計球位於桌面上方時造成的比例差。 |
| `--plane_height_cm` | `--plane_height_cm 26` | 假設球所在的高度平面，單位為 cm。目前建議值為 `26`。 |
| `--camera_focal_scale` | `--camera_focal_scale 1.0` | 控制高度平面補償強度。目前建議值為 `1.0`。 |

目前速度分析不是只用桌面平面計算，而是會考慮：

1. 桌面四角點。
2. 近端 / 遠端桌邊寬度。
3. 球所在位置的深度比例。
4. 球中心 y offset。
5. 球高平面補償。
6. 最後 GT scale 校正。

`--use_height_plane_scale` 開啟後，detail CSV 中會出現或使用以下欄位協助檢查補償結果：

| 欄位 | 說明 |
|---|---|
| `plane_scale_ratio` | 球高平面補償比例 |
| `best_speed_plane_kmh` | 套用球高平面補償後的速度 |
| `best_speed_raw_before_gt_scale` | 套用最後 GT scale 前的速度 |
| `orange_blue_ratio_x` | 高度平面與桌面平面在 x 方向的比例差 |
| `orange_blue_ratio_y` | 高度平面與桌面平面在 y 方向的比例差 |
| `orange_blue_ratio_len` | 高度平面與桌面平面的整體比例差 |

### 7.5 深度補償相關欄位

除了球高平面補償外，系統也會根據球在畫面中的 y 位置估計桌面深度比例。

桌球桌在影像中會有透視變形：

```text
遠端桌邊比較窄
近端桌邊比較寬
```

因此同樣的 pixel 位移，在遠端與近端代表的真實距離不同。

detail CSV 中可以檢查以下欄位：

| 欄位 | 說明 |
|---|---|
| `blue_depth_ratio` | 根據桌面平面估計出的深度補償比例 |
| `blue_local_width_px` | 球所在 y 位置附近的桌面寬度 |
| `far_width_px` | 遠端桌邊寬度 |
| `near_width_px` | 近端桌邊寬度 |
| `avg_width_px` | 遠近桌邊平均寬度 |
| `avg_px_per_cm` | 平均 pixel/cm |
| `speed_mid_y` | 速度段中間位置的 y |
| `speed_table_y` | 經過 y offset 修正後，用來估計比例的 y 位置 |

如果速度忽高忽低，可以先檢查這些欄位是否有異常。

### 7.6 stroke 切分相關參數

以下參數會影響 stroke 如何被切分：

| 參數 | 用法 | 說明 |
|---|---|---|
| `--min_left_segments` | `--min_left_segments 5` | 偵測 stroke start 前，需要有幾段穩定往左移動的軌跡。 |
| `--min_candidate_frames` | `--min_candidate_frames 50` | 一段有效 stroke 至少需要的 frame 數。太高可能漏球，太低可能把雜訊當 stroke。 |
| `--min_no_hit_candidate_frames` | `--min_no_hit_candidate_frames 20` | no-hit 候選片段的最短 frame 數。 |
| `--max_step_th` | `--max_step_th 300` | 相鄰球點最大允許跳動距離。太小可能切太碎，太大可能把誤抓點接進來。 |
| `--max_abs_dy_th` | `--max_abs_dy_th 45` | 找穩定移動段時，y 方向允許的最大變化。 |
| `--left_half_ratio` | `--left_half_ratio 0.35` | no-hit 或左側區域判斷比例。 |
| `--right_side_ratio` | `--right_side_ratio 0.5` | 有效 stroke 需要到達的右側畫面比例。 |

如果 summary CSV 中 stroke 數量明顯太少或太多，通常需要檢查這一組參數與 `ball.csv` 的球點品質。

### 7.7 near-net box 相關參數

near-net box 是目前計算網前最大速度的重要區域。  
系統不是直接取整段球路最大速度，而是優先取 near-net box 附近的最大合理速度。

| 參數 | 用法 | 說明 |
|---|---|---|
| `--near_dist` | `--near_dist 35` | near-net box 往網前方向延伸的距離設定。 |
| `--box_height` | `--box_height 180` | near-net box 的高度設定。 |

如果 near-net box 太小，可能會導致：

```text
no_ball_in_net_zone
```

如果 near-net box 太大，可能會包含太多不相關球點，讓速度結果不穩。

檢查方式：

1. 打開 `*_stroke_zone_visualize.mp4`。
2. 看 near-net box 是否畫在合理位置。
3. 看最大速度段是否真的落在 near-net box 附近。

### 7.8 主要輸出欄位

`stroke_zone_analysis.py` 會輸出 summary CSV 與 detail CSV。

summary CSV 中主要看：

| 欄位 | 說明 |
|---|---|
| `stroke_id` | stroke 編號 |
| `frame_start` | stroke 起始 frame |
| `frame_end` | stroke 結束 frame |
| `bounce_frame` | bounce frame |
| `net_zone_max_speed_kmh` | near-net box 附近最大速度 |
| `net_zone_max_speed_type` | 最大速度使用的方法，例如 `1f`、`2f` 或 `c2f` |
| `net_zone_max_speed_start_frame` | 最大速度段起始 frame |
| `net_zone_max_speed_end_frame` | 最大速度段結束 frame |
| `zone_label` | 落點區域 |
| `in_table` | 落點是否在桌面內 |
| `valid` | 是否為有效 stroke |
| `note` | 額外註記，例如 `net_stop`、`no_ball_in_net_zone` |

detail CSV 中主要用來 debug：

| 欄位 | 說明 |
|---|---|
| `speed_1f_kmh` | 相鄰 1 frame 速度 |
| `speed_2f_kmh` | 間隔 2 frames 速度 |
| `speed_c2f_kmh` | centered 2-frame 速度 |
| `best_speed_kmh` | 當前 frame 選出的最佳速度 |
| `best_speed_type` | 速度類型 |
| `best_speed_raw_before_gt_scale` | 最後 GT scale 前的速度 |
| `sx_cm_per_px_used` | x 方向使用的 cm/pixel |
| `sy_cm_per_px_used` | y 方向使用的 cm/pixel |
| `blue_depth_ratio` | 深度補償比例 |
| `plane_scale_ratio` | 球高平面補償比例 |
| `in_net` | 是否在 near-net box 內 |
| `use_for_net_max` | 是否可用於 net-zone max speed |

### 7.9 建議使用方式

一般情況建議使用：

```bash
python speed_analysis/stroke_zone_analysis.py --video_file /path/to/C0081.MP4 --ball_csv /path/to/output/C0081_ball.csv --save_dir /path/to/output --helper_table_json /path/to/output/C0081_helper_table.json --save_video --use_height_plane_scale --plane_height_cm 26 --camera_focal_scale 1.0
```

如果輸出影片失敗，改用：

```bash
python speed_analysis/stroke_zone_analysis.py --video_file /path/to/C0081.MP4 --ball_csv /path/to/output/C0081_ball.csv --save_dir /path/to/output --helper_table_json /path/to/output/C0081_helper_table.json --save_video --video_codec libx264 --use_height_plane_scale --plane_height_cm 26 --camera_focal_scale 1.0
```

如果只跑 CSV，不需要影片：

```bash
python speed_analysis/stroke_zone_analysis.py --ball_csv /path/to/output/C0081_ball.csv --save_dir /path/to/output --helper_table_json /path/to/output/C0081_helper_table.json --fps 120 --frame_w 1920 --frame_h 1080 --use_height_plane_scale --plane_height_cm 26 --camera_focal_scale 1.0
```

---

## 8. 目前核心修改重點

### 8.1 多候選球點選擇

TrackNet heatmap 可能同時產生多個候選點，例如真球、殘影、反光或背景白點。本專案在 `predict.py` / `test.py` 中加入 candidate selection，會根據最近的歷史軌跡、候選點面積、跳點距離與方向延續性選出較合理的球點。

---

### 8.2 Inpaint mask 補點策略

不是所有 `Visibility = 0` 都會補點。系統只補「前後都有合理球點」的短暫缺失片段，避免把已經飛出畫面或打飛的球強行補回軌跡。

---

### 8.3 大影片讀取與輸出

`--large_video` 會使用串流方式讀影片，降低記憶體使用量。輸出影片改用 FFmpeg writer，支援：

```text
h264_nvenc：NVIDIA GPU 硬體編碼
libx264：CPU 編碼，相容性較高
```

---

### 8.4 速度分析校正

目前速度分析不再只使用固定全域 pixel-to-cm 比例，而是根據 helper table 的球桌四點與球當下位置估計局部比例。若啟用 `--use_height_plane_scale`，會再用簡化 camera geometry 建立 raised plane，估計球離桌面高度造成的 scale ratio。

目前速度估計主要考慮：

1. 桌面四角點。
2. 近端 / 遠端桌邊寬度。
3. 球所在位置的深度比例。
4. 球中心 y offset。
5. height-plane correction。
6. 最後 GT scale。

---

## 9. 檔案說明

| 檔案 / 資料夾 | 用途 |
|---|---|
| `predict.py` | TrackNetV3 inference 主程式，輸出 ball CSV 與預測影片 |
| `test.py` | heatmap 後處理、candidate selection、ensemble、inpaint mask 等邏輯 |
| `dataset.py` | dataset 與影片讀取流程，包含大影片串流讀取 |
| `model.py` | TrackNet 與 InpaintNet model 定義 |
| `train.py` | 訓練 TrackNet / InpaintNet，基本沿用原始 TrackNetV3 |
| `generate_mask_data.py` | 產生 InpaintNet 訓練用 mask data |
| `utils/general.py` | 通用工具函式，包含 CSV 輸出、影片輸出、模型建立等 |
| `speed_analysis/helper_table.py` | 手動標記球桌四點與 near-net box |
| `speed_analysis/stroke_zone_analysis.py` | stroke、速度、落點分析主程式 |
| `speed_analysis/stroke_analysis.py` | stroke 切分邏輯 |
| `speed_analysis/bounce_landing_analysis.py` | bounce frame 與落點區域分析 |
| `speed_analysis/plot_speed.py` | 速度折線圖 |
| `speed_analysis/plot_speed_bounce.py` | 依球種 / target mode 輸出速度與 bounce 圖 |

---

## 10. 常見問題

### 10.1 `h264_nvenc` 失敗怎麼辦？

這通常是 FFmpeg / GPU 編碼問題，不是模型 inference 失敗。可以改用 CPU 編碼：

```bash
--video_codec libx264
```

或是不輸出影片，只輸出 CSV。

---

### 10.2 速度看起來偏快或偏慢怎麼辦？

目前速度最後有一個整體校正比例 `SPEED_GT_SCALE_FACTOR`，用來根據發球機或人工 GT 速度做整體修正。若使用新的 GT 校正結果，可以在 `speed_analysis/stroke_zone_analysis.py` 中調整這個值。

但在調整前，請先確認：

1. `ball.csv` 沒有明顯誤抓。
2. 影片 FPS 正確。
3. helper table 四角點正確。
4. near-net box 位置合理。
5. 最大速度段出現在合理 frame。

---

### 10.3 換相機角度可以直接用嗎？

不建議直接沿用。每次換相機角度、球桌位置或影片解析度，都應該重新執行 `helper_table.py` 標記球桌四點，必要時也要重新調整 near-net box、stroke 判斷與速度校正參數。

---

### 10.4 現在是 2D 還是 3D 速度？

目前仍是以單相機估計的 2D / pseudo-3D 校正速度。`--use_height_plane_scale` 會用簡化 camera model 估計高度平面的比例差，但它不是完整雙相機 3D 重建。

---

## 11. 建議檢查順序

每次換新影片時，建議照以下順序確認：

1. 檢查 `*_helper_table.png` 的四個球桌角點是否正確。
2. 檢查 `*_predict.mp4` 或 `*_ball.csv`，確認球軌跡是否追到真球。
3. 檢查 `*_stroke_zone_visualize.mp4`，確認 stroke start / end / net zone 是否合理。
4. 檢查 `*_net_zone_speed_detail.csv`，確認最大速度段是否出現在合理 frame。
5. 檢查 `landing_heatmap.png` 與 `landing_zones.png`，確認落點是否符合影片內容。

---

## 12. 詳細文件

更多系統設計、輸入規範、輸出欄位與參數說明，請參考：

```text
docs/system_design.md
docs/input_requirements.md
docs/output_description.md
docs/parameter_notes.md
speed_analysis/README.md
```
