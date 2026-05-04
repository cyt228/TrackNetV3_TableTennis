# TrackNetV3 for Table Tennis

本專案是以 TrackNetV3 為基礎，針對桌球影片進行球軌跡追蹤、補點與後續速度 / 落點分析的版本。

TrackNetV3 主要由兩個模型組成：

- **TrackNet**：從影片 frame 預測球的位置 heatmap，產生初步球軌跡。
- **InpaintNet**：針對 TrackNet 漏偵測或短暫不連續的軌跡進行 inpainting 補點。

[模型下載](https://1drv.ms/u/c/ab3b33d5410e04f3/IQCwzwpuGP6pSpgw0VyyRSCzAa4jTyVFYiFWUgSd8gPeCf0?e=hWQh7G)

> 目前 `nonoverlap` branch 的主要目標是加快 inference 速度，並保留原本的後處理、inpaint、stroke / speed analysis 流程。這版預設使用 `--eval_mode nonoverlap`，也加入 DataLoader prefetch、pin memory、AMP、channels_last 等加速參數。

---

## Development Environment

建議使用 GPU server 或 Vast.ai 這類雲端環境執行，也可以使用本地 NVIDIA GPU。

```text
Platform: Vast.ai / GPU server
GPU: NVIDIA GeForce RTX 4090 or above
VRAM: 24 GB or above
CPU: 12–24 vCPU
RAM: 64 GB or above
Disk: 150 GB or above
Runtime: Linux container
Access: Jupyter Terminal / VSCode Remote SSH
```

---

## Installation

### Clone this repository

```bash
git clone https://github.com/cyt228/TrackNetV3_TableTennis.git
cd TrackNetV3_TableTennis
git checkout nonoverlap
```

### Create environment

```bash
conda create -n tracknetV3 python=3.8
conda activate tracknetV3
```

### Install requirements

```bash
pip install -r requirements.txt
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

---

## Auto Run Scripts

以下提供三種自動執行方式，可以依照資料型態選擇使用。路徑、輸出位置、GPU id、參數可以直接在 `.sh` 檔案中修改。

| Script | 用途 |
|---|---|
| `run_one.sh` | 執行單一影片 |
| `run_all.sh` | 執行單一資料夾內的所有影片 |
| `run_by_folder.sh` | 依照子資料夾逐一執行，每個資料夾跑完才會換下一個 |

執行前請先確認目前位於專案根目錄：

```bash
bash run_one.sh
bash run_all.sh
bash run_by_folder.sh
```

---

## 主要內容

本專案從原版 TrackNetV3 修改而來，目前主要修改重點如下：

| 檔案 | 修改重點 |
|---|---|
| `predict.py` | 支援單支影片 / 整個資料夾批次預測、輸出預測影片、大影片模式、`nonoverlap` inference、DataLoader 加速、AMP、channels_last、candidate select、reset、inpaint mask 呼叫 |
| `test.py` | 保留原版 evaluation / testing 流程，並放入 `predict.py` 會引用的後處理 function，例如 `get_ensemble_weight()`、`generate_inpaint_mask()`、`predict_location_candidates()`、`select_best_candidate()`、`should_reset_track()` |
| `dataset.py` | Dataset 與影片 frame 讀取相關設定，支援一般 Dataset 與 `Video_IterableDataset` |
| `utils/general.py` | 通用工具函式，例如模型建立、影片讀取、csv 輸出、影片輸出、inpaint mask 相關處理 |
| `speed_analysis/` | 速度、落點、stroke 切分與可視化分析 |

train 相關流程基本沿用原版 TrackNetV3，若要重新訓練，可以參考 [TrackNetV3 原始專案](https://github.com/qaz812345/TrackNetV3)。

[speed_analysis](./speed_analysis) 的詳細介紹另外放在 `speed_analysis/README.md`。

---

## 檔案說明

| 檔案 / 資料夾 | 用途 |
|---|---|
| `predict.py` | 主要預測程式，用來對影片產生球的位置 CSV，也可以輸出畫上軌跡的影片 |
| `test.py` | 原版測試與 evaluation 流程，包含 heatmap 轉座標、ensemble 權重、評估指標，以及本專案的 select / reset / inpaint mask 後處理 |
| `train.py` | 訓練 TrackNet / InpaintNet，基本沿用原版 |
| `dataset.py` | Dataset 與影片 frame 讀取相關設定 |
| `model.py` | TrackNet 與 InpaintNet model 定義 |
| `utils/general.py` | 通用工具函式，例如模型建立、影片讀取、CSV 輸出、影片輸出 |
| `preprocess.py` | 原版資料前處理 |
| `preprocess_median.py` | median background 相關前處理 |
| `generate_mask_data.py` | 產生 InpaintNet 訓練用的 mask data |
| `correct_label.py` | 修正 label 相關工具 |
| `corrected_test_label/` | 修正後的測試標籤資料 |
| `error_analysis.py` | 原版 error analysis 介面 |
| `requirements.txt` | 環境套件 |
| `speed_analysis/` | 速度、落點、stroke 分析 |
| `images/` | README 或分析文件使用的圖片 |

---

## predict.py 使用方式

### 單支影片預測

```bash
CUDA_VISIBLE_DEVICES=0 python predict.py --video_file 048/C0045.mp4 --tracknet_file exp/TrackNet_best.pt --inpaintnet_file exp/InpaintNet_best.pt --save_dir 048 --eval_mode nonoverlap --output_video --large_video
```

### 整個資料夾預測

```bash
CUDA_VISIBLE_DEVICES=0 python predict.py --video_dir /home/code-server/NO3 --tracknet_file exp/TrackNet_best.pt --inpaintnet_file exp/InpaintNet_best.pt --save_dir /home/code-server/NO3/pred_result --eval_mode nonoverlap --output_video --large_video
```

### predict.py 參數說明

| 參數 | 預設值 | 說明 |
|---|---:|---|
| `--video_file` | `None` | 單支影片路徑 |
| `--video_dir` | `None` | 整個資料夾批次預測，會遞迴尋找 `.mp4` / `.MP4` |
| `--tracknet_file` | - | TrackNet 權重檔 |
| `--inpaintnet_file` | `''` | InpaintNet 權重檔，不填則只跑 TrackNet |
| `--batch_size` | `32` | inference batch size |
| `--num_workers` | `4` | 一般 Dataset 的 DataLoader workers |
| `--video_num_workers` | `0` | `Video_IterableDataset` 的 DataLoader workers，建議維持 0，避免 IterableDataset 重複讀 frame |
| `--prefetch_factor` | `2` | DataLoader `num_workers > 0` 時的 prefetch factor |
| `--no_pin_memory` | `False` | 關閉 DataLoader pin memory |
| `--amp` | `fp16` | mixed precision inference，可選 `none`、`fp16`、`bf16` |
| `--channels_last` | `True` | 使用 channels_last memory format，加速 CUDA convolution inference |
| `--no_channels_last` | `False` | 關閉 channels_last |
| `--eval_mode` | `nonoverlap` | inference 模式，可選 `nonoverlap`、`average`、`weight` |
| `--max_sample_num` | `100` | 大影片產生 median background 時最多取樣幾個 frame |
| `--video_range` | `10,20` | 指定用哪一段影片秒數產生 background，例如 `10,20` |
| `--save_dir` | `pred_result` | 輸出資料夾 |
| `--large_video` | `False` | 大影片模式，使用 `Video_IterableDataset`，避免一次讀完整部影片造成記憶體不足 |
| `--output_video` | `False` | 是否輸出畫上軌跡的影片 |
| `--traj_len` | `8` | 輸出影片中顯示幾個 frame 的歷史軌跡 |

---

## eval_mode 差異

| 模式 | sliding step | 概念 | 優點 | 缺點 |
|---|---:|---|---|---|
| `nonoverlap` | `seq_len` | 每次取不重疊的 frame sequence，例如 `0–7`、`8–15`、`16–23` | 最快、計算量最低 | 少了 temporal ensemble，穩定度可能比 `average` / `weight` 低 |
| `average` | `1` | 重疊滑窗，每個 frame 會被多個 sequence 預測，再取平均 | 比 nonoverlap 穩定 | 比 nonoverlap 慢 |
| `weight` | `1` | 重疊滑窗，每個 frame 會被多個 sequence 預測，再依位置權重加權融合 | 原本較穩定的做法 | 最慢、CPU / RAM 壓力較大 |

目前這個 branch 主要使用 `nonoverlap`，目標是降低重疊滑窗造成的重複推論成本。原本 `weight` / `average` 會用 `sliding_step=1`，同一個 frame 會在多個 sequence 中被重複預測；`nonoverlap` 改成 `sliding_step=seq_len`，同一段 frame 只跑一次，因此速度會明顯提升。

---

## 輸出檔案

每支影片會輸出：

| 檔案 | 說明 |
|---|---|
| `影片名稱_ball.csv` | 每一 frame 的球座標 |
| `影片名稱_predict.mp4` | 如果有加 `--output_video`，會輸出畫上球軌跡的影片 |

CSV 格式：

| 欄位 | 說明 |
|---|---|
| `Frame` | frame 編號 |
| `Visibility` | 是否有偵測到球，1 代表有球，0 代表無球 |
| `X` | 球的 x 座標 |
| `Y` | 球的 y 座標 |
| `Inpaint_Mask` | 是否交給 InpaintNet 補點，1 代表補，0 代表不補 |

---

## 修改 / 新增的核心邏輯

這部分主要說明本專案為了讓 TrackNetV3 更適合桌球影片，額外修改或新增的後處理邏輯。主要包含：

- `nonoverlap inference`：降低重疊滑窗造成的重複推論成本。
- `make_dataloader()`：加入 num_workers、prefetch_factor、pin_memory 等加速設定。
- `amp_context()` / `channels_last`：使用 mixed precision 與 channels_last 加速 GPU inference。
- `generate_inpaint_mask()`：決定哪些缺失軌跡要交給 InpaintNet 補。
- `select_best_candidate()`：當同一 frame 有多個候選球點時，選出最可能是真球的位置。
- `should_reset_track()`：判斷目前是否追錯球，是否需要重新開始追蹤。
- `write_pred_video()`：輸出預測影片，方便檢查軌跡結果。

---

## nonoverlap inference：降低重複推論

原版 temporal ensemble 會使用 overlap sliding window，例如 sequence length = 8 時，會產生：

```text
0–7
1–8
2–9
3–10
...
```

這樣可以讓同一個 frame 被多次預測後再融合，但代價是計算量很大。

`nonoverlap` 模式改成：

```text
0–7
8–15
16–23
24–31
...
```

也就是 `sliding_step = seq_len`，讓每個 frame sequence 不重疊。這樣可以大幅減少 TrackNet 與 InpaintNet 的 forward 次數。

### 在 TrackNet 階段

```python
if args.eval_mode == 'nonoverlap':
    dataset = Video_IterableDataset(..., sliding_step=seq_len, ...)
```

或一般 Dataset：

```python
dataset = Shuttlecock_Trajectory_Dataset(..., sliding_step=seq_len, padding=True)
```

### 在 InpaintNet 階段

```python
dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=seq_len, data_mode='coordinate', pred_dict=tracknet_pred_dict, padding=True)
```

### 使用注意

`nonoverlap` 的優點是速度快，但因為沒有 overlap ensemble，所以如果某個 frame 剛好在 sequence 邊界附近，穩定度可能會比 `weight` 低。這也是為什麼後面仍保留 candidate select、reset 與 inpaint mask，避免因為速度提升而完全失去後處理保護。

---

## DataLoader / GPU 加速設定

這版新增 `make_dataloader()`，集中管理 DataLoader 的加速參數。

### 一般 Dataset

一般 Dataset 可以使用多 worker、prefetch、pin memory：

```bash
python predict.py ... --num_workers 4 --prefetch_factor 2
```

### Video_IterableDataset

大影片模式使用 `Video_IterableDataset`，預設：

```bash
--video_num_workers 0
```

因為 IterableDataset 如果沒有特別做 multi-worker sharding，多 worker 可能會造成 frame 重複讀取，所以這裡建議維持 0。

### AMP

預設使用：

```bash
--amp fp16
```

如果遇到數值或相容性問題，可以改成：

```bash
--amp none
```

### channels_last

預設開啟 channels_last：

```bash
--channels_last
```

如果遇到相容性問題，可以關閉：

```bash
--no_channels_last
```

---

## `generate_inpaint_mask()`：控制哪些缺失片段要補

`generate_inpaint_mask()` 的作用是產生 InpaintNet 要補的 mask。TrackNet 預測後，可能會有一些 frame 沒有偵測到球：

```csv
Visibility = 0
X = 0
Y = 0
```

但不是所有 `Visibility = 0` 都應該補。例如：

- 球真的飛出畫面，不應該補。
- 影片開頭還沒出現球，不應該補。
- 影片結尾球已經離開畫面，不應該補。
- 中間短暫 miss，才適合交給 InpaintNet 補。

所以這邊只補「前後都有球」的短暫缺失片段，也就是：

```text
有球 → 短暫消失 → 有球
```

### Function 參數

```python
def generate_inpaint_mask(pred_dict, frame_w, frame_h, max_gap=8, border_margin_x=150, max_angle_diff=100.0, min_valid_run=1, angle_check_min_gap=4, max_reverse_dx=40.0):
```

### predict.py 實際呼叫值

```python
tracknet_pred_dict['Inpaint_Mask'] = generate_inpaint_mask(
    tracknet_pred_dict,
    frame_w=w,
    frame_h=h,
    max_gap=15,
    border_margin_x=160,
    max_angle_diff=100.0,
    min_valid_run=1,
    angle_check_min_gap=15,
)
```

### 參數說明

| 參數 | function 預設值 | predict.py 目前值 | 說明 |
|---|---:|---:|---|
| `pred_dict` | - | - | TrackNet 預測結果，包含 `Frame`、`X`、`Y`、`Visibility` |
| `frame_w` | - | 原影片寬度 | 用來判斷右邊界 |
| `frame_h` | - | 原影片高度 | 保留參數，這版主要只看右邊界 |
| `max_gap` | `8` | `15` | 最多允許連續幾個 frame 消失還可以補 |
| `border_margin_x` | `150` | `160` | 右邊界保護範圍，gap 前後球點太靠近右邊界時不補 |
| `max_angle_diff` | `100.0` | `100.0` | 較長 gap 前後移動方向允許的最大角度差 |
| `min_valid_run` | `1` | `1` | gap 前後至少需要幾個連續可見點 |
| `angle_check_min_gap` | `4` | `15` | gap 長度達到這個值才做方向檢查 |
| `max_reverse_dx` | `40.0` | `40.0` | 較長 gap 前後 x 方向明顯反轉時的判斷門檻 |

### 補洞流程

```text
讀取 TrackNet 預測結果
↓
找到 Visibility = 0 的連續缺失片段
↓
排除影片開頭或結尾的 gap
↓
確認 gap 長度沒有超過 max_gap
↓
確認 gap 前後都有可見球點
↓
確認 gap 前後球點沒有太靠近右邊界
↓
確認 gap 前後有足夠的連續可見點
↓
短 gap 直接標記為需要補
↓
長 gap 額外檢查方向角度與 x 方向反轉
↓
產生 Inpaint_Mask
```

### 設計重點

目前 `predict.py` 使用 `max_gap=15`、`angle_check_min_gap=15`，代表 15 frame 以內的 gap 才有機會補，而且小於 15 frame 的 gap 會比較直接地被補；達到 15 frame 的 gap 才會額外檢查方向角度與 x 方向反轉。

這樣設計是為了高速桌球影片：如果補洞條件太嚴格，短暫 miss 很容易補不到；但如果完全不限制，InpaintNet 可能會補到已經出畫面的球或不同顆球。

---

## `select_best_candidate()`：從多個候選球點中選出最合理的位置

TrackNet 輸出的 heatmap 可能有多個亮點，例如真正的球、背景白點、殘影、反光或遠處其他球。如果只選面積最大的點，容易在球速快或短暫 miss 後選錯。

所以這版會先取出最多 3 個候選點：

```python
MAX_CANDIDATES = 3
candidates = predict_location_candidates(heatmap, max_candidates=MAX_CANDIDATES)
```

再交給 `select_best_candidate()` 根據歷史軌跡挑選。

### Function 參數

```python
def select_best_candidate(candidates, history, miss_count=0, min_area_no_history=6.0, min_area_with_history=2.0, min_y=350, max_y=900, debug=False):
```

### 參數說明

| 參數 | 目前值 | 說明 |
|---|---:|---|
| `candidates` | - | 當前 frame 從 heatmap 找到的候選球點，最多 3 個 |
| `history` | - | 前面 frame 的球軌跡紀錄，格式為 `(x, y, visibility)` |
| `miss_count` | `0` | 目前已經連續 miss 幾個 frame |
| `min_area_no_history` | `6.0` | 沒有歷史軌跡時，候選點最小面積限制 |
| `min_area_with_history` | `2.0` | 有歷史軌跡時，候選點最小面積限制 |
| `min_y` | `350` | 候選點 y 座標下限 |
| `max_y` | `900` | 候選點 y 座標上限 |
| `debug` | `False` | 是否輸出 debug 訊息 |

另外 `predict.py` 目前設定：

```python
HISTORY_SIZE = 25
```

代表 `history` 最多保留最近 25 筆追蹤狀態。這版比之前保留更長的 history，目的是讓短暫 miss 後仍然可以參考前面的移動趨勢。

### 基本選點流程

```text
讀取當前 frame 的 candidates
↓
如果沒有 candidates，回傳 None
↓
先用 min_y / max_y 過濾候選點
↓
如果過濾後沒有 candidates，回傳 None
↓
從 history 取出 visibility = 1 的有效球點
↓
如果沒有有效 history，用面積選球
↓
如果有有效 history，根據上一個球點與預測位置篩選 candidates
↓
依照 miss_count 放寬 x / y 距離限制
↓
排除方向突然反轉或離預測位置太遠的 candidate
↓
選出最接近預測位置的 candidate
```

### miss_count 距離限制

這版不只放寬 x，也同步依照 miss_count 放寬 y：

| miss_count 狀況 | `max_x_gap` | `max_y_gap` | 說明 |
|---|---:|---:|---|
| `miss_count == 0` | `130.0` | `100.0` | 沒有 miss，球應該離上一點不會太遠 |
| `miss_count <= 2` | `220.0` | `140.0` | 短暫 miss，允許較大位移 |
| `miss_count <= 4` | `300.0` | `180.0` | miss 更久，進一步放寬 |
| `miss_count > 4` | `600.0` | `220.0` | miss 很久，允許重新抓回距離較遠的球 |

### 沒有 history 時

如果目前沒有有效歷史軌跡，就不能根據前一點或方向判斷，因此會改用面積：

```python
valid_candidates = [c for c in candidates if c["area"] >= min_area_no_history]
return max(valid_candidates, key=lambda c: c["area"])
```

### 有 history 時

如果有歷史軌跡，會用最後一個有效點與前一個有效點估計球的移動方向：

```python
hist_dx = last_x - prev_x
hist_dy = last_y - prev_y
pred_x = last_x + hist_dx
pred_y = last_y + hist_dy
```

然後優先選最接近預測位置的 candidate。

### 方向反轉限制

在沒有 miss 且至少有兩個有效 history 時，會避免 x 方向突然反轉：

```python
if hist_dx > 12 and dx < -12:
    continue
if hist_dx < -12 and dx > 12:
    continue
```

也會限制 candidate 不要離預測位置太遠：

```python
if x_to_pred > 120 or y_to_pred > 80:
    continue
```

---

## `should_reset_track()`：判斷是否重新追蹤

`should_reset_track()` 是用來判斷目前的追蹤狀態是否還可信。當程式追到背景球、停滯白點、飛出畫面的球，或垂直方向的錯誤軌跡時，就需要 reset，讓後續可以重新找球。

### Function 參數

```python
def should_reset_track(history, frame_w, frame_h, border_margin=40, stale_frames=6, stale_avg_step_thresh=6.5, stale_y_span_thresh=12.0, stale_x_span_thresh=35.0, debug=False):
```

### predict.py 實際呼叫值

```python
need_reset, reset_reason = should_reset_track(
    track_state["history"],
    frame_w=frame_w,
    frame_h=frame_h,
    border_margin=40,
    stale_frames=5,
    stale_avg_step_thresh=8.0,
    stale_y_span_thresh=12.0,
    stale_x_span_thresh=35.0,
)
```

### 參數說明

| 參數 | function 預設值 | predict.py 目前值 | 說明 |
|---|---:|---:|---|
| `history` | - | - | 前面 frame 的追蹤紀錄，格式為 `(x, y, visibility)` |
| `frame_w` | - | 原影片寬度 | 用來判斷左右邊界 |
| `frame_h` | - | 原影片高度 | 用來判斷上下邊界 |
| `border_margin` | `40` | `40` | 距離畫面邊界多少 pixel 內視為靠近邊界 |
| `stale_frames` | `6` | `5` | 檢查最近幾個有效球點是否幾乎不動 |
| `stale_avg_step_thresh` | `6.5` | `8.0` | 最近幾個有效球點的平均移動距離門檻 |
| `stale_y_span_thresh` | `12.0` | `12.0` | 最近幾個有效球點的 y 方向最大變化門檻 |
| `stale_x_span_thresh` | `35.0` | `35.0` | 最近幾個有效球點的 x 方向最大變化門檻 |
| `debug` | `False` | `False` | 是否輸出 reset 原因 |

### reset reason

| reset reason | 說明 |
|---|---|
| `border_out` | 球靠近邊界，而且移動方向是往畫面外 |
| `stale_ball` | 最近幾個球點幾乎不動，可能追到背景球、球桌上的白點或停住的錯誤點 |
| `vertical_false_track` | x 幾乎不動，但 y 方向大幅變化，可能是垂直方向的錯誤軌跡 |

### ignore stale 設計

如果 reset reason 是 `stale_ball` 或 `vertical_false_track`，`predict.py` 會暫時忽略舊位置附近的 candidate：

```python
track_state["ignore_stale_until"] = int(f_i) + 80
track_state["ignore_stale_pos"] = last_valid
```

忽略範圍目前是：

```python
abs(c["cx"] - sx) <= 300
abs(c["cy"] - sy) <= 180
```

這樣可以避免 reset 後又馬上選回同一個錯誤亮點。

---

## 參數調整建議

| 問題 | 建議調整 |
|---|---|
| inference 太慢 | 使用 `--eval_mode nonoverlap`、確認有開 GPU、維持 `--amp fp16`、開啟 `--channels_last` |
| 一般 Dataset CPU preprocessing 太慢 | 調大 `--num_workers`，例如 4 或 8，但要注意 RAM |
| 大影片模式 frame 重複或順序怪 | `--video_num_workers` 維持 0 |
| GPU 支援 fp16 不穩 | 改成 `--amp none` 或 `--amp bf16` |
| 球速快，短暫 miss 後抓不回來 | 放寬 `select_best_candidate()` 中的 `max_x_gap` / `max_y_gap` |
| 容易選到畫面上方或下方背景點 | 調整 `min_y` / `max_y` |
| 補洞補太少 | 調大 `max_gap` 或 `angle_check_min_gap` |
| 補到已經飛出畫面的球 | 調小 `max_gap` 或調大 `border_margin_x` |
| reset 太頻繁 | 調大 `stale_frames` 或降低 stale 判斷敏感度 |
| reset 後又選回同一個錯誤點 | 調大 ignore stale 範圍或延長 `ignore_stale_until` |

---

## 建議執行流程

一般完整流程可以分成兩段：

```text
predict.py
↓
產生 *_ball.csv 與 *_predict.mp4
↓
speed_analysis/stroke_zone_analysis.py
↓
產生 stroke_zone.csv、zone_detail.csv、speed_compare.csv、landing 圖表與 debug 影片
```

如果只是要先確認 TrackNetV3 的球軌跡，先跑 `predict.py` 即可。如果要分析每一球的速度、落點與 stroke，則再執行 `speed_analysis/stroke_zone_analysis.py`。
