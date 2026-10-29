# 古地圖地名分群：以圖神經網路進行字元連結預測

本 repo 為古地圖地名分群研究的程式碼。輸入古地圖經 OCR 得到的字框與字，輸出由哪些字組成一個地名。

方法把「把字組成地名」定義為字對之間的**連結預測**：對每一組有序字對，預測「後一字是否為同一地名中、依閱讀順序緊接在前一字之後的字」，再把預測相連的字對取連通分量，得到地名。模型以 GATv2 在幾何建構的訊息傳遞圖上計算每個字的表示，邊特徵（兩字的距離、方向、字級比、軸向重疊等 11 維）同時進入注意力計算與連結預測器。

本研究另探討以視覺語言模型（VLM）改善上游 OCR 的字框偵測，相關程式也在本 repo。

## 主要結果（測試集 39 張，3 個訓練種子平均 ± 標準差）

| 方法 | GT 框 F1 | 端到端 Strict F1 | 端到端 Relaxed F1 |
|---|---|---|---|
| 前人方法（規則式） | — | 0.6459 | 0.7372 |
| Baseline A（距離與角度門檻，規則式） | 0.8863 | 0.6562 | 0.7529 |
| XGBoost | 0.9029 ± 0.88pp | 0.6786 ± 0.14pp | 0.7749 ± 0.14pp |
| **本方法** | **0.9651 ± 0.17pp** | **0.7076 ± 0.14pp** | **0.8071 ± 0.08pp** |

- **GT 框**：以人工標註的字框與字為輸入，評估分群本身。
- **端到端**：以真實 OCR 輸出為輸入；Strict 要求分群與字都正確，Relaxed 只要求分群正確。
- 規則式方法沒有隨機性，只有一次結果。Baseline A 的參數（d_max=60、θ_max=25）以網格搜尋選定：GT 框在訓練切分上選，端到端在驗證集上選。

## 目錄結構

```
src/
  config.py               所有實驗設定（多數可用環境變數覆寫）
  data.py                 讀取 XML、建圖、節點與邊特徵、資料切分、checkpoint 設定還原
  models.py               模型：節點編碼 + GATv2（或 GCN / GraphSAGE）+ 連結預測器
  train.py                訓練；結束時在驗證集選 GT 框閾值並評估測試集
  metrics.py              連通分量分群與地名層級指標
  evaluate.py             GT 框層級評估（獨立執行）
  e2e_evaluate.py         端到端評估（OCR 輸入）；也用於在驗證集選端到端閾值
  e2e_common.py           端到端評估共用函式（IoU 配對、Strict / Relaxed、錯誤類型）
  visualize.py            預測結果視覺化
  baseline_A.py           基準方法 A：距離與角度門檻的規則式分群
  baseline_B.py           基準方法 B：成對分類器（XGBoost、隨機森林）
  eval_upstream_baseline.py  以相同協定評估前人方法（上游系統輸出的分組）
  noise_ablation.py       噪聲消融：逐步以標註資料替換 OCR 的漏字、錯字、雜訊框與座標偏移
  refill_rate_curve.py    補字率曲線（noise_ablation.py 的相依模組）
  missing_rate_curve.py   缺字率曲線（noise_ablation.py 的相依模組）
  llm_detect.py           以 VLM 偵測字框（GPT / Gemini / Claude）
  llm_build_xml.py        把 VLM 偵測結果組成端到端評估用的 XML
```

## 環境安裝

開發環境為 Python 3.10、PyTorch 2.5.1、PyTorch Geometric 2.6.1。

```bash
conda create -n gnn python=3.10 -y
conda activate gnn

# PyTorch：依 CUDA 版本選擇（以下為 CUDA 11.8；只用 CPU 時改用 https://download.pytorch.org/whl/cpu）
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.5.0+cu118.html

pip install -r requirements.txt
```

## 資料準備

資料不包含在本 repo。程式從 `DATA_DIR` 讀取資料；未設定環境變數 `DATA_DIR` 時，預設為本 repo 的**上一層資料夾**（資料與 repo 並排放置）。

```
<DATA_DIR>/
  xml_revised/              人工標註（192 張），訓練、驗證、測試與 GT 框評估都讀這裡
  xml_gt/                   端到端評估的標註答案（測試集）
  xml_ocr_output/           測試集 39 張的上游 OCR 輸出（端到端評估的輸入）
  xml_ocr_output_train/ocr/ 訓練切分的上游 OCR 輸出（驗證集選端到端閾值用）
  cc.zh.300.kv              FastText 中文詞向量（gensim KeyedVectors 格式）
  cc.zh.300.kv.vectors.npy
  map/original/             地圖原圖（VLM 偵測、視覺化用）
  map/font/                 視覺化用的中文字型
  test10_file/              （自動建立）模型 checkpoint
  mp_gg_file/               （自動建立）訓練紀錄與評估報告
```

XML 格式：每個 `<box>` 有 `left`、`top`、`width`、`height` 屬性與 `<label>`（字）；`id` 屬性相同的字框屬於同一地名，地名內的字依閱讀順序排列。

資料切分固定：192 張以 `RANDOM_SEED=28` 切為訓練 153 張、測試 39 張，再從訓練切分中抽 15 張為驗證集（實際訓練 138 張）。**請勿修改 `RANDOM_SEED`、`SKIP_FILES`、`STOP_FILE`**，否則測試集會改變。

## 訓練與評估

所有指令都在 `src/` 底下執行。設定以環境變數傳入，預設值即為最終模型。

### 1. 訓練

```bash
cd src
EXPERIMENT_NAME=final_s28 TRAIN_SEED=28 NUM_WORKERS=4 python train.py
```

- 本研究以 `TRAIN_SEED` = 28、29、30 各訓練一次，報告平均 ± 標準差。
- `NUM_WORKERS` 會影響鄰居取樣的亂數，進而影響結果；本研究的實驗一律為 4。
- 單張 RTX 3090 約 2.8 小時。訓練完成後會在驗證集上選定 GT 框閾值並存入 checkpoint，接著評估測試集，報告寫在 `mp_gg_file/final_metrics_<實驗名稱>.txt`。

### 2. 端到端評估

先在驗證集上選定端到端閾值（寫回 checkpoint），再評估測試集：

```bash
EXPERIMENT_NAME=final_s28 python e2e_evaluate.py --split val --scan-threshold --save-threshold
EXPERIMENT_NAME=final_s28 python e2e_evaluate.py
```

測試集只在最後評估一次，閾值不在測試集上選。加上 `--dump-errors` 可輸出每個地名的判定結果（JSON）。

### 3. GT 框評估（獨立執行）

```bash
EXPERIMENT_NAME=final_s28 python evaluate.py
```

## 邊特徵

每組有序字對（前一字 → 後一字）的邊特徵。最終模型用 11 維（基本 5 維 + `EXTRA_EDGE_FEATURES` 預設的 6 維）；消融實驗另有只用基本 5 維、以及再加 3 個鄰域特徵的 14 維。距離類特徵以字框左上角計算。

| 特徵 | 說明 | 5 維 | 11 維（最終） | 14 維 |
|---|---|:-:|:-:|:-:|
| `distance` | 兩字的距離 | ✓ | ✓ | ✓ |
| `delta_x`、`delta_y` | 水平、垂直位移 | ✓ | ✓ | ✓ |
| `cos`、`sin` | 方向的單位向量 | ✓ | ✓ | ✓ |
| `size_ratio` | 兩字字級的比值（字級 = √(寬 × 高)） | | ✓ | ✓ |
| `align_ovl` | 兩框在水平或垂直方向的投影重疊率（取較大者） | | ✓ | ✓ |
| `lin_any` | 任一字是否為線性地物的通名（河、路等） | | ✓ | ✓ |
| `dist_norm` | 距離 ÷ 兩字平均字級 | | ✓ | ✓ |
| `bigram_lp` | 字元接續機率，只以訓練切分的地名建表（訓練圖以留一圖法扣掉自己） | | ✓ | ✓ |
| `gap_norm` | 兩框最短間距 ÷ 兩字平均字級 | | ✓ | ✓ |
| `rel_dist` | 距離 ÷ 兩字各自最近鄰距離的較小者（鄰域特徵） | | | ✓ |
| `size_rel` | 兩字平均字級 ÷ 全圖字級中位數（鄰域特徵） | | | ✓ |
| `rank_max` | 兩字在彼此距離排序中的名次，取較大者（鄰域特徵） | | | ✓ |

## 重現實驗

以下各組都是在最終模型的設定上只改列出的環境變數，訓練後照上一節做端到端評估。

| 類別 | 實驗 | 設定 |
|---|---|---|
| 對照方法 | 前人方法 | `python eval_upstream_baseline.py` |
| 對照方法 | Baseline A（規則式） | `python baseline_A.py` |
| 對照方法 | XGBoost / 隨機森林 | `python baseline_B.py --model xgb --seed 28`（隨機森林改 `--model rf`） |
| 對照方法、消融：訊息傳遞層 | GCN / GraphSAGE | `GNN_TYPE=gcn` / `GNN_TYPE=sage` |
| 消融：訊息傳遞圖 | 其他建圖方式 | `MP_EDGE_MODE=delaunay` / `gabriel` / `knn` / `none`（無訊息傳遞） |
| 消融：邊特徵 | 加入 3 個鄰域特徵（14 維） | `EXTRA_EDGE_FEATURES=rel_dist,size_ratio,align_ovl,lin_any,size_rel,dist_norm,bigram_lp,gap_norm,rank_max` |
| 消融：邊特徵 | 只用基本 5 維邊特徵 | `EXTRA_EDGE_FEATURES=`（空字串） |
| 消融：文字 | 無節點文字 | `USE_TEXT_EMBED=0` |
| 消融：文字 | 無任何文字 | `USE_TEXT_EMBED=0 EXTRA_EDGE_FEATURES=size_ratio,align_ovl,dist_norm,gap_norm` |
| 消融：邊特徵注入位置 | 注意力不使用邊特徵 | `GNN_EDGE_IN_ATTN=0` |
| 消融：邊特徵注入位置 | 連結預測器不使用邊特徵 | `GNN_EDGE_IN_PRED=0` |
| 消融：訓練策略 | 一般交叉熵 | `USE_FOCAL_LOSS=False` |
| 消融：訓練策略 | 不保留難負樣本 | `NEG_KEEP_HARD=0` |
| 消融：訓練策略 | 不做翻轉增強 | `ORIENT_AUG=0` |
| 誤差分解 | 噪聲消融 | `EXPERIMENT_NAME=final_s28 python noise_ablation.py` |

XGBoost、隨機森林與 GNN 使用相同的切分、候選字對、負樣本抽樣與閾值選法；`baseline_B.py` 執行完會直接輸出 GT 框與端到端結果。

評估端會從 checkpoint 還原訓練時的架構與特徵設定（`data.restore_checkpoint_config`），評估時不需要再設定上表的環境變數。

## 以 VLM 改善字框偵測

需要的 API 金鑰與端點以環境變數提供，變數名稱見 `.env.example`。程式不讀 `.env` 檔，金鑰請勿寫進程式。

```bash
# 1. VLM 偵測：切塊送 VLM，還原到全圖座標並去重（結果快取在 <DATA_DIR>/llm_detect/<設定名稱>/）
#    --deployment 以 gemini 開頭時呼叫 Gemini、以 claude 開頭時呼叫 Claude，其餘為 Azure OpenAI 的部署名稱
python llm_detect.py --maps <地圖編號...> --tile 768 --scale 1 --overlap 192 --workers 8 --deployment <模型或部署名稱>

# 2. 組成各方案的 OCR XML（方案 A：VLM 偵測 + VLM 辨識；方案 C：OCR + VLM 補框）
python llm_build_xml.py --cfg <設定名稱> --out-root <輸出資料夾>

# 3. 端到端評估：以環境變數 OCR_XML_DIR 指定輸入
OCR_XML_DIR=<輸出資料夾>/xml_C_t090 EXPERIMENT_NAME=final_s28 python e2e_evaluate.py
```

方案 B、D（VLM 偵測的字框交給 OCR 辨識）的字元辨識由上游 OCR 系統的辨識模組執行，該模組不在本 repo；辨識完成的 XML 同樣以 `OCR_XML_DIR` 指定後評估。

## 注意事項

- **結果的可重現性**：GPU 上部分運算（如 scatter）本身不具確定性，同一個種子重跑也可能有些微差異；本研究以 3 個種子的平均與標準差報告，端到端評估都在 CPU 上執行（設定 `CUDA_VISIBLE_DEVICES=` 可強制使用 CPU）。
- **舊 checkpoint**：`config.py` 中標註「僅供還原舊 checkpoint」的選項（如 1:1 取樣、atan2 角度）是為了讓研究過程中的舊模型仍能載入評估，最終模型不使用。
