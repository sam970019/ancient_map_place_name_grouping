# ==================== 實驗設定 ====================
# 修改這裡的參數即可切換不同實驗配置，不需要去動其他檔案

import os
from data import DATA_DIR

# --- 實驗控制 ---
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "final")  # 訓練：輸出的模型名稱；評估：要載入的模型名稱
BIDIRECTIONAL_GNN  = os.environ.get("BIDIRECTIONAL_GNN", "True") == "True" # True: 雙向傳遞 (無向邊)，False: 單向傳遞 (有向邊)
TRAIN_SAMPLE_MODE = os.environ.get("TRAIN_SAMPLE_MODE", "NxN")  # "NxN": 全部有序字對為候選（本研究）；"1:1": 早期的對稱取樣，僅供還原舊 checkpoint
TEST_SAMPLE_MODE  = os.environ.get("TEST_SAMPLE_MODE", "NxN")   # "NxN": 模擬真實推論全圖，"1:1": 測試集也僅用對稱取樣 (Evaluate & Training will both use this)
NEG_EDGE_DISTANCE_LIMIT = 0 # 1:1 模式下採樣負邊的最大距離限制（像素），僅供還原舊 checkpoint
# NxN 模式下候選負邊的距離上限 (0:無限制 = 純幾何全 N×N，>0:過濾掉超過此距離的負邊)。
# 注意：正邊不受此限制，>0 時「長邊存在」本身就洩漏標籤（訓練時 >1100px 的候選邊 100% 為正）。
# 定案為 0（最終模型即用 0）；1100 只供重現舊實驗。評估端會從 checkpoint 還原訓練時的值。
NXN_EDGE_DISTANCE_LIMIT = int(os.environ.get("NXN_EDGE_DISTANCE_LIMIT", "0"))

# --- 訊息傳遞邊 (MP 邊) 建構策略 ---
# 原實作以 XML 的 id 屬性（地名分組）串出字鏈當作 GNN 的訊息傳遞邊，
# 但該 id 在部署時來自上游分群，等於「用標籤資訊建圖」，會把答案
# 透過圖結構洩漏給模型。以下開關可改用純幾何規則建構 MP 邊。
#
# 注意：連線預測的正樣本標籤 (place_name_edge) 永遠來自 id 鏈，
#       不受本設定影響 —— 只有訊息傳遞的拓樸會變。
#
#   id       : 原樣的 id 鏈（對照組，可位元級重現改動前的行為）
#   none     : 每節點一條自環邊、原始特徵全零（可執行的「無訊息傳遞」消融）
#   empty    : 真正的空邊集（GATv2 的 edge_attr mean fill 可能產生 NaN，僅供診斷）
#   delaunay : Delaunay 三角化
#   gabriel  : Gabriel graph（Delaunay 子圖，等價 β=1 的 β-skeleton）
#   knn      : 每節點最近 MP_KNN_K 個鄰居（對稱化）
#   dirnn    : 上/下/左/右四個方向各取最近鄰（GLAM 式，具異向性）
#   union:a+b: 取多個模式的聯集，例如 union:delaunay+dirnn
# 預設為 dirnn（最終模型）；delaunay、gabriel、knn、none 為消融比較用。
MP_EDGE_MODE = os.environ.get("MP_EDGE_MODE", "dirnn")
MP_KNN_K     = int(os.environ.get("MP_KNN_K", "8"))   # knn 模式的鄰居數
# 拓樸判定用的座標基準："tl" 左上角(維持原行為，預設) / "center" 框中心點。
# 邊特徵的數值一律以左上角計算，不受此設定影響，
# 使「拓樸」成為唯一變動的變數。
MP_ANCHOR    = os.environ.get("MP_ANCHOR", "tl")

# --- 訓練時的負邊抽樣 ---
# 候選邊集合維持全 N×N（NXN_EDGE_DISTANCE_LIMIT=0 時），推論與評估一律掃全部；
# 只有「訓練時進入 loss 的負邊」按此比例每個 epoch 均勻重抽，正邊全部保留。
#   1/6 : 最終模型（預設）。負邊約為正邊的 900 倍，抽 1/6 後約 1:150，訓練時間亦約降為 1/6
#   1.0 : 不抽樣
# 在期望上等價於把 focal loss 的負邊權重 (1-α) 乘上此比例，
# 是全 N×N loss 的無偏隨機估計（PyG negative_sampling 的同類做法），
# 不帶任何幾何條件。每個 epoch 重抽，120 epoch 下每條負邊平均被看到 20 次。
# 可寫小數或分數，例如 NEG_SAMPLE_RATIO=1/6 或 0.1667
from fractions import Fraction as _Fraction
NEG_SAMPLE_RATIO = float(_Fraction(os.environ.get("NEG_SAMPLE_RATIO", "1/6").strip()))
# 針對性負邊保留：抽樣時「難負邊」全留不抽，其餘照 NEG_SAMPLE_RATIO。
# 難負邊 = 新特徵允許放寬距離的族群：兩端平均字級 > HARD_NEG_SIZE_REL × 該圖中位（大字，約佔候選對 1.5%）。
# 動機：14f 的分析顯示「大字 + 遠 + 不共線」的負邊在均勻 1/6 抽樣下太稀，
# 模型學到「大字可以連很遠」卻沒學到它的邊界（縣名互連）。
# 線性通名（河/路/溝…）的對預設不算難負邊：溝、河太常見，會佔到候選對的 9%，
# 而 lin_any 的影響本來就小；要納入時設 HARD_NEG_INCLUDE_LIN=1。
NEG_KEEP_HARD = os.environ.get("NEG_KEEP_HARD", "1") == "1"   # 最終模型：1
HARD_NEG_SIZE_REL = 1.5
# 難負邊的保留比例：1 = 全留（最終模型），1/2 = 留一半。可寫分數。只在 NEG_KEEP_HARD=1 時有意義。
HARD_NEG_RATIO = float(_Fraction(os.environ.get("HARD_NEG_RATIO", "1").strip()))
HARD_NEG_INCLUDE_LIN = os.environ.get("HARD_NEG_INCLUDE_LIN", "0") == "1"
# 直橫書翻轉增強：訓練圖另建一份沿反對角線翻轉 (x,y)→(−y,−x)、w↔h 的副本，
# 奇數 epoch 用原圖、偶數 epoch 用翻轉圖。右→左橫書 ↔ 上→下直書 互換、閱讀順序不變，
# 直書樣本從 13.7% 變成約 50%。每個 epoch 的邊數不變，訓練時間不變；記憶體約 ×2。
ORIENT_AUG = os.environ.get("ORIENT_AUG", "1") == "1"   # 最終模型：1
# 翻轉週期：每 ORIENT_AUG_PERIOD 個 epoch 用一次翻轉圖。2 = 一半 epoch 翻（最終模型），
# 3 = 三分之一翻（橫書 2/3、直書 1/3）。
ORIENT_AUG_PERIOD = int(os.environ.get("ORIENT_AUG_PERIOD", "2"))

# --- Focal Loss 訓練 ---
USE_FOCAL_LOSS = os.environ.get("USE_FOCAL_LOSS", "True") == "True"  # 適合配合 NxN 使用 #True: 使用 Focal Loss，False: 使用 Cross Entropy
FOCAL_ALPHA = 0.8   # 對稀有類別（正樣本）的權重，數字越大越鼓勵模型預測正邊 (>0.5 保護正樣本，<0.5 保護負樣本) (可針對較嚴峻的資料不平衡狀況 ex1:1000)
FOCAL_GAMMA = 2.0    # 壓制 Easy Negatives 的強度，數字越大越專注於 Hard Negatives (對於1:5 1:10的資料不平衡 用此參數調整即可)

# --- 訓練超參數 ---
EPOCHS = int(os.environ.get("EPOCHS", "120"))
LEARNING_RATE = 0.002
BATCH_SIZE = 4096
RANDOM_SEED = 28          # 只決定資料切分（訓練 / 測試），改了測試集就變了，不要動
# 訓練隨機性（模型初始權重、每 epoch 的負邊抽樣、loader 打亂順序）。多 seed 實驗改這個，
# 資料切分不受影響。GPU 上的 scatter 等運算本身不確定，同 seed 也不保證位元級相同。
TRAIN_SEED = int(os.environ.get("TRAIN_SEED", "28"))

# --- 訓練計時 ---
# 1：每個 epoch 印出時間分配（重建 loader / 等資料 / GPU 運算 / 評估…）並寫入
#     OUTPUT_DIR/profile_<EXPERIMENT_NAME>.csv。GPU 計時需要 cuda.synchronize，
#     會讓訓練稍慢，只在找瓶頸時開；不影響訓練結果。
TRAIN_PROFILE = os.environ.get("TRAIN_PROFILE", "0") == "1"

# --- 評估超參數 ---
PRED_THRESHOLD = float(os.environ.get("PRED_THRESHOLD", "0.65"))

# --- 資料集設定 ---
MAP_NUM = int(os.environ.get("MAP_NUM", "192"))  # 使用的地圖總數（環境變數只供本機小規模測試，預設 192）
TRAIN_RATIO = 0.8          # 訓練集比例，其餘為測試集

# --- 驗證集 ---
# 從訓練切分中抽 VAL_NUM 張（只抽有原始 OCR 的圖，驗證集才能做 E2E），測試集維持原本 39 張不變。
# 訓練中每個 epoch 只看驗證集：選最佳 checkpoint、驅動學習率排程；訓練後在驗證集上選 GT 框閾值，
# E2E 閾值另以 e2e_evaluate.py --split val --scan-threshold --save-threshold 選定。測試集只在最後評估一次。
# VAL_NUM=0 退回舊協定（以測試集選模型），僅供重現舊結果。
VAL_NUM = int(os.environ.get("VAL_NUM", "15"))
VAL_SEED = int(os.environ.get("VAL_SEED", "28"))
VAL_FILES = ()             # 訓練時由 split_maps 填入並隨 checkpoint 封存，評估端據此還原同一批驗證圖
VAL_OCR_DIR = os.path.join(DATA_DIR, "xml_ocr_output_train", "ocr")  # 訓練切分的原始 OCR 輸出

# --- DataLoader worker 數 ---
# -1：自動（Windows 0；其他系統 min(CPU 核心數/2, 8)）。同一台機器同時跑多個實驗時調低可省記憶體，
# 計時結果顯示等資料只佔每個 epoch 約 4%，worker 減半影響很小。
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "-1"))

# --- 結尾視覺化 ---
# 1：訓練結束後把每張測試圖的預測畫成圖（OUTPUT_DIR/test_pred_vis，各實驗共用、會互相覆蓋）；0：略過（預設）
VISUALIZE_TEST = os.environ.get("VISUALIZE_TEST", "0") == "1"

# --- 訓練集評估（只用來畫學習曲線，計算量大）---
# 0：不評估（預設）；N：每 N 個 epoch 評估一次訓練集
TRAIN_EVAL_EVERY = int(os.environ.get("TRAIN_EVAL_EVERY", "0"))

# --- 模型架構設定 ---
IN_CHANNELS = 4            # 空間特徵維度 (x, y, w, h)
# 1：節點文字特徵使用 FastText 嵌入（預設）；0：文字輸入全換成零向量、架構不變（文字嵌入消融）。
# 要同時拿掉邊特徵中的文字資訊，另外從 EXTRA_EDGE_FEATURES 移除 bigram_lp（字元接續機率）
# 與 lin_any（依字元判斷是否為河、路等線狀地物）。
USE_TEXT_EMBED = os.environ.get("USE_TEXT_EMBED", "1") == "1"

# --- 邊特徵 ---
# 基礎維（一律以左上角計算）：距離, Δx, Δy, 方向
#   EDGE_ANGLE_MODE="cossin"（預設）: 方向用 (cos θ, sin θ) 兩維 → 基礎 5 維
#   EDGE_ANGLE_MODE="atan2"        : 方向用 arctan2 一維         → 基礎 4 維（舊 checkpoint 自動還原成這個）
#   換掉 atan2 的理由：值域 (-π, π] 的不連續點剛好落在最常見的方向上
#   （橫書 99.6% 是右→左，dx<0、dy≈0，dy 的符號一變 angle 就在 ±π 之間跳），
#   模型因此學會忽略它（置換重要性 0.05）。cos/sin 連續、有界。
EDGE_ANGLE_MODE = os.environ.get("EDGE_ANGLE_MODE", "cossin").strip()
if EDGE_ANGLE_MODE not in ("cossin", "atan2"):
    raise ValueError(f"EDGE_ANGLE_MODE 只能是 cossin 或 atan2，收到 {EDGE_ANGLE_MODE!r}")

# 追加維（皆對 src/dst 交換對稱），逐項開關、以逗號分隔；空字串即退回只有基礎維。
#   ── 第一梯隊（已驗證）──
#   rel_dist   距離 / min(兩端最近鄰距)                → 局部脈絡，壓過度合併
#   size_ratio max(字級)/min(字級)，字級=sqrt(w*h)      → 兩端字級落差（作用族群小，保留供消融）
#   align_ovl  max(x軸投影重疊率, y軸投影重疊率)        → 共線性
#   ── 敢連方向（針對「小而遠」「大而遠」）──
#   lin_any    兩端任一字屬線性地物通名（河/路/鉄…）  → 允許沿線分散的字距
#   size_rel   兩端平均字級 / 該圖字級中位數           → 大標籤本來就排得開
#   dist_norm  距離 / 兩端平均字級                     → 以字寬為單位的距離，讓大字借用小字學到的間距
#   ── 不亂連方向（針對「緊貼的不同地名」）──
#   bigram_lp  max(log P(b|a), log P(a|b))，由訓練切分 GT 地名學得的字元接續先驗
#              （訓練圖以留一圖法扣掉自己的地名；表存進 checkpoint，評估時直接載入）
#   gap_norm   框與框的最短間距 / 平均字級（重疊 = 0）  → 中心距在字級不同時會誤導，框間距不會
#   rank_max   對方在自己距離排序中的名次，取兩方向較大者 → 互為最近鄰是很強的同名線索
# 做特徵消融不需改任何程式碼，例如：
#   EXTRA_EDGE_FEATURES=""                                 只有基礎維
#   EXTRA_EDGE_FEATURES="rel_dist,size_ratio,align_ovl"    第一梯隊（7features / ns6 那幾版）
#   EXTRA_EDGE_FEATURES="rel_dist,size_ratio,align_ovl,lin_any,size_rel,dist_norm,bigram_lp,gap_norm,rank_max"
#                                                          全部 14 維（text_full 那版）
# 預設只用純成對特徵（11 維），不含 rel_dist、size_rel、rank_max 三個鄰域特徵：
# 訊息傳遞 × 鄰域特徵的 2×2 實驗（3 seeds）中，GNN 拿掉這三個特徵後 GT 框 0.9593 → 0.9651、
# E2E Strict 0.7012 → 0.7076（逐圖 p=0.0008）；沒有訊息傳遞的模型拿掉後則掉約 5pp，
# 代表訊息傳遞能自行學到這些鄰域資訊。
# 欄位順序固定（見 data.py 的 _EXTRA_FEATURE_ORDER），與這裡的書寫順序無關。
# 公式、剪裁值與篩選依據見 data.py 的「邊幾何特徵」區塊。
EXTRA_EDGE_FEATURES = tuple(
    f.strip() for f in
    os.environ.get("EXTRA_EDGE_FEATURES",
                   "size_ratio,align_ovl,lin_any,dist_norm,bigram_lp,gap_norm").split(",")
    if f.strip()
)
EDGE_FEATURE_DIM = (5 if EDGE_ANGLE_MODE == "cossin" else 4) + len(EXTRA_EDGE_FEATURES)
# lin_any 的字集：線性地物（河川、道路、鐵路、溝渠）的通名
LINEAR_FEATURE_CHARS = "河江路線鉄鐵溝道渠圳"
# bigram_lp 的 add-k 平滑常數
BIGRAM_SMOOTHING = 0.5
HIDDEN_CHANNELS = 256
OUT_CHANNELS = 32
# 訊息傳遞層的堆疊層數（最終模型 2 層）。
GNN_LAYERS = int(os.environ.get("GNN_LAYERS", "2"))
# 訊息傳遞層的種類（架構對照實驗用；層數、隱藏維度、層間 ELU 都不變，只換卷積層）：
#   gatv2（預設）: GATv2，邊特徵進入注意力計算
#   gcn          : GCN（對稱正規化 + 自環）；卷積層本身不支援邊特徵
#   sage         : GraphSAGE（平均聚合）；卷積層本身不支援邊特徵
# 三者的連結預測器都一樣會用到候選字對的邊特徵。
GNN_TYPE = os.environ.get("GNN_TYPE", "gatv2").strip().lower()
if GNN_TYPE not in ("gatv2", "gcn", "sage"):
    raise ValueError(f"GNN_TYPE 只能是 gatv2、gcn、sage，收到 {GNN_TYPE!r}")
# 1（預設）：GATv2 的注意力計算使用訊息傳遞邊的邊特徵；0：不使用（訊息照常沿邊傳遞，
# 注意力只看兩端節點）。用來區分「注意力機制」與「邊特徵」各自的貢獻；只對 gatv2 有效。
GNN_EDGE_IN_ATTN = os.environ.get("GNN_EDGE_IN_ATTN", "1") == "1"
# 1（預設）：連結預測器輸入 [z_i‖z_j‖e_ij]；0：只輸入 [z_i‖z_j]，邊特徵只能經由訊息傳遞
# 影響判斷。與 GNN_EDGE_IN_ATTN 搭配做邊特徵注入位置的比較（4.4.5）。
GNN_EDGE_IN_PRED = os.environ.get("GNN_EDGE_IN_PRED", "1") == "1"

# 訓練、驗證、測試都讀人工標註（xml_revised），切分由 split_maps 依 RANDOM_SEED 決定
TRAIN_XML_DIR = os.path.join(DATA_DIR, "xml_revised")
TEST_XML_DIR  = os.path.join(DATA_DIR, "xml_revised")

# XML_DIR 保留為向後相容，指向訓練集目錄（fit_global_scaler / build_global_vocab 使用）
XML_DIR = TRAIN_XML_DIR

GT_XML_DIR = os.path.join(DATA_DIR, "xml_gt") # E2E評估的解答
OCR_XML_DIR = os.environ.get("OCR_XML_DIR", os.path.join(DATA_DIR, "xml_ocr_output")) # E2E評估的OCR結果（可用環境變數換成 LLM 偵測版）
MAP_DIR = os.path.join(DATA_DIR, "map")  # 地圖影像（baseline_A 與視覺化用）
OUTPUT_DIR = os.path.join(DATA_DIR, "mp_gg_file")        # 訓練紀錄、評估報告
MODEL_SAVE_DIR = os.path.join(DATA_DIR, "test10_file")  # 模型 checkpoint

# --- 過濾清單（有問題的 XML 檔案）---
SKIP_FILES = ["18242.xml"]
STOP_FILE = "25225.xml"

# --- VLM 偵測（llm_detect.py）---
# API 金鑰一律從環境變數讀取，不寫進程式（避免跟著 git 外流）。
# Windows 上若 app 啟動後才設定使用者環境變數，改從登錄檔讀取；讀不到時為空字串，
# 只有實際呼叫 LLM 的程式會用到，訓練與評估不受影響。
def _secret(name):
    if os.environ.get(name):
        return os.environ[name]
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            return winreg.QueryValueEx(k, name)[0]
    except (ImportError, OSError):
        return ""

# Azure OpenAI 設定
AZURE_API_KEY = _secret("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = _secret("AZURE_OPENAI_ENDPOINT")
# Azure 的部署名稱由使用者自訂；未設定時用本研究使用的 gpt-5.6-sol
DEPLOYMENT_NAME = _secret("AZURE_OPENAI_DEPLOYMENT") or "gpt-5.6-sol"

# Google Gemini 設定（REST API）
GEMINI_API_KEY = _secret("GEMINI_API_KEY")

LLM_MAX_RETRIES = 10                # 當遇到 429 頻率限制時的重試次數
LLM_BACKOFF_FACTOR = 2             # 重試等待時間的底數 (每次等待底數的次方秒)



