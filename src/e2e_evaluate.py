"""
e2e_evaluate.py — 端到端 (End-to-End) 評估腳本
=================================================
使用前人方法上游 OCR 的輸出 XML 作為輸入，
讓 GNN 對 OCR 偵測到的文字框進行地名分群預測，
再與人工標注的 GT XML 做跨來源比對，得出真實的端到端效能數據。

OCR 檔案與 GT 檔案同名。地圖影像的檔名首位數字不同（9xxxx → 1xxxx、8xxxx → 2xxxx，
見 e2e_common.ocr_filename_to_gt），只在視覺化與 --target-map 比對時用到。

彙總規則：每張圖都計入（預測為空時該圖的 GT 地名全數算漏抓），
任何一張圖讀取或建圖失敗都直接報錯，避免測試集悄悄少圖。

執行方式：
  cd src
  python e2e_evaluate.py
  python e2e_evaluate.py --scan-threshold

注意：需先在 config.py 中設定正確的 EXPERIMENT_NAME。
"""

import argparse
import os
import warnings

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from tqdm import tqdm

import config
import data as data_module
from data import (
    read_xml, fit_global_scaler, build_graph, build_data,
    build_global_vocab, load_fasttext_embeddings, DATA_DIR,
    restore_checkpoint_config, split_maps, resolve_threshold, THRESHOLD_GRID,
)
from metrics import check_pred_clusters, calculate_exact_match_metrics
from e2e_common import (
    ocr_filename_to_gt, read_xml_with_encoding_fix,
    parse_gt_clusters_from_xml, evaluate_map_iou,
)

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# ─────────────────────────────────────────────
# 字體設定（複製自 evaluate.py，已進行警告抑制與雙路徑優化）
# ─────────────────────────────────────────────
def setup_matplotlib_fonts():
    import logging
    # 抑制 Matplotlib 找不到字體時的刷螢幕警告
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

    from matplotlib import font_manager
    import platform

    font_files = ['msjh.ttc', 'msjhbd.ttc', 'msjhl.ttc', 'SimHei.ttf']
    font_loaded = False
    for f in font_files:
        paths_to_try = [
            os.path.join(DATA_DIR, 'map', 'font', f),
            os.path.join(DATA_DIR, 'font', f)
        ]
        for font_path in paths_to_try:
            if os.path.exists(font_path):
                try:
                    font_manager.fontManager.addfont(font_path)
                    prop = font_manager.FontProperties(fname=font_path)
                    font_name = prop.get_name()
                    plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
                    font_loaded = True
                    break
                except Exception:
                    continue
        if font_loaded:
            break
            
    if not font_loaded:
        if platform.system() == 'Windows':
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
        else:
            plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False

setup_matplotlib_fonts()


# ─────────────────────────────────────────────
# 0. 解析命令列參數
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="GNN 端到端評估腳本")
parser.add_argument("--scan-threshold", action="store_true",
                    help="掃描 0.1~0.9 所有閾值並找出最佳 E2E F1")
parser.add_argument("--visualize-entity", action="store_true",
                    help="產出實體級別視覺化圖（哪些地名群組正確/錯誤），使用 config.PRED_THRESHOLD")
parser.add_argument("--target-map", type=str, default=None,
                    help="指定要單獨評估的地圖名稱，可省略 .xml，多張地圖用逗號分隔 (例如: 95609 或 95609,85137)")
parser.add_argument("--dump-errors", action="store_true",
                    help="匯出每張地圖的錯誤分類 (gt_status) 與缺字框座標為 JSON，供蝴蝶效應分析使用")
parser.add_argument("--dump-path", type=str, default=None,
                    help="指定錯誤資料 JSON 的輸出路徑 (預設 OUTPUT_DIR/error_dump_<EXPERIMENT>.json)")
parser.add_argument("--split", choices=("test", "val"), default="test",
                    help="test：39 張 E2E 測試圖（預設）；val：訓練時切出的驗證集（讀訓練切分的原始 OCR 與 xml_revised GT）")
parser.add_argument("--save-threshold", action="store_true",
                    help="把驗證集掃描出的最佳 Strict 閾值寫回 checkpoint（只能與 --split val --scan-threshold 併用）")
args = parser.parse_args()
if args.save_threshold and not (args.split == "val" and args.scan_threshold):
    parser.error("--save-threshold 只能與 --split val --scan-threshold 一起使用（閾值不可在測試集上選）")


# ─────────────────────────────────────────────
# 1. 載入模型（與 evaluate.py 相同邏輯）
# ─────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = os.path.join(config.MODEL_SAVE_DIR, f'trained_model_{config.EXPERIMENT_NAME}.pth')
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"找不到模型檔案：{model_path}\n"
        f"請確認 config.py 中的 EXPERIMENT_NAME = '{config.EXPERIMENT_NAME}' 是否正確。"
    )

loaded = torch.load(model_path, map_location=device, weights_only=False)

# 關鍵架構 / 資料切分參數與 bigram 表的還原邏輯集中在 data.restore_checkpoint_config
# （含舊 checkpoint 的向後相容規則），與 evaluate 及各分析腳本共用。
if restore_checkpoint_config(loaded, config) is not None:
    print("🛡️ 已成功啟用防護罩：載入時空膠囊中的關鍵架構參數。")


# ─────────────────────────────────────────────
# 2. 準備全局 Scaler、詞彙表與 FastText
# ─────────────────────────────────────────────
# 詞彙表與 scaler 一律以 xml_revised 擬合（與訓練端相同；scaler 參數若已封存於 checkpoint 則直接沿用）
vocab_xml_dir = os.path.join(DATA_DIR, "xml_revised")

# 備份原 config 中的 XML_DIR 並暫時切換
original_xml_dir = config.XML_DIR
config.XML_DIR = vocab_xml_dir

all_xml_files = sorted([
    f for f in os.listdir(config.XML_DIR)
    if f.endswith('.xml') and not f.startswith('B-')
])

print("計算全局正規化參數...")
fit_global_scaler(all_xml_files)

print("建立詞彙表...")
build_global_vocab(all_xml_files)
data_module.restore_vocab(loaded)  # 新 checkpoint 內存有詞彙表時直接沿用

# 還原為原設定的 XML_DIR
config.XML_DIR = original_xml_dir

print("載入 FastText 嵌入...")
embedding_dim, vocab_size, embedding_matrix = load_fasttext_embeddings()


# ─────────────────────────────────────────────
# 2.5 建構模型
# ─────────────────────────────────────────────
from models import Model
model = Model(
    in_channels=config.IN_CHANNELS,
    hidden_channels=config.HIDDEN_CHANNELS,
    out_channels=config.OUT_CHANNELS,
    edge_feature_dim=config.EDGE_FEATURE_DIM,
    vocab_size=data_module.vocab_count,
    embedding_dim=embedding_dim,
    embedding_matrix=embedding_matrix,
).to(device)

if isinstance(loaded, dict) and 'model_state_dict' in loaded:
    model.load_state_dict(loaded['model_state_dict'])
else:
    model = loaded
model.eval()
print(f"成功載入模型權重：{model_path}")


# ─────────────────────────────────────────────
# 3. 找出 OCR XML 檔案，並對應 GT XML
# ─────────────────────────────────────────────
if args.split == "val":
    # 驗證集地圖取自訓練切分：OCR 讀訓練切分的原始輸出，GT 讀 xml_revised，兩邊檔名都是 gt_id
    _, val_files, _ = split_maps(all_xml_files, config)
    if not val_files:
        raise RuntimeError("這個 checkpoint 沒有驗證集（VAL_NUM=0 或舊協定訓練），無法做 --split val")
    if 'OCR_XML_DIR' in os.environ:
        print(f"⚠️  --split val 會忽略環境變數 OCR_XML_DIR，改讀 {config.VAL_OCR_DIR}")
    config.OCR_XML_DIR = config.VAL_OCR_DIR
    config.GT_XML_DIR = os.path.join(DATA_DIR, "xml_revised")
    print(f"📏 驗證集 E2E：{len(val_files)} 張（{', '.join(os.path.splitext(f)[0] for f in val_files)}）")

OCR_PREFIXES = ('8', '9')  # 8開頭或9開頭的檔案為 OCR 輸出 (若有)
ocr_xml_files = sorted([
    f for f in os.listdir(config.OCR_XML_DIR)
    if f.endswith('.xml')
])
if args.split == "val":
    ocr_xml_files = [f for f in ocr_xml_files if f in set(val_files)]
    if len(ocr_xml_files) != len(val_files):
        raise RuntimeError(f"驗證集有 {len(val_files)} 張，但 {config.OCR_XML_DIR} 只找到 {len(ocr_xml_files)} 張")

if not ocr_xml_files:
    raise RuntimeError(
        f"在 {config.OCR_XML_DIR} 中找不到任何 OCR XML 檔案。"
    )

if args.target_map:
    target_names = [name.strip() for name in args.target_map.split(',')]
    # 清理副檔名
    clean_targets = []
    for name in target_names:
        if name.endswith('.xml'):
            name = name[:-4]
        clean_targets.append(name)
        
    valid_files = []
    invalid_targets = []
    
    for t in clean_targets:
        matched = []
        for f in ocr_xml_files:
            f_base = f[:-4]
            # 1. 完全一致
            if f_base == t:
                matched.append(f)
            # 2. 轉換為 GT 檔名後一致 (例如輸入 15609 匹配 95609.xml)
            elif ocr_filename_to_gt(f)[:-4] == t:
                matched.append(f)
            # 3. 尾碼一致 (例如輸入 5609 匹配 95609.xml)
            elif f_base.endswith(t) and len(t) >= 4:
                matched.append(f)
                
        if matched:
            valid_files.extend(matched)
        else:
            invalid_targets.append(t)
            
    # 去重並保持順序
    seen = set()
    valid_files = [x for x in valid_files if not (x in seen or seen.add(x))]
    
    if valid_files:
        ocr_xml_files = valid_files
        print(f"🎯 已指定單獨評估地圖: {', '.join(valid_files)}")
    if invalid_targets:
        print(f"❌ 找不到與以下輸入匹配的 OCR 地圖: {', '.join(invalid_targets)}")
    if not valid_files:
        print("⚠️ 無法找到任何指定的地圖，將評估所有測試集。")

print(f"\n找到 {len(ocr_xml_files)} 份 OCR XML 檔案：")
for f in ocr_xml_files:
    gt_f = f
    gt_path = os.path.join(config.GT_XML_DIR, gt_f)
    status = "✅" if os.path.exists(gt_path) else "❌ (GT 不存在)"
    print(f"  {f}  →  {gt_f}  {status}")


# ─────────────────────────────────────────────
# 4. 逐一讀取 OCR XML，建構推論用圖，並蒐集 GT 群集
# ─────────────────────────────────────────────
all_data_x, all_edge_index = [], []
all_edge_label_index, all_edge_label = [], []
all_edge_label_attr, all_edge_attr = [], []
all_label_id = []
global_text_idx = {}

# 按地圖記錄對應的 GT 群集（E2E 用）
map_gt_clusters = []   # list of (node_offset, num_nodes, gt_clusters_for_this_map)

node_offset = 0
edge_offset = 0
s_edge_offset = 0

for ocr_file in tqdm(ocr_xml_files, desc="讀取 OCR 地圖"):
    gt_file = ocr_file
    gt_path = os.path.join(config.GT_XML_DIR, gt_file)

    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"{ocr_file}：找不到對應的 GT 檔案 {gt_path}")

    # 讀取 OCR XML（修正 encoding 問題）；解析或建圖失敗直接報錯，不跳過
    ocr_path = os.path.join(config.OCR_XML_DIR, ocr_file)
    try:
        ocr_root = read_xml_with_encoding_fix(ocr_path)
        # 用 OCR 資料建構 PyG 圖（位置特徵、文字特徵來自 OCR 偵測結果）
        data, text_idx, _, place_name_edge, ocr_results = build_graph(ocr_root, model)
    except Exception as e:
        raise RuntimeError(f"{ocr_file}：讀取或建圖失敗") from e

    # 將節點索引加入全局字典
    for idx, info in text_idx.items():
        global_text_idx[idx + node_offset] = info

    # 建構 NxN 候選邊（is_test=True 使用 TEST_SAMPLE_MODE）
    test_data = build_data(data, place_name_edge, ocr_results, is_test=True)

    num_nodes = data.x.size(0)
    num_edges = test_data.edge_label_index.shape[1]
    num_s_edges = test_data.edge_index.shape[1]

    # 從 GT XML 解析真實地名群集（跨來源比對用）
    gt_clusters = parse_gt_clusters_from_xml(gt_path)
    map_gt_clusters.append({
        'ocr_file': ocr_file,
        'gt_file': gt_file,
        'node_offset': node_offset,
        'num_nodes': num_nodes,
        'edge_offset': edge_offset,
        'num_edges': num_edges,
        's_edge_offset': s_edge_offset,
        'num_s_edges': num_s_edges,
        'gt_clusters': gt_clusters,
        'ocr_results': ocr_results,   # 節點座標，供視覺化使用
    })

    all_data_x.append(test_data.x)
    all_edge_index.append(test_data.edge_index + node_offset)
    all_edge_label_index.append(test_data.edge_label_index + node_offset)
    all_edge_label.append(test_data.edge_label)
    all_edge_label_attr.append(test_data.edge_label_attr)
    all_edge_attr.append(test_data.edge_attr)
    all_label_id.append(test_data.label_id)

    node_offset += num_nodes
    edge_offset += num_edges
    s_edge_offset += num_s_edges

if not all_data_x:
    raise RuntimeError("沒有任何 OCR 地圖成功載入，請檢查檔案與路徑設定。")

combined_data = Data(
    x=torch.cat(all_data_x),
    edge_index=torch.cat(all_edge_index, dim=1),
    edge_attr=torch.cat(all_edge_attr, dim=0),
    edge_label_index=torch.cat(all_edge_label_index, dim=1),
    edge_label=torch.cat(all_edge_label),
    edge_label_attr=torch.cat(all_edge_label_attr, dim=0),
    label_id=torch.cat(all_label_id, dim=0),
)

# 彙整所有地圖的 GT 群集（用於整體評估）
all_gt_clusters = [c for m in map_gt_clusters for c in m['gt_clusters']]
print(f"\n共載入 {len(map_gt_clusters)} 張 OCR 地圖，GT 地名群集共 {len(all_gt_clusters)} 個。")

# --- MP 邊建構結果摘要（須與訓練時的 MP_EDGE_MODE 一致，否則會是 train/test mismatch）---
_mp_n = combined_data.x.size(0)
_mp_e = combined_data.edge_index.size(1)
print(f"🔗 MP 邊模式 = {getattr(config, 'MP_EDGE_MODE', 'id')} | "
      f"拓樸座標 = {getattr(config, 'MP_ANCHOR', 'tl')}"
      + (f" | k = {getattr(config, 'MP_KNN_K', 8)}"
         if 'knn' in str(getattr(config, 'MP_EDGE_MODE', '')) else ""))
print(f"   節點 {_mp_n} | MP邊 {_mp_e} (平均度數 {_mp_e / max(_mp_n, 1):.2f}) | "
      f"候選邊 {combined_data.edge_label_index.size(1)} "
      f"(模式 {getattr(config, 'TEST_SAMPLE_MODE', '?')})")


# ─────────────────────────────────────────────
# 5. E2E 實體視覺化函式
# ─────────────────────────────────────────────
def find_missing_gt_boxes(gt_clusters, ocr_boxes, iou_threshold=0.35):
    """
    回傳此地圖中所有「OCR 沒偵測到」的 GT 框中心點座標（純幾何，不需模型）。

    判定邏輯與 evaluate_map_iou 內完全一致：對每個 GT 框找出與所有 OCR 框的
    最佳 IoU，若 < iou_threshold 即視為缺字框（OCR 漏抓）。

    此函式刻意獨立於 evaluate_map_iou，不修改其任何邏輯，確保不影響既有評估。

    Returns:
        list of dict: [{'char': str, 'cx': float, 'cy': float}, ...]
    """
    missing = []
    for gt_c in gt_clusters:
        for gt_box in gt_c['boxes']:
            gl, gt_top, gw, gh = gt_box['left'], gt_box['top'], gt_box['width'], gt_box['height']
            best_iou = 0.0
            for o_box in ocr_boxes:
                ol, ot, ow, oh = o_box['x'], o_box['y'], o_box['w'], o_box['h']
                inter_l = max(gl, ol)
                inter_t = max(gt_top, ot)
                inter_r = min(gl + gw, ol + ow)
                inter_b = min(gt_top + gh, ot + oh)
                inter_area = max(0, inter_r - inter_l) * max(0, inter_b - inter_t)
                if inter_area > 0:
                    iou = inter_area / (gw * gh + ow * oh - inter_area)
                    if iou > best_iou:
                        best_iou = iou
            if best_iou < iou_threshold:
                missing.append({
                    'char': gt_box['char'],
                    'cx': float(gt_box['cx']),
                    'cy': float(gt_box['cy']),
                })
    return missing


def draw_e2e_entity_results(map_info, pred_clusters, pred_labels, pred_status, gt_status, threshold, output_dir=None):
    import networkx as nx
    from matplotlib.patches import Patch, Rectangle
    from PIL import Image

    n_off = map_info['node_offset']
    n_num = map_info['num_nodes']
    e_off = map_info['edge_offset']
    e_num = map_info['num_edges']
    ocr_results = map_info['ocr_results']
    gt_clusters = map_info['gt_clusters']
    ocr_file = map_info['ocr_file']
    gt_file = map_info['gt_file']

    base_name = os.path.splitext(ocr_filename_to_gt(gt_file))[0]
    img_path = os.path.join(config.DATA_DIR, 'map', 'marked', f'{base_name}.png')
    if not os.path.exists(img_path):
        img_path = os.path.join(config.DATA_DIR, 'map', 'marked', f'{base_name}.jpg')

    map_edge_label_index = map_info.get('map_edge_label_index')

    node_status_map = {}
    
    # 預先找出所有屬於 FN_type_a (缺字) 和 FN_type_c (漏連) 的 GT OCR nodes
    for i, gt_c in enumerate(gt_clusters):
        if gt_status[i] == 'FN_type_a':
            # 把它能找到的框標為橘色(缺字地名的殘骸)
            for idx in gt_c['ocr_match_indices']:
                node_status_map[idx] = 'fn_type_a'
        elif gt_status[i] == 'FN_type_c':
            for idx in gt_c['ocr_match_indices']:
                node_status_map[idx] = 'fn_type_c'

    # 預測群組蓋過去
    for j, pred_c in enumerate(pred_clusters):
        st = pred_status[j]
        color_val = 'fp'
        if st == 'TP_strict': color_val = 'tp_strict'
        elif st == 'TP_type_b': color_val = 'tp_type_b'
        
        for idx in pred_c['node_indices']:
            node_status_map[idx] = color_val

    color_map = {
        'tp_strict': '#2ecc71',   # 綠
        'tp_type_b': '#3498db',   # 藍
        'fn_type_a': '#f1c40f',   # 黃 (缺字)
        'fn_type_c': '#9b59b6',   # 紫 (漏連)
        'fp': '#e74c3c'           # 紅
    }

    if os.path.exists(img_path):
        img = Image.open(img_path)
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.imshow(img)
    else:
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_facecolor('white')

    for n, status in node_status_map.items():
        node = ocr_results[n]
        x, y, w, h = node['x'], node['y'], node['w'], node['h']
        color = color_map[status]
        rect = Rectangle((x, y), w, h, linewidth=2.5, edgecolor=color, facecolor=color, alpha=0.25)
        ax.add_patch(rect)

    # 找出 edge_to_group 以便畫線
    node_to_group = {}
    for j, pred_c in enumerate(pred_clusters):
        for idx in pred_c['node_indices']:
            node_to_group[idx] = j

    for i in range(e_num):
        if pred_labels[i] < threshold: continue
        src = int(map_edge_label_index[0, i])
        dst = int(map_edge_label_index[1, i])
        g_idx = node_to_group.get(src)
        g_idx_dst = node_to_group.get(dst)
        
        if g_idx is None or g_idx != g_idx_dst: continue
        st = pred_status[g_idx]
        if st == 'TP_strict': line_color = '#2ecc71'
        elif st == 'TP_type_b': line_color = '#3498db'
        else: line_color = '#e74c3c'
            
        sx = ocr_results[src]['x'] + ocr_results[src]['w'] / 2
        sy = ocr_results[src]['y'] + ocr_results[src]['h'] / 2
        ex = ocr_results[dst]['x'] + ocr_results[dst]['w'] / 2
        ey = ocr_results[dst]['y'] + ocr_results[dst]['h'] / 2
        ax.plot([sx, ex], [sy, ey], color=line_color, linewidth=1.5, alpha=0.6)

    tp_strict_count = pred_status.count('TP_strict')
    tp_type_b_count = pred_status.count('TP_type_b')
    fp_count = pred_status.count('FP')
    fn_a_count = gt_status.count('FN_type_a')
    fn_c_count = gt_status.count('FN_type_c')

    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.6, label=f'完全命中 TP ({tp_strict_count} 個)'),
        Patch(facecolor='#3498db', alpha=0.6, label=f'分群正確但 OCR 錯字 ({tp_type_b_count} 個)'),
        Patch(facecolor='#f1c40f', alpha=0.6, label=f'OCR漏抓/缺字 ({fn_a_count} 個)'),
        Patch(facecolor='#9b59b6', alpha=0.6, label=f'GNN漏連/亂連 ({fn_c_count} 個)'),
        Patch(facecolor='#e74c3c', alpha=0.6, label=f'GNN誤判假群組 FP ({fp_count} 個)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    ax.set_title(f"E2E 實體識別結果 - {os.path.splitext(ocr_file)[0]} → {os.path.splitext(gt_file)[0]} (IoU Match)", fontsize=12)
    ax.axis('off')

    if output_dir is None:
        subfolder = 'e2e_entity_vis_gnn'
        output_dir = os.path.join(config.OUTPUT_DIR, subfolder)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'{os.path.splitext(ocr_file)[0]}_e2e_entity.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out_path


def e2e_evaluate(threshold: float, verbose: bool = True, dump: bool = False) -> dict:
    total_tp_s, total_tp_b, total_fn_a, total_fn_c, total_fp = 0, 0, 0, 0, 0
    total_gt = 0
    total_pred = 0

    if verbose:
        print("\n=== 逐地圖 E2E 結果 ===")

    map_pred_results = []
    dump_records = []  # 蝴蝶效應分析用：僅在 dump=True 時填入
    
    map_bar = tqdm(map_gt_clusters, desc="🔍 [Evaluating]", disable=not verbose)
    for map_info in map_bar:
        ocr_file = map_info['ocr_file']
        map_bar.set_description(f"🔍 [Evaluating] {ocr_file}")
        n_off = map_info['node_offset']
        n_num = map_info['num_nodes']
        e_off = map_info['edge_offset']
        e_num = map_info['num_edges']
        s_e_off = map_info['s_edge_offset']
        s_e_num = map_info['num_s_edges']
        
        map_node_set = set(range(n_off, n_off + n_num))
        map_text_idx = {k - n_off: v for k, v in global_text_idx.items() if k in map_node_set}
        
        mini_data = Data(
            x=combined_data.x[n_off:n_off + n_num],
            edge_index=combined_data.edge_index[:, s_e_off:s_e_off + s_e_num] - n_off,
            edge_attr=combined_data.edge_attr[s_e_off:s_e_off + s_e_num],
            edge_label_index=combined_data.edge_label_index[:, e_off:e_off + e_num] - n_off,
            edge_label=combined_data.edge_label[e_off:e_off + e_num],
            edge_label_attr=combined_data.edge_label_attr[e_off:e_off + e_num],
            label_id=combined_data.label_id[n_off:n_off + n_num],
        )
        
        map_info['map_edge_label_index'] = mini_data.edge_label_index
        
        pred_clusters, preds = check_pred_clusters(
            model, mini_data, device, map_text_idx, threshold=threshold
        )
        gt_clusters = map_info['gt_clusters']
        ocr_boxes = map_info['ocr_results']
        
        # 每張圖都計入：預測為空時 GT 全數算漏抓（Type A / C），GT 為空時預測全數算 FP
        tp_s, tp_b, fn_a, fn_c, fp, p_stat, g_stat = evaluate_map_iou(pred_clusters, gt_clusters, ocr_boxes)

        s_prec = tp_s / len(pred_clusters) if len(pred_clusters) > 0 else 0
        s_rec = tp_s / len(gt_clusters) if len(gt_clusters) > 0 else 0
        s_f1 = 2 * s_prec * s_rec / (s_prec + s_rec) if (s_prec + s_rec) > 0 else 0

        tp_rel = tp_s + tp_b
        r_prec = tp_rel / len(pred_clusters) if len(pred_clusters) > 0 else 0
        r_rec = tp_rel / len(gt_clusters) if len(gt_clusters) > 0 else 0
        r_f1 = 2 * r_prec * r_rec / (r_prec + r_rec) if (r_prec + r_rec) > 0 else 0

        total_tp_s += tp_s
        total_tp_b += tp_b
        total_fn_a += fn_a
        total_fn_c += fn_c
        total_fp += fp
        total_gt += len(gt_clusters)
        total_pred += len(pred_clusters)
            
        map_pred_results.append((pred_clusters, preds, p_stat, g_stat))

        # [DUMP] 蒐集蝴蝶效應分析所需的原始資料（僅在 dump=True 時執行，不影響評估）
        if dump:
            cluster_dumps = [{
                'chars': list(gt_c['chars']),
                'name': ''.join(gt_c['chars']),
                'centroid': [float(gt_c['centroid'][0]), float(gt_c['centroid'][1])],
                'char_xy': [[float(b['cx']), float(b['cy'])] for b in gt_c['boxes']],
                'status': g_stat[i],
            } for i, gt_c in enumerate(gt_clusters)]
            dump_records.append({
                'ocr_file': map_info['ocr_file'],
                'gt_file': map_info['gt_file'],
                'gt_clusters': cluster_dumps,
                'missing_gt_boxes': find_missing_gt_boxes(gt_clusters, ocr_boxes),
            })

        ocr_file = map_info['ocr_file']
        gt_file = map_info['gt_file']
        if verbose:
            tqdm.write(f"  {ocr_file} → {gt_file} | Strict F1={s_f1:.4f} (Relaxed F1={r_f1:.4f})  (pred={len(pred_clusters)}, gt={len(gt_clusters)})")
        
    # 整體指標
    overall_s_prec = total_tp_s / total_pred if total_pred > 0 else 0
    overall_s_rec = total_tp_s / total_gt if total_gt > 0 else 0
    overall_s_f1 = 2 * overall_s_prec * overall_s_rec / (overall_s_prec + overall_s_rec) if (overall_s_prec + overall_s_rec) > 0 else 0
    
    total_tp_rel = total_tp_s + total_tp_b
    overall_r_prec = total_tp_rel / total_pred if total_pred > 0 else 0
    overall_r_rec = total_tp_rel / total_gt if total_gt > 0 else 0
    overall_r_f1 = 2 * overall_r_prec * overall_r_rec / (overall_r_prec + overall_r_rec) if (overall_r_prec + overall_r_rec) > 0 else 0
    
    metrics_res = {
        'Strict': (overall_s_prec, overall_s_rec, overall_s_f1),
        'Relaxed': (overall_r_prec, overall_r_rec, overall_r_f1),
        'Errors': {
            'TP_strict': total_tp_s,
            'TP_relaxed_only': total_tp_b,
            'FN_type_a_miss': total_fn_a,
            'FN_type_c_gnn': total_fn_c
        },
        'map_pred_results': map_pred_results
    }
    if dump:
        metrics_res['dump_records'] = dump_records
    return metrics_res

def scan_threshold():
    print(f"\n🚀 開始掃描 E2E 閾值 (模型: {config.EXPERIMENT_NAME})...")
    thresholds = THRESHOLD_GRID  # 0.10 to 0.95
    if args.split == "test":
        print("⚠️  這是在測試集上的閾值掃描，只能當敏感度分析；正式閾值請用 --split val --scan-threshold --save-threshold 選定。")
    
    best_strict_f1 = 0
    best_strict_th = 0
    best_relaxed_f1 = 0
    best_relaxed_th = 0
    
    print(f"{'Threshold':<10} | {'Strict F1':<12} | {'Relaxed F1':<12}")
    print("-" * 40)
    
    for th in thresholds:
        m = e2e_evaluate(th, verbose=False)
        s_f1 = m['Strict'][2]
        r_f1 = m['Relaxed'][2]
        
        print(f"{th:<10.2f} | {s_f1:<12.4f} | {r_f1:<12.4f}")
        
        if s_f1 > best_strict_f1:
            best_strict_f1 = s_f1
            best_strict_th = th
            
        if r_f1 > best_relaxed_f1:
            best_relaxed_f1 = r_f1
            best_relaxed_th = th
            
    print("-" * 40)
    print(f"🏆 最佳 Strict 閾值: {best_strict_th:.2f} (F1 = {best_strict_f1:.4f})")
    print(f"🏆 最佳 Relaxed 閾值: {best_relaxed_th:.2f} (F1 = {best_relaxed_f1:.4f})")
    print("（同一個系統只能有一個閾值：以 Strict 選定，Relaxed 套用同一個閾值）")
    return best_strict_th, best_strict_f1


if __name__ == "__main__":
    m = None

    if args.scan_threshold:
        best_th, best_f1 = scan_threshold()
        if args.save_threshold:
            if not isinstance(loaded, dict):
                raise RuntimeError("這個 checkpoint 不是含 config 的字典格式，無法寫入閾值")
            loaded.setdefault('thresholds', {})
            loaded['thresholds'] = dict(loaded['thresholds'] or {}, e2e=best_th)
            torch.save(loaded, model_path)
            print(f"💾 已把驗證集選定的 E2E 閾值 {best_th:.2f}（驗證集 Strict F1 = {best_f1:.4f}）寫回 {model_path}")
        if args.dump_errors:
            print("\n⚠️  --dump-errors 需在固定閾值下執行，已略過（請勿與 --scan-threshold 併用）。")
    else:
        threshold, threshold_src = resolve_threshold(loaded, 'e2e')
        print(f"\n使用閾值: {threshold}（來源：{threshold_src}）")
        m = e2e_evaluate(threshold, dump=args.dump_errors)
        print("\n=============================================")
        print(f"模型：{config.EXPERIMENT_NAME}")
        print(f"閾值：{threshold}")
        print("---------------------------------------------")
        print("【嚴格模式 (Strict E2E) - OCR文字與GNN分群均正確】")
        print(f"  Precision : {m['Strict'][0]:.4f}")
        print(f"  Recall    : {m['Strict'][1]:.4f}")
        print(f"  F1        : {m['Strict'][2]:.4f}")
        print("---------------------------------------------")
        print("【寬鬆模式 (GNN Grouping) - 容忍OCR錯字，僅看群集結構】")
        print(f"  Precision : {m['Relaxed'][0]:.4f}")
        print(f"  Recall    : {m['Relaxed'][1]:.4f}")
        print(f"  F1        : {m['Relaxed'][2]:.4f}")
        print("---------------------------------------------")
        err = m['Errors']
        if err:
            total_gt_err = err['TP_strict'] + err['TP_relaxed_only'] + err['FN_type_a_miss'] + err['FN_type_c_gnn']
            print(f"【IoU 錯誤歸因分析 (以 GT 總數 {total_gt_err} 為基準)】")
            if total_gt_err > 0:
                p_s = err['TP_strict'] / total_gt_err * 100
                p_b = err['TP_relaxed_only'] / total_gt_err * 100
                p_a = err['FN_type_a_miss'] / total_gt_err * 100
                p_c = err['FN_type_c_gnn'] / total_gt_err * 100
            else:
                p_s = p_b = p_a = p_c = 0.0
                
            print(f"  ✅ Perfect Match (OCR框對, 字對, GNN圈對) : {err['TP_strict']:4d}  ({p_s:5.1f}%)")
            print(f"  🟦 Type B (OCR錯字, 但GNN圈對)             : {err['TP_relaxed_only']:4d}  ({p_b:5.1f}%)")
            print(f"  🟨 Type A (OCR漏抓/缺字, GNN無能為力)      : {err['FN_type_a_miss']:4d}  ({p_a:5.1f}%)")
            print(f"  🟪 Type C (OCR框都有, 但GNN漏連/亂連)      : {err['FN_type_c_gnn']:4d}  ({p_c:5.1f}%)")
        print("=============================================")

        # ── 匯出錯誤案例資料（供蝴蝶效應分析）──
        if args.dump_errors and m is not None and 'dump_records' in m:
            import json
            dump_path = args.dump_path or os.path.join(
                config.OUTPUT_DIR, f'error_dump_{config.EXPERIMENT_NAME}.json'
            )
            os.makedirs(os.path.dirname(dump_path), exist_ok=True)
            with open(dump_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'experiment': config.EXPERIMENT_NAME,
                    'threshold': threshold,
                    'iou_threshold': 0.35,
                    'maps': m['dump_records'],
                }, f, ensure_ascii=False, indent=2)
            n_maps = len(m['dump_records'])
            n_typec = sum(
                1 for mp in m['dump_records']
                for c in mp['gt_clusters'] if c['status'] == 'FN_type_c'
            )
            n_missing = sum(len(mp['missing_gt_boxes']) for mp in m['dump_records'])
            print(f"\n📦 錯誤案例資料已匯出至: {dump_path}")
            print(f"   共 {n_maps} 張地圖 | FN_type_c 群集 {n_typec} 個 | 缺字框 {n_missing} 個")


    # ── 視覺化（可與任何模式搭配）──
    if args.visualize_entity:
        vis_threshold, _ = resolve_threshold(loaded, 'e2e')
        print(f"\n開始 E2E 實體級別視覺化（閾值={vis_threshold}）...")

        # 若先前跑了 scan_threshold 導致 m 為 None，則在此處補跑一次評估以獲得實體預測
        if m is None:
            print("正在產生 E2E 實體聚類預測...")
            m = e2e_evaluate(vis_threshold)

        # 一次性取得全部邊分數
        model.eval()
        with torch.no_grad():
            gnn_data = Data(
                x=combined_data.x.to(device),
                edge_index=combined_data.edge_index.to(device),
                edge_attr=combined_data.edge_attr.to(device),
                label_id=combined_data.label_id.to(device)
            )
            node_embeddings = model.get_node_embeddings(gnn_data)
            edge_label_index = combined_data.edge_label_index.to(device)
            edge_label_attr  = combined_data.edge_label_attr.to(device)
            batch_size = 100000
            num_total_edges = edge_label_index.shape[1]
            all_logits = []
            for s in range(0, num_total_edges, batch_size):
                e = min(s + batch_size, num_total_edges)
                logits = model.predict_edges(
                    node_embeddings,
                    edge_label_index[:, s:e],
                    edge_label_attr[s:e]
                )
                all_logits.append(logits.cpu())
            all_edge_scores = torch.sigmoid(torch.cat(all_logits)).numpy()

        vis_bar = tqdm(map_gt_clusters, desc="產生 E2E 實體視覺化")
        for idx, map_info in enumerate(vis_bar):
            # 取回 e2e_evaluate 產生的地圖級別預測狀態
            pred_clusters, pred_labels, p_stat, g_stat = m['map_pred_results'][idx]
            out = draw_e2e_entity_results(
                map_info, pred_clusters, pred_labels, p_stat, g_stat, threshold=vis_threshold
            )
            vis_bar.set_postfix({"已儲存": os.path.basename(out)})
            tqdm.write(f" → {out}")
        print("E2E 實體視覺化已完成！")


