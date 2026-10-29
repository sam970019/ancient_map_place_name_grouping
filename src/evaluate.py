"""
evaluate.py — 獨立評估腳本
============================
載入訓練好的模型，對測試集進行推論與評估。
完全不需要重新訓練，只要改 config.py 就能調整：
  - EXPERIMENT_NAME : 決定要載入哪個模型檔案
  - PRED_THRESHOLD  : 決定判定「連線」的信心門檻
  - MAP_NUM         : 評估用幾張地圖

執行方式：
  cd src
  python evaluate.py

可選：掃描最佳閾值
  python evaluate.py --scan-threshold
"""

import argparse
import os
import random
import warnings

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data

import config
import data as data_module
from data import (
    read_xml, fit_global_scaler, build_graph, build_data,
    build_global_vocab, load_fasttext_embeddings, DATA_DIR,
    restore_checkpoint_config, split_maps, resolve_threshold,
)
from metrics import get_place_metrics
from visualize import visualize_all_maps, visualize_all_maps_entity

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
# 設置中文字體
def setup_matplotlib_fonts():
    from matplotlib import font_manager
    import platform
    from data import DATA_DIR

    # 1. 優先嘗試載入專案內的字體檔
    font_files = ['msjh.ttc', 'msjhbd.ttc', 'msjhl.ttc', 'SimHei.ttf']
    font_loaded = False
    
    for f in font_files:
        font_path = os.path.join(DATA_DIR, 'map', 'font', f)
        if os.path.exists(font_path):
            try:
                font_manager.fontManager.addfont(font_path)
                if 'msjh' in f:
                    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
                elif 'SimHei' in f:
                    plt.rcParams['font.sans-serif'] = ['SimHei']
                font_loaded = True
                break
            except Exception:
                continue

    # 2. 如果專案字體載入失敗，根據 OS 設定預設字體
    if not font_loaded:
        if platform.system() == 'Windows':
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
        else:
            plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'WenQuanYi Micro Hei', 'Droid Sans Fallback']
    
    plt.rcParams['axes.unicode_minus'] = False

setup_matplotlib_fonts()

# ─────────────────────────────────────────────
# 0. 解析命令列參數
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="GNN 評估腳本")
parser.add_argument("--scan-threshold", action="store_true",
                    help="掃描 0.1~0.9 所有閾值並找出最佳 F1")
parser.add_argument("--visualize-edge", action="store_true",
                    help="輸出邊級別視覺化圖（TP/FP/FN 連線，使用 config.PRED_THRESHOLD）")
parser.add_argument("--no-fp", action="store_true",
                    help="搭配 --visualize-edge 使用：隱藏 FP 紅色連線，只顯示 TP/FN")
parser.add_argument("--visualize-entity", action="store_true",
                    help="輸出實體級別視覺化圖（文字框著色，使用 config.PRED_THRESHOLD）")
parser.add_argument("--target-map", type=str, default=None,
                    help="指定要單獨評估的地圖名稱，可省略 .xml，多張地圖用逗號分隔 (例如: 15619 或 15619,15814)")
args = parser.parse_args()

# ─────────────────────────────────────────────
# 1. 載入時空膠囊 (優先覆寫架構)
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
# （含舊 checkpoint 的向後相容規則），與 e2e_evaluate 及各分析腳本共用。
if restore_checkpoint_config(loaded, config) is not None:
    print("🛡️ 已成功啟用防護罩：載入時空膠囊中的關鍵架構參數，防止資料脫鉤。")

# ─────────────────────────────────────────────
# 2. 準備資料（根據模型訓練背景動態決定詞彙與 Scaler 來源）
# ─────────────────────────────────────────────
folder_path = DATA_DIR

# 詞彙表與 scaler 一律以 xml_revised 擬合（與訓練端相同；scaler 參數若已封存於 checkpoint 則直接沿用）
vocab_xml_dir = os.path.join(DATA_DIR, "xml_revised")

# 備份原 config 中的 XML_DIR 並暫時切換
original_xml_dir = config.XML_DIR
config.XML_DIR = vocab_xml_dir

xml_files_for_vocab = [f for f in os.listdir(config.XML_DIR) if f.endswith(".xml") and not f.startswith("B-")]
xml_files_for_vocab = sorted(xml_files_for_vocab)

print("計算全局正規化參數...")
fit_global_scaler(xml_files_for_vocab)

print("建立詞彙表...")
build_global_vocab(xml_files_for_vocab)
data_module.restore_vocab(loaded)  # 新 checkpoint 內存有詞彙表時直接沿用

# 還原為原設定的 XML_DIR
config.XML_DIR = original_xml_dir

print("載入 FastText 嵌入...")
embedding_dim, vocab_size, embedding_matrix = load_fasttext_embeddings()

# 定義當前模式的基礎 xml_files
xml_files = [f for f in os.listdir(config.XML_DIR) if f.endswith(".xml") and not f.startswith("B-")]
xml_files = sorted(xml_files)

# 根據當前評估設定決定測試集地圖列表
# 與 train.py 共用 split_maps，切法由 checkpoint 還原的參數決定
_, _, test_xml_files = split_maps(xml_files, config)
print(f"隨機 {round(config.TRAIN_RATIO*100)}/{round((1-config.TRAIN_RATIO)*100)} 分割")

if args.target_map:
    target_names = [name.strip() for name in args.target_map.split(',')]
    target_files = [name if name.endswith('.xml') else f"{name}.xml" for name in target_names]
    
    eval_dir = config.XML_DIR
    valid_files = [f for f in target_files if os.path.exists(os.path.join(eval_dir, f))]
    invalid_files = [f for f in target_files if not os.path.exists(os.path.join(eval_dir, f))]
    
    if valid_files:
        test_xml_files = valid_files
        print(f"🎯 已指定單獨評估地圖: {', '.join(valid_files)}")
    if invalid_files:
        print(f"❌ 找不到以下指定的地圖: {', '.join(invalid_files)}")
    if not valid_files:
        print("⚠️ 無法找到任何指定的地圖，將評估所有測試集。")

print(f"使用測試集地圖數: {len(test_xml_files)}")

# ─────────────────────────────────────────────
# 2.5 建構與載入模型權重
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
# 3. 建構測試集 combined_test
# ─────────────────────────────────────────────
all_test_data_x, all_test_edge_index = [], []
all_test_edge_label_index, all_test_edge_label = [], []
all_test_edge_label_attr, all_test_edge_attr = [], []
all_test_label_id = []
test_global_text_idx = {}
test_map_infos = []   # 記錄每張地圖的局部偏移資訊，供視覺化使用
node_offset = 0
edge_offset = 0

from tqdm import tqdm

for file in tqdm(test_xml_files, desc="處理測試地圖"):
    if file in config.SKIP_FILES:
        continue
    root = read_xml(file)
    data, text_idx, _, place_name_edge, ocr_results = build_graph(root, model)

    for idx, info in text_idx.items():
        test_global_text_idx[idx + node_offset] = info

    # 這裡的 is_test=True 會讓 build_data 讀取 config.TEST_SAMPLE_MODE
    test_data = build_data(data, place_name_edge, ocr_results, is_test=True)

    img_file = os.path.join(DATA_DIR, "map/marked", f"{os.path.splitext(file)[0]}.png")
    num_edges = test_data.edge_label_index.shape[1]
    num_nodes = data.x.size(0)

    # 記錄每張地圖的偏移資訊，之後視覺化切片要用
    test_map_infos.append({
        'file': file,
        'ocr_results': ocr_results,
        'img_file': img_file,
        'num_edges': num_edges,
        'edge_offset': edge_offset,
        'node_offset': node_offset,
        'dataset_type': 'test',
        'num_nodes': num_nodes,
    })

    all_test_data_x.append(test_data.x)
    all_test_edge_index.append(test_data.edge_index + node_offset)
    all_test_edge_label_index.append(test_data.edge_label_index + node_offset)
    all_test_edge_label.append(test_data.edge_label)
    all_test_edge_label_attr.append(test_data.edge_label_attr)
    all_test_edge_attr.append(test_data.edge_attr)
    all_test_label_id.append(test_data.label_id)
    edge_offset += num_edges
    node_offset += num_nodes

combined_test = Data(
    x=torch.cat(all_test_data_x),
    edge_index=torch.cat(all_test_edge_index, dim=1),
    edge_attr=torch.cat(all_test_edge_attr, dim=0),
    edge_label_index=torch.cat(all_test_edge_label_index, dim=1),
    edge_label=torch.cat(all_test_edge_label),
    edge_label_attr=torch.cat(all_test_edge_label_attr, dim=0),
    label_id=torch.cat(all_test_label_id, dim=0),
)

# ─────────────────────────────────────────────
# 4. 評估函式（支援自訂閾值）
# ─────────────────────────────────────────────
def evaluate(threshold: float):
    p, r, f1, ep, er, ef1, tt, pred_clusters = get_place_metrics(
        model, combined_test, device, test_global_text_idx, threshold=threshold, map_infos=test_map_infos
    )
    return {"Precision": p, "Recall": r, "F1": f1, "Edge_P": ep, "Edge_R": er, "Edge_F1": ef1}, pred_clusters


# ─────────────────────────────────────────────
# 5. 閾值掃描（可選）
# ─────────────────────────────────────────────
def scan_threshold():
    thresholds = [round(t, 2) for t in np.arange(0.1, 0.95, 0.05)]
    results = []
    print("\n⚠️  這是在測試集上的閾值掃描，只能當敏感度分析；正式閾值由訓練時在驗證集上選定。")
    print("\n=== 閾值掃描結果 ===")
    print(f"{'Threshold':>10} | {'Ent_P':>8} {'Ent_R':>8} {'Ent_F1':>8} | {'Edge_P':>8} {'Edge_R':>8} {'Edge_F1':>8}")
    print("-" * 75)
    for t in thresholds:
        m, _ = evaluate(t)
        results.append((t, m["Precision"], m["Recall"], m["F1"], m["Edge_P"], m["Edge_R"], m["Edge_F1"]))
        print(f"{t:>10.2f} | {m['Precision']:>8.4f} {m['Recall']:>8.4f} {m['F1']:>8.4f} | {m['Edge_P']:>8.4f} {m['Edge_R']:>8.4f} {m['Edge_F1']:>8.4f}")

    best = max(results, key=lambda x: x[3])
    print(f"\n最佳閾值: {best[0]:.2f}  ->  Entity F1 = {best[3]:.4f}  (P={best[1]:.4f}, R={best[2]:.4f})")

    # 繪製 Threshold vs F1 / P / R 曲線
    ts, ps, rs, fs, eps, ers, efs = zip(*results)
    plt.figure(figsize=(9, 5))
    plt.plot(ts, fs, '-', label='Entity F1', linewidth=2, color='green')
    plt.plot(ts, ps, '-', label='Entity P', color='blue')
    plt.plot(ts, rs, '-', label='Entity R', color='orange')
    
    plt.plot(ts, efs, '--', label='Edge F1', linewidth=2, color='lightgreen')
    plt.plot(ts, eps, '--', label='Edge P', color='lightblue')
    plt.plot(ts, ers, '--', label='Edge R', color='wheat')
    
    plt.axvline(best[0], color='red', linestyle=':', label=f'Best Ent_F1={best[0]:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(f'Threshold Scan (Entity vs Edge) — {config.EXPERIMENT_NAME}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    out_path = os.path.join(config.OUTPUT_DIR, f'threshold_scan_{config.EXPERIMENT_NAME}.png')
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"閾值掃描曲線圖已儲存至：{out_path}")

    # ==========================================
    # 繪製 PR 曲線圖 (Precision-Recall Trajectory)
    # ==========================================
    plt.figure(figsize=(8, 6))
    
    # Entity PR 軌跡
    plt.plot(rs, ps, linestyle='-', color='gray', alpha=0.4, zorder=1)
    sc = plt.scatter(rs, ps, c=ts, cmap='viridis', s=80, edgecolor='k', zorder=2, label='Entity')
    
    # Edge PR 軌跡
    plt.plot(ers, eps, linestyle='--', color='gray', alpha=0.4, zorder=1)
    plt.scatter(ers, eps, c=ts, cmap='viridis', s=80, edgecolor='white', marker='s', zorder=2, label='Edge')
    
    cbar = plt.colorbar(sc)
    cbar.set_label('Threshold')
    
    # 用醒目的紅星標示最佳 F1 分數點
    best_r = best[2]
    best_p = best[1]
    plt.scatter([best_r], [best_p], marker='*', s=350, facecolors='red', edgecolors='black', zorder=3, label=f'Best Ent_F1:\n(Thr={best[0]:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Trajectory (Entity vs Edge) — {config.EXPERIMENT_NAME}')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 畫 F1 等高線 (背景參考線)
    f1_scores = np.linspace(0.1, 0.9, 9)
    for f1 in f1_scores:
        x = np.linspace(0.01, 1.0, 100)
        y = f1 * x / (2 * x - f1)
        valid = (y >= 0) & (y <= 1)
        plt.plot(x[valid], y[valid], color='lightgray', linestyle=':', alpha=0.5, zorder=0)

    plt.legend(bbox_to_anchor=(1.20, 1), loc='upper left')
    plt.tight_layout()
    
    pr_out_path = os.path.join(config.OUTPUT_DIR, f'pr_curve_{config.EXPERIMENT_NAME}.png')
    plt.savefig(pr_out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"PR 曲線軌跡圖已儲存至：{pr_out_path}")


# ─────────────────────────────────────────────
# 6. 主程式入口
# ─────────────────────────────────────────────
if __name__ == "__main__":

    refined_clusters = None

    if args.scan_threshold:
        scan_threshold()
    else:
        # 預設使用驗證集選定的 GT 框閾值（環境變數 PRED_THRESHOLD 可覆寫）
        threshold, threshold_src = resolve_threshold(loaded, 'gt')
        print(f"\n使用閾值: {threshold}（來源：{threshold_src}）")
        m, refined_clusters = evaluate(threshold)
        print("\n=====================================")
        print(f"模型：{config.EXPERIMENT_NAME}")
        print(f"閾值：{threshold}")
        print("-------------------------------------")
        for k, v in m.items():
            print(f"{k:12}: {v:.4f}")
        print("=====================================")

    # --- 視覺化（各自獨立，可與 any 模式搭配）---
    vis_threshold, _ = resolve_threshold(loaded, 'gt')

    if args.visualize_edge:
        show_fp = not args.no_fp
        fp_label = "（不含 FP）" if not show_fp else "（含 FP）"
        print(f"\n開始邊級別視覺化{fp_label}（閾值={vis_threshold}）...")
        visualize_all_maps(
            model, combined_test, test_map_infos,
            threshold=vis_threshold,
            show_fp=show_fp,
        )
        print("邊級別視覺化已完成！")

    if args.visualize_entity:
        print(f"\n開始實體級別視覺化（閾值={vis_threshold}）...")
        # 若未跑過單次評估 (如 scan_threshold 完畢)，則在此處單次推論取得 LLM 實體以避免二度推論
        if refined_clusters is None:
            print("正在產生實體聚類預測...")
            _, refined_clusters = evaluate(vis_threshold)
            
        visualize_all_maps_entity(
            model, combined_test, test_map_infos,
            threshold=vis_threshold,
            pred_clusters_all=refined_clusters,
        )
        print("實體級別視覺化已完成！")



