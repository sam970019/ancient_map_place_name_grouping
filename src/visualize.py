"""
visualize.py — 地圖預測結果視覺化模組
======================================
提供兩種視覺化：
  1. draw_prediction_results      → 邊級別 (TP/FP/FN 連線)
  2. draw_entity_results          → 實體級別 (文字框著色表示地名群組是否正確)

使用方式：
  from visualize import visualize_all_maps, visualize_all_maps_entity
"""

import os
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from PIL import Image
from torch_geometric.data import Data
from tqdm import tqdm

from data import DATA_DIR
import config

# 設定中文字體
def setup_matplotlib_fonts():
    import logging
    # 抑制 Matplotlib 找不到字體時的刷螢幕警告
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

    from matplotlib import font_manager
    import platform

    # 1. 優先嘗試載入專案內的字體檔 (檢查 map/font 與 font 兩個可能的位置)
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
                    # print(f"Successfully loaded project font: {font_path}")
                    break
                except Exception as e:
                    pass
        if font_loaded:
            break

    # 2. 如果專案字體載入失敗，根據 OS 設定預設字體
    if not font_loaded:
        if platform.system() == 'Windows':
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
        else:
            # Linux 常見中文字體
            plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'WenQuanYi Micro Hei', 'Droid Sans Fallback']
    
    plt.rcParams['axes.unicode_minus'] = False

setup_matplotlib_fonts()


def draw_prediction_results(
    xml_file, ocr_results, data, img_path, edge_scores,
    threshold=0.5, show_fp=True, dataset_type='test', output_dir=None
):
    """
    視覺化單一地圖的模型預測邊結果。
    畫出 TP(綠) / FN(橙)，以及可選的 FP(紅)，TN 不畫。

    Args:
        xml_file    : 地圖 XML 檔名（用於標題與儲存檔名）
        ocr_results : list of dict，每個節點的 {x, y, w, h} 座標
        data        : PyG Data，需含 edge_label_index 和 edge_label
        img_path    : 底圖路徑（不存在時以白底代替）
        edge_scores : numpy array，每條邊的預測機率分數 (0~1)
        threshold   : 判定為正邊的閾值
        show_fp     : 是否畫出 FP 紅色連線（預設 True）
        dataset_type: 用於命名輸出子資料夾（'train' 或 'test'）
        output_dir  : 輸出資料夾，預設用 DATA_DIR/test11_file/{dataset_type}_pred_vis
    Returns:
        output_path : 儲存的圖片路徑
    """
    num_edges = data.edge_label_index.shape[1]
    true_labels = (
        data.edge_label.cpu().numpy()
        if hasattr(data.edge_label, 'cpu')
        else np.array(data.edge_label)
    )
    pred_labels = edge_scores > threshold

    # 統計供圖例顯示
    tp = int(((true_labels == 1) & pred_labels).sum())
    fp = int(((true_labels == 0) & pred_labels).sum())
    fn = int(((true_labels == 1) & ~pred_labels).sum())

    # 載入底圖
    if os.path.exists(img_path):
        img = Image.open(img_path)
        plt.figure(figsize=(15, 15))
        plt.imshow(img)
    else:
        plt.figure(figsize=(15, 10))

    # 繪製邊：TP(綠) / FN(橙)，及可選的 FP(紅)，TN 跳過
    for i in range(num_edges):
        gt   = true_labels[i]
        pred = pred_labels[i]

        if gt == 1:
            color = 'green' if pred else 'orange'   # TP or FN
        else:
            if pred and show_fp:
                color = 'red'                        # FP
            else:
                continue                             # TN 或 FP 被關閉，不畫

        src = int(data.edge_label_index[0, i])
        dst = int(data.edge_label_index[1, i])
        sx = ocr_results[src]['x'] + ocr_results[src]['w'] / 2
        sy = ocr_results[src]['y'] + ocr_results[src]['h'] / 2
        ex = ocr_results[dst]['x'] + ocr_results[dst]['w'] / 2
        ey = ocr_results[dst]['y'] + ocr_results[dst]['h'] / 2
        plt.plot([sx, ex], [sy, ey], color=color, linewidth=2, alpha=0.6)

    # 圖例（右上角）
    legend_elements = [
        Patch(facecolor='green',  alpha=0.7, label=f'TP ({tp})'),
        Patch(facecolor='orange', alpha=0.7, label=f'FN ({fn})'),
    ]
    if show_fp:
        legend_elements.append(Patch(facecolor='red', alpha=0.7, label=f'FP ({fp})'))
    plt.legend(handles=legend_elements, loc='upper right')

    plt.title(
        f"預測結果 - {os.path.splitext(xml_file)[0]} "
        f"({dataset_type}, threshold={threshold:.2f}{'  [FP 隱藏]' if not show_fp else ''})"
    )
    plt.axis('off')

    # 儲存
    if output_dir is None:
        output_dir = os.path.join(config.OUTPUT_DIR, f'{dataset_type}_pred_vis')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f'{os.path.splitext(xml_file)[0]}_pred.png'
    )
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path


def visualize_all_maps(
    model, combined_test, test_map_infos, threshold=0.5,
    show_fp=True, output_dir=None,
):
    """
    對所有測試集地圖進行推論並逐一輸出視覺化圖。

    Args:
        model           : 訓練好的 GNN 模型
        combined_test   : 合併後的 PyG Data（所有測試地圖）
        test_map_infos  : list of dict，每個地圖的邊偏移與節點偏移資訊
        threshold       : 預測閾值
        show_fp         : 是否畫出 FP 紅色連線（預設 True）
        output_dir      : 輸出資料夾（預設由 draw_prediction_results 決定）
    Returns:
        saved_paths : list of str，所有輸出圖片的路徑
    """
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        # 方案 C: 分批處理預測，避免 OOM
        gnn_data = Data(
            x=combined_test.x.to(device),
            edge_index=combined_test.edge_index.to(device),
            edge_attr=combined_test.edge_attr.to(device),
            label_id=combined_test.label_id.to(device)
        )
        node_embeddings = model.get_node_embeddings(gnn_data)
        
        edge_label_index = combined_test.edge_label_index.to(device)
        edge_label_attr = combined_test.edge_label_attr.to(device)
        batch_size = 100000
        num_edges = edge_label_index.shape[1]
        all_logits = []
        for start_idx in range(0, num_edges, batch_size):
            end_idx = min(start_idx + batch_size, num_edges)
            batch_edge_idx = edge_label_index[:, start_idx:end_idx]
            batch_edge_attr = edge_label_attr[start_idx:end_idx]
            logits = model.predict_edges(node_embeddings, batch_edge_idx, batch_edge_attr)
            all_logits.append(logits.cpu())
            
        edge_scores = torch.sigmoid(torch.cat(all_logits)).numpy()

    saved_paths = []
    viz_bar = tqdm(test_map_infos, desc="產生預測視覺化 (連線)")
    for info in viz_bar:
        start       = info['edge_offset']
        end         = start + info['num_edges']
        node_offset = info['node_offset']

        # 切出這張地圖的邊索引（轉回局部節點編號）與分數
        map_edge_index  = combined_test.edge_label_index[:, start:end] - node_offset
        map_edge_scores = edge_scores[start:end]
        map_edge_label  = combined_test.edge_label[start:end]

        map_data = Data(
            edge_label_index=map_edge_index,
            edge_label=map_edge_label.clone()
            if hasattr(map_edge_label, 'clone') else torch.tensor(map_edge_label),
        )

        path = draw_prediction_results(
            info['file'], info['ocr_results'], map_data, info['img_file'],
            map_edge_scores, threshold=threshold,
            show_fp=show_fp,
            dataset_type=info.get('dataset_type', 'test'),
            output_dir=output_dir,
        )
        saved_paths.append(path)
        viz_bar.set_postfix({"視覺化完成": os.path.basename(path)})

    return saved_paths



# ======================================================================
# 實體級別視覺化（Entity-level）
# ======================================================================

def draw_entity_results(
    xml_file, ocr_results, data, img_path, edge_scores,
    threshold=0.5, dataset_type='test', output_dir=None,
    pred_clusters_input=None
):
    """
    實體級別視覺化：以文字框著色呈現哪些地名群組被正確識別。

    顏色定義：
      🟩 綠色框：此字隸屬於「模型完全猜對」的地名群組 (TP entity)
      🟧 橙色框：此字隸屬於 GT 地名，但模型沒有完整預測到 (FN entity)
      🟥 紅色框：此字被模型預測進某個群組，但該群組在 GT 中不存在 (FP entity)
      (無框)  ：孤立字，GT 中不屬於任何地名，模型也未連線

    圖例統計的是「地名數量」，不是邊數量。
    """
    num_edges = data.edge_label_index.shape[1]
    true_labels = (
        data.edge_label.cpu().numpy()
        if hasattr(data.edge_label, 'cpu')
        else np.array(data.edge_label)
    )
    pred_labels = edge_scores > threshold
    N = len(ocr_results)

    # ── 建立 GT 圖與預測圖，找連通分量 ──
    G_gt   = nx.Graph()
    G_pred = nx.Graph()
    G_gt.add_nodes_from(range(N))
    G_pred.add_nodes_from(range(N))

    for i in range(num_edges):
        src = int(data.edge_label_index[0, i])
        dst = int(data.edge_label_index[1, i])
        if true_labels[i] == 1:
            G_gt.add_edge(src, dst)
        if pred_labels[i]:
            G_pred.add_edge(src, dst)

    # 只取「多字地名」（size > 1），單字不算
    gt_clusters   = {frozenset(c) for c in nx.connected_components(G_gt)   if len(c) > 1}
    if pred_clusters_input is not None:
        pred_clusters = {frozenset(c['node_indices']) for c in pred_clusters_input if len(c['node_indices']) > 1}
    else:
        pred_clusters = {frozenset(c) for c in nx.connected_components(G_pred) if len(c) > 1}
    matched = gt_clusters & pred_clusters   # 完全一致才算 TP

    # ── 建立每個節點的狀態 ──
    gt_node_map   = {n: c for c in gt_clusters   for n in c}
    pred_node_map = {n: c for c in pred_clusters for n in c}

    # 狀態優先順序：correct > fn > fp > isolated
    node_status = {}
    for n in range(N):
        gt_c   = gt_node_map.get(n)
        pred_c = pred_node_map.get(n)
        if gt_c is not None:
            node_status[n] = 'correct' if gt_c in matched else 'fn'
        elif pred_c is not None:
            node_status[n] = 'fp'
        # else: isolated → 不畫框

    # ── 統計 ──
    n_correct = len(matched)
    n_fn      = len(gt_clusters)   - len(matched)
    n_fp      = len(pred_clusters) - len(matched)

    color_map = {'correct': '#2ecc71', 'fn': '#e67e22', 'fp': '#e74c3c'}

    # ── 繪圖 ──
    if os.path.exists(img_path):
        img = Image.open(img_path)
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.imshow(img)
    else:
        fig, ax = plt.subplots(figsize=(15, 10))

    # 畫文字框
    for n, status in node_status.items():
        color = color_map[status]
        node  = ocr_results[n]
        x, y, w, h = node['x'], node['y'], node['w'], node['h']
        rect = Rectangle(
            (x, y), w, h,
            linewidth=2.5, edgecolor=color,
            facecolor=color, alpha=0.25
        )
        ax.add_patch(rect)

    # 畫預測群組的連線（讓你看出模型把哪些字連在一起）
    for i in range(num_edges):
        if not pred_labels[i]:
            continue
        src = int(data.edge_label_index[0, i])
        dst = int(data.edge_label_index[1, i])
        pred_c = pred_node_map.get(src)
        pred_c_dst = pred_node_map.get(dst)
        
        # 如果兩端點不是同一個 Refined Cluster (或者是單字)，就不畫連線
        if pred_c is None or pred_c != pred_c_dst:
            continue
            
        line_color = '#2ecc71' if (pred_c in matched) else '#e74c3c'
        sx = ocr_results[src]['x'] + ocr_results[src]['w'] / 2
        sy = ocr_results[src]['y'] + ocr_results[src]['h'] / 2
        ex = ocr_results[dst]['x'] + ocr_results[dst]['w'] / 2
        ey = ocr_results[dst]['y'] + ocr_results[dst]['h'] / 2
        ax.plot([sx, ex], [sy, ey], color=line_color, linewidth=1.5, alpha=0.6)

    # 圖例
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.6, label=f'正確地名 TP ({n_correct} 個)'),
        Patch(facecolor='#e67e22', alpha=0.6, label=f'漏掉的地名 FN ({n_fn} 個)'),
        Patch(facecolor='#e74c3c', alpha=0.6, label=f'誤判的群組 FP ({n_fp} 個)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    ax.set_title(
        f"實體識別結果 - {os.path.splitext(xml_file)[0]} "
        f"({dataset_type}, threshold={threshold:.2f})",
        fontsize=13
    )
    ax.axis('off')

    # 儲存
    if output_dir is None:
        subfolder = f'{dataset_type}_entity_vis_gnn'
        output_dir = os.path.join(config.OUTPUT_DIR, subfolder)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f'{os.path.splitext(xml_file)[0]}_entity.png'
    )
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path


def visualize_all_maps_entity(
    model, combined_test, test_map_infos, threshold=0.5,
    output_dir=None, pred_clusters_all=None,
):
    """
    對所有測試集地圖進行推論並輸出實體級別視覺化圖。

    Args:
        model           : 訓練好的 GNN 模型
        combined_test   : 合併後的 PyG Data
        test_map_infos  : list of dict，每個地圖的邊偏移與節點偏移資訊
        threshold       : 預測閾值
        output_dir      : 輸出資料夾（預設 DATA_DIR/test11_file/{dataset_type}_entity_vis）
    Returns:
        saved_paths : list of str
    """
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        # 方案 C: 分批處理預測，避免 OOM
        gnn_data = Data(
            x=combined_test.x.to(device),
            edge_index=combined_test.edge_index.to(device),
            edge_attr=combined_test.edge_attr.to(device),
            label_id=combined_test.label_id.to(device)
        )
        node_embeddings = model.get_node_embeddings(gnn_data)
        
        edge_label_index = combined_test.edge_label_index.to(device)
        edge_label_attr = combined_test.edge_label_attr.to(device)
        batch_size = 100000
        num_edges = edge_label_index.shape[1]
        all_logits = []
        for start_idx in range(0, num_edges, batch_size):
            end_idx = min(start_idx + batch_size, num_edges)
            batch_edge_idx = edge_label_index[:, start_idx:end_idx]
            batch_edge_attr = edge_label_attr[start_idx:end_idx]
            logits = model.predict_edges(node_embeddings, batch_edge_idx, batch_edge_attr)
            all_logits.append(logits.cpu())
            
        edge_scores = torch.sigmoid(torch.cat(all_logits)).numpy()

    saved_paths = []
    viz_bar = tqdm(test_map_infos, desc="產生實體視覺化 (色塊)")
    for info in viz_bar:
        start       = info['edge_offset']
        end         = start + info['num_edges']
        node_offset = info['node_offset']

        map_edge_index  = combined_test.edge_label_index[:, start:end] - node_offset
        map_edge_scores = edge_scores[start:end]
        map_edge_label  = combined_test.edge_label[start:end]

        map_data = Data(
            edge_label_index=map_edge_index,
            edge_label=map_edge_label.clone()
            if hasattr(map_edge_label, 'clone') else torch.tensor(map_edge_label),
        )

        pred_clusters_input = None

        path = draw_entity_results(
            info['file'], info['ocr_results'], map_data, info['img_file'],
            map_edge_scores, threshold=threshold,
            dataset_type=info.get('dataset_type', 'test'),
            output_dir=output_dir,
            pred_clusters_input=pred_clusters_input
        )
        saved_paths.append(path)
        viz_bar.set_postfix({"實體視覺化完成": os.path.basename(path)})

    return saved_paths
