import torch
import numpy as np
import networkx as nx
import config

def check_pred_clusters(model, data, device, global_text_idx, threshold=config.PRED_THRESHOLD):
    """
    根據模型預測結果，用連通分量找出預測的地名聚類。
    
    Args:
        model: 訓練好的模型
        data: PyG Data 物件
        device: torch device
        global_text_idx: 全局字典 {節點索引: (id, 文字, x, y)}
        threshold: 預測閾值
    
    Returns:
        pred_clusters: list of dict，包含 'chars' 和 'centroid'
    """
    model.eval()
    with torch.no_grad():
        from torch_geometric.data import Data
        gnn_data = Data(
            x=data.x.to(device),
            edge_index=data.edge_index.to(device),
            edge_attr=data.edge_attr.to(device),
            label_id=data.label_id.to(device)
        )
        node_embeddings = model.get_node_embeddings(gnn_data)
        
        edge_label_index = data.edge_label_index.to(device)
        edge_label_attr = data.edge_label_attr.to(device)
        
        batch_size = 100000
        num_edges = edge_label_index.shape[1]
        all_logits = []
        
        for start_idx in range(0, num_edges, batch_size):
            end_idx = min(start_idx + batch_size, num_edges)
            batch_edge_idx = edge_label_index[:, start_idx:end_idx]
            batch_edge_attr = edge_label_attr[start_idx:end_idx]
            
            logits = model.predict_edges(node_embeddings, batch_edge_idx, batch_edge_attr)
            all_logits.append(logits.cpu())
            
        all_logits = torch.cat(all_logits)
        preds = torch.sigmoid(all_logits).numpy()
        
        pred_positive = preds > threshold
        
        edge_indices = data.edge_label_index.cpu().numpy()
        positive_edge_index = edge_indices[:, pred_positive]
    
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        
        num_nodes = data.x.size(0)
        row = positive_edge_index[0]
        col = positive_edge_index[1]
        data_vals = np.ones(len(row))
        adj_matrix = coo_matrix((data_vals, (row, col)), shape=(num_nodes, num_nodes))
        
        n_components, labels = connected_components(csgraph=adj_matrix, directed=False, return_labels=True)
        
        # 使用超高速 NumPy 向量化 Argsort 與 Unique 進行分組
        sorted_indices = np.argsort(labels)
        sorted_labels = labels[sorted_indices]
        unique_labels, index_offsets, counts = np.unique(sorted_labels, return_index=True, return_counts=True)
        
        clusters = []
        for offset, count in zip(index_offsets, counts):
            if count > 1:
                clusters.append(set(sorted_indices[offset:offset+count]))
    except ImportError:
        G = nx.Graph()
        edges = positive_edge_index.T
        G.add_edges_from(edges)
        clusters = list(nx.connected_components(G))
    
    pred_clusters = []
    for cluster in clusters:
        chars = []
        xs, ys = [], []
        cluster_nodes = []
        for node_idx in cluster:
            node_idx = int(node_idx)
            cluster_nodes.append(node_idx)
            if node_idx in global_text_idx:
                _, char, x, y = global_text_idx[node_idx]
                chars.append(char)
                xs.append(x)
                ys.append(y)
            else:
                chars.append("?")
        # 連通分量只是字的集合，這裡的字序沒有意義（兩種評估都只比對集合）
        pred_clusters.append({
            'chars': tuple(chars),
            'centroid': (np.mean(xs) if xs else 0, np.mean(ys) if ys else 0),
            'node_indices': cluster_nodes
        })
    
    return pred_clusters, preds

def check_gt_clusters(data, global_text_idx):
    positive_mask = data.edge_label == 1
    positive_edges = data.edge_label_index[:, positive_mask]
    
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        
        positive_edges_np = positive_edges.cpu().numpy()
        num_nodes = data.x.size(0)
        row = positive_edges_np[0]
        col = positive_edges_np[1]
        data_vals = np.ones(len(row))
        adj_matrix = coo_matrix((data_vals, (row, col)), shape=(num_nodes, num_nodes))
        
        n_components, labels = connected_components(csgraph=adj_matrix, directed=False, return_labels=True)
        
        # 使用超高速 NumPy 向量化 Argsort 與 Unique 進行分組
        sorted_indices = np.argsort(labels)
        sorted_labels = labels[sorted_indices]
        unique_labels, index_offsets, counts = np.unique(sorted_labels, return_index=True, return_counts=True)
        
        clusters = []
        for offset, count in zip(index_offsets, counts):
            if count > 1:
                clusters.append(set(sorted_indices[offset:offset+count]))
    except ImportError:
        G = nx.Graph()
        edges = positive_edges.cpu().numpy().T
        G.add_edges_from(edges)
        clusters = list(nx.connected_components(G))
    
    gt_clusters = []
    for cluster in clusters:
        chars = []
        xs, ys = [], []
        for node_idx in cluster:
            node_idx = int(node_idx)
            if node_idx in global_text_idx:
                _, char, x, y = global_text_idx[node_idx]
                chars.append(char)
                xs.append(x)
                ys.append(y)
        # 連通分量只是字的集合，這裡的字序沒有意義（兩種評估都只比對集合）
        gt_clusters.append({
            'chars': tuple(chars),
            'centroid': (np.mean(xs) if xs else 0, np.mean(ys) if ys else 0)
        })
    
    return gt_clusters

def calculate_exact_match_metrics(predicted_clusters, ground_truth_clusters,
                                  dist_threshold=50.0):
    """
    使用貪婪空間匹配計算指標。
    文字必須完全相同，且群集中心點距離在 dist_threshold 像素以內才算匹配。
    這樣可正確區分同一張地圖中位置不同的同名地名。

    依序處理每個預測群集，在「尚未被配對、文字相同」的 GT 中挑中心點最近且距離小於
    門檻者（距離相同時取 GT 索引較小者）。以文字為鍵先分組，只計算同文字候選的距離，
    結果與逐一比對全部 GT 完全相同，但不必建立 P×G 的距離矩陣。

    Args:
        predicted_clusters: list of dict {'chars': tuple, 'centroid': (x, y)}
        ground_truth_clusters: list of dict {'chars': tuple, 'centroid': (x, y)}
        dist_threshold: 最大中心點距離 (像素)，預設 50

    Returns:
        precision, recall, f1, total
    """
    total = len(ground_truth_clusters)
    gt_centroids = np.array([g['centroid'] for g in ground_truth_clusters], dtype=np.float64).reshape(-1, 2)

    gt_by_key = {}
    for j, gt in enumerate(ground_truth_clusters):
        gt_by_key.setdefault(tuple(sorted(gt['chars'])), []).append(j)
    gt_by_key = {k: np.array(v) for k, v in gt_by_key.items()}

    matched_gt = np.zeros(total, dtype=bool)
    correct_matches = 0
    for pred in predicted_clusters:
        cands = gt_by_key.get(tuple(sorted(pred['chars'])))
        if cands is None:
            continue
        cands = cands[~matched_gt[cands]]
        if cands.size == 0:
            continue
        dists = np.linalg.norm(gt_centroids[cands] - np.asarray(pred['centroid'], dtype=np.float64), axis=1)
        ok = dists < dist_threshold
        if not ok.any():
            continue
        best = cands[ok][np.argmin(dists[ok])]
        matched_gt[best] = True
        correct_matches += 1

    precision = correct_matches / len(predicted_clusters) if predicted_clusters else 0.0
    recall    = correct_matches / total if total > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # 注意：此任務下 entity-level 沒有 TN，因此不計算傳統 accuracy。
    return precision, recall, f1, total

def get_place_metrics(model, data, device, global_text_idx, threshold=None, map_infos=None):
    """
    獲取地名識別的性能指標。

    Args:
        model: 訓練好的模型
        data: PyG Data 物件
        device: torch device
        global_text_idx: 全局字典 {節點索引: (id, 文字, x, y)}
        threshold: 預測閾值，預設使用 config.PRED_THRESHOLD
        map_infos: 保留參數以相容既有呼叫，目前未使用

    Returns:
        precision, recall, f1, edge_p, edge_r, edge_f1, total, predicted_clusters
    """
    if threshold is None:
        threshold = config.PRED_THRESHOLD
    predicted_clusters, preds = check_pred_clusters(model, data, device, global_text_idx, threshold=threshold)
    
    ground_truth_clusters = check_gt_clusters(data, global_text_idx)

    # Edge-level 指標
    gt_edges = data.edge_label.cpu().numpy()
    pred_edges_binary = preds > threshold

    edge_tp = np.sum((pred_edges_binary == 1) & (gt_edges == 1))
    edge_fp = np.sum((pred_edges_binary == 1) & (gt_edges == 0))
    edge_fn = np.sum((pred_edges_binary == 0) & (gt_edges == 1))

    edge_p  = float(edge_tp / (edge_tp + edge_fp)) if (edge_tp + edge_fp) > 0 else 0.0
    edge_r  = float(edge_tp / (edge_tp + edge_fn)) if (edge_tp + edge_fn) > 0 else 0.0
    edge_f1 = float(2 * edge_p * edge_r / (edge_p + edge_r)) if (edge_p + edge_r) > 0 else 0.0

    if not predicted_clusters or not ground_truth_clusters:
        return 0.0, 0.0, 0.0, edge_p, edge_r, edge_f1, 0, predicted_clusters

    precision, recall, f1, total = calculate_exact_match_metrics(
        predicted_clusters, ground_truth_clusters
    )
    return precision, recall, f1, edge_p, edge_r, edge_f1, total, predicted_clusters
