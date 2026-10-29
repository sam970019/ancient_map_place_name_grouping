"""
baseline_A.py — 基準方法 A：距離與角度門檻的規則式分群
=========================================================
演算法：每個字連到距離 d_max 以內最近的 3 個字，在連通分量內搜尋轉折角不超過 theta_max 的
共線路徑，再依「越長越優先、同長度取越直者」貪婪挑選路徑作為地名。沒有任何學習成分。

評估協定與其他方法相同：
  - 資料切分用 data.split_maps（訓練 138 / 驗證 15 / 測試 39）。
  - 兩個參數比照 GNN 的閾值選法，GT 框與端到端各自在非測試資料上以網格搜尋選定：
      GT 框  ：在訓練切分的人工標註輸入上，取地名 F1 最高的 (d_max, theta_max)
      端到端 ：在驗證集的 OCR 輸入上，取 Strict F1 最高的 (d_max, theta_max)
    F1 相同時取 d_max 較小、再取 theta_max 較小者。最佳值落在網格邊界時會提醒擴大範圍。
  - GT 框計分用 metrics.calculate_exact_match_metrics（與 GNN 的 GT 框 F1 相同）；
    端到端計分用 e2e_common.evaluate_map_iou（IoU 0.35、Strict / Relaxed、Type A/B/C）。
  - 沒有隨機性，只跑一次。

輸出（config.OUTPUT_DIR）：
  baseline_A_results.json      選定參數、GT 框與端到端結果、錯誤類型、逐圖 Strict F1
  baseline_A_grid_gt.png       GT 框網格搜尋熱力圖
  baseline_A_grid_e2e.png      端到端網格搜尋熱力圖
  baseline_A_vis/              （--target-map）預測地名畫在地圖上

執行方式（在 src/ 底下）：
  python baseline_A.py
  python baseline_A.py --d-grid 20,40,60,80,100,120,150,200 --theta-grid 5,10,15,20,25,30,35
  python baseline_A.py --target-map 15619                    # 另外畫出指定地圖的預測
  python baseline_A.py --target-map 15619 --d-max 50 --theta-max 15   # 跳過網格搜尋，只畫圖
"""

import os
import json
import argparse
from multiprocessing import Pool, cpu_count

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

import config
# 規則式方法不使用任何特徵；建圖程式仍會計算邊特徵，這裡只保留基礎幾何維，
# 免去需要字元接續表的追加特徵
config.EXTRA_EDGE_FEATURES = ()
config.EDGE_FEATURE_DIM = 5 if config.EDGE_ANGLE_MODE == "cossin" else 4

from data import (
    read_xml, fit_global_scaler, build_graph, build_data, build_global_vocab,
    split_maps, DATA_DIR,
)
from metrics import check_gt_clusters, calculate_exact_match_metrics
from e2e_common import read_xml_with_encoding_fix, parse_gt_clusters_from_xml, evaluate_map_iou

DEFAULT_D_GRID = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
DEFAULT_THETA_GRID = [5, 10, 15, 20, 25, 30, 35]

# ─────────────────────────────────────────────
# 0. 幾何運算輔助函數
# ─────────────────────────────────────────────
def get_angle_between_vectors(v1, v2):
    """計算兩個 2D 向量之間的夾角偏差 (度數，範圍為 0 ~ 180)"""
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    norm1 = np.sqrt(v1[0]**2 + v1[1]**2)
    norm2 = np.sqrt(v2[0]**2 + v2[1]**2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    cos_angle = dot / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

def evaluate_path_straightness(path, coords):
    """計算一條路徑的平均轉折偏差角。若長度 < 3 則轉折角為 0"""
    if len(path) < 3:
        return 0.0
    deviations = []
    for i in range(len(path) - 2):
        p1 = coords[path[i]]
        p2 = coords[path[i+1]]
        p3 = coords[path[i+2]]
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        deviations.append(get_angle_between_vectors(v1, v2))
    return float(np.mean(deviations))

# ─────────────────────────────────────────────
# 1. 路徑搜尋 DFS（每個字只往最近的 3 個鄰居延伸）
# ─────────────────────────────────────────────
def get_all_valid_paths(cc_nodes, coords, d_max, theta_max):
    """
    在一個連通分量內，以限制深度的 DFS 搜尋所有符合角度約束、長度 2 到 7 的簡單路徑。
    每個字只保留 d_max 內距離最近的 3 個鄰居作為延伸方向，避免組合爆炸。
    """
    cc_nodes_list = list(cc_nodes)
    adj = {u: [] for u in cc_nodes_list}
    for u in cc_nodes_list:
        dists = []
        for v in cc_nodes_list:
            if u == v:
                continue
            dist = np.sqrt((coords[u][0] - coords[v][0])**2 + (coords[u][1] - coords[v][1])**2)
            if dist < d_max:
                dists.append((dist, v))
        dists.sort(key=lambda x: x[0])
        adj[u] = [v for _, v in dists[:3]]
                
    valid_paths = []
    
    def dfs(u, path):
        if len(path) >= 2:
            valid_paths.append(list(path))
            
        if len(path) >= 7:  # 地名最大長度限制為 7
            return
            
        for nxt in adj[u]:
            if nxt in path:
                continue
                
            # 角度約束檢查
            if len(path) >= 2:
                # 1. 局部連續轉折角檢查
                p_prev = coords[path[-2]]
                p_curr = coords[path[-1]]
                p_next = coords[nxt]
                
                v_prev = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
                v_next = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])
                
                phi_local = get_angle_between_vectors(v_prev, v_next)
                if phi_local > theta_max:
                    continue
                    
                # 2. 全域主方向偏離檢查
                p_start = coords[path[0]]
                v_global = (p_next[0] - p_start[0], p_next[1] - p_start[1])
                phi_global = get_angle_between_vectors(v_next, v_global)
                if phi_global > theta_max:
                    continue
                    
            path.append(nxt)
            dfs(nxt, path)
            path.pop()
            
    for start in cc_nodes_list:
        dfs(start, [start])
        
    return valid_paths

# ─────────────────────────────────────────────
# 2. 貪婪挑選路徑作為地名
# ─────────────────────────────────────────────
def run_static_greedy_extraction(all_nodes, coords, d_max, theta_max):
    """
    每個字連到 d_max 內最近的 3 個字後取連通分量，在各分量內搜尋候選路徑，
    再依「越長越優先、同長度取越直者」貪婪挑選，已用過的字不再使用。
    距離門檻固定，無法同時適應字距差異很大的地名（例如大字的縣名）。
    """
    G = nx.Graph()
    G.add_nodes_from(all_nodes)
    nodes_list = list(all_nodes)
    
    for u in nodes_list:
        dists = []
        for v in nodes_list:
            if u == v:
                continue
            dist = np.sqrt((coords[u][0] - coords[v][0])**2 + (coords[u][1] - coords[v][1])**2)
            if dist < d_max:
                dists.append((dist, v))
        dists.sort(key=lambda x: x[0])
        for _, v in dists[:3]:
            G.add_edge(u, v)
                
    ccs = list(nx.connected_components(G))
    
    global_candidate_paths = []
    for cc in ccs:
        if len(cc) < 2:
            continue
        cc_paths = get_all_valid_paths(cc, coords, d_max, theta_max)
        
        seen = set()
        for p in cc_paths:
            if p[0] > p[-1]:
                r_path = p[::-1]
            else:
                r_path = p
            t_path = tuple(r_path)
            if t_path not in seen:
                seen.add(t_path)
                global_candidate_paths.append(r_path)
                
    # 排序：優先取最長、其次最直的路徑
    global_candidate_paths.sort(key=lambda p: (-len(p), evaluate_path_straightness(p, coords)))
    
    # 貪婪挑選（每個字只屬於一個地名）
    extracted_places = []
    used_nodes = set()
    
    for path in global_candidate_paths:
        if any(node in used_nodes for node in path):
            continue
        extracted_places.append(path)
        used_nodes.update(path)
        
    return extracted_places

# ─────────────────────────────────────────────
# 3. 讀取地圖
# ─────────────────────────────────────────────
def _coords(text_idx):
    """規則使用的座標：字框左上角（與邊特徵相同）"""
    return {i: (x, y) for i, (_, _, x, y) in text_idx.items()}


def load_gt_map(file):
    """人工標註輸入：GT 地名的定義與 GNN 的 GT 框評估相同（check_gt_clusters）"""
    root = read_xml(file)
    graph, text_idx, _, place_name_edge, ocr_results = build_graph(root, None)
    data = build_data(graph, place_name_edge, ocr_results, is_test=True)
    return {'file': file, 'text_idx': text_idx, 'coords': _coords(text_idx),
            'gt_clusters': check_gt_clusters(data, text_idx)}


def load_e2e_map(file, ocr_dir, gt_dir):
    """OCR 輸入：讀法與 e2e_evaluate.py 相同；讀不到時回傳 None"""
    gt_path = os.path.join(gt_dir, file)
    if not os.path.exists(gt_path):
        print(f"⚠️  跳過 {file}：找不到 GT 檔")
        return None
    try:
        ocr_root = read_xml_with_encoding_fix(os.path.join(ocr_dir, file))
        _, text_idx, _, _, ocr_results = build_graph(ocr_root, None)
    except Exception as e:
        print(f"⚠️  跳過 {file}：讀取或建圖失敗（{e}）")
        return None
    return {'file': file, 'text_idx': text_idx, 'coords': _coords(text_idx),
            'ocr_results': ocr_results, 'gt_clusters': parse_gt_clusters_from_xml(gt_path)}


# ─────────────────────────────────────────────
# 4. 預測與計分
# ─────────────────────────────────────────────
def predict_clusters(m, d_max, theta_max):
    places = run_static_greedy_extraction(list(m['coords']), m['coords'], d_max, theta_max)
    clusters = []
    for path in places:
        info = [m['text_idx'][n] for n in path]
        clusters.append({
            'chars': tuple(c for _, c, _, _ in info),
            'centroid': (float(np.mean([x for _, _, x, _ in info])), float(np.mean([y for _, _, _, y in info]))),
            'node_indices': list(path),
        })
    return clusters


def score_gt(maps, d_max, theta_max):
    """GT 框層級：全部地圖的預測合併後計分（與 GNN 相同）"""
    preds, gts = [], []
    for m in maps:
        preds.extend(predict_clusters(m, d_max, theta_max))
        gts.extend(m['gt_clusters'])
    if not preds or not gts:
        return 0.0, 0.0, 0.0
    p, r, f1, _ = calculate_exact_match_metrics(preds, gts)
    return p, r, f1


def score_e2e(maps, d_max, theta_max):
    """端到端層級：逐圖以 IoU 配對計分後加總（與 e2e_evaluate.py 相同，每張圖都計入）"""
    tot = dict(tp_s=0, tp_b=0, fn_a=0, fn_c=0, fp=0, n_pred=0, n_gt=0)
    per_map = {}
    for m in maps:
        preds = predict_clusters(m, d_max, theta_max)
        gts = [{'chars': c['chars'], 'centroid': c['centroid'], 'boxes': [dict(b) for b in c['boxes']]}
               for c in m['gt_clusters']]
        tp_s, tp_b, fn_a, fn_c, fp, _, _ = evaluate_map_iou(preds, gts, m['ocr_results'])
        per_map[m['file']] = prf(tp_s, len(preds), len(gts))[2]
        for k, v in zip(('tp_s', 'tp_b', 'fn_a', 'fn_c', 'fp'), (tp_s, tp_b, fn_a, fn_c, fp)):
            tot[k] += v
        tot['n_pred'] += len(preds)
        tot['n_gt'] += len(gts)
    strict = prf(tot['tp_s'], tot['n_pred'], tot['n_gt'])
    relaxed = prf(tot['tp_s'] + tot['tp_b'], tot['n_pred'], tot['n_gt'])
    return {
        'strict': dict(zip(('precision', 'recall', 'f1'), strict)),
        'relaxed': dict(zip(('precision', 'recall', 'f1'), relaxed)),
        'counts': {'TP_strict': tot['tp_s'], 'TypeB': tot['tp_b'], 'TypeA': tot['fn_a'],
                   'TypeC': tot['fn_c'], 'FP': tot['fp'], 'gt': tot['n_gt'], 'pred': tot['n_pred']},
        'per_map_strict_f1': per_map,
    }


def prf(tp, n_pred, n_gt):
    p = tp / n_pred if n_pred > 0 else 0.0
    r = tp / n_gt if n_gt > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


# ─────────────────────────────────────────────
# 5. 網格搜尋（多程序平行）
# ─────────────────────────────────────────────
_WORKER_MAPS = None


def _init_worker(maps):
    global _WORKER_MAPS
    _WORKER_MAPS = maps


def _gt_task(params):
    d_max, theta_max = params
    return d_max, theta_max, score_gt(_WORKER_MAPS, d_max, theta_max)[2]


def _e2e_task(params):
    d_max, theta_max = params
    return d_max, theta_max, score_e2e(_WORKER_MAPS, d_max, theta_max)['strict']['f1']


def grid_search(maps, task, d_grid, theta_grid, workers, desc):
    """回傳 (最佳 d_max, 最佳 theta_max, 最佳 F1, F1 矩陣)；F1 相同時取 d_max、theta_max 較小者"""
    params = [(d, t) for d in d_grid for t in theta_grid]
    with Pool(processes=workers, initializer=_init_worker, initargs=(maps,)) as pool:
        results = list(tqdm(pool.imap_unordered(task, params), total=len(params), desc=desc))
    grid = np.zeros((len(d_grid), len(theta_grid)))
    for d, t, f1 in results:
        grid[d_grid.index(d), theta_grid.index(t)] = f1
    best_d, best_t, best_f1 = min(results, key=lambda x: (-x[2], x[0], x[1]))
    if best_d in (d_grid[0], d_grid[-1]) or best_t in (theta_grid[0], theta_grid[-1]):
        print(f"⚠️  {desc}：最佳參數 (d_max={best_d}, theta_max={best_t}) 落在網格邊界，建議擴大搜尋範圍")
    return best_d, best_t, best_f1, grid


def save_heatmap(grid, d_grid, theta_grid, best, title, path):
    plt.figure(figsize=(8, 6))
    plt.imshow(grid, cmap='viridis', aspect='auto', origin='lower')
    plt.colorbar(label='F1')
    plt.xticks(range(len(theta_grid)), theta_grid)
    plt.yticks(range(len(d_grid)), d_grid)
    plt.xlabel(r'Angle limit $\theta_{max}$ (deg)')
    plt.ylabel(r'Distance limit $d_{max}$ (px)')
    plt.title(title)
    plt.scatter([theta_grid.index(best[1])], [d_grid.index(best[0])], color='red', marker='*', s=200,
                label=f'best d={best[0]}, θ={best[1]}')
    plt.legend()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────
# 6. 指定地圖的預測視覺化
# ─────────────────────────────────────────────
def visualize_baseline_A_map(map_data, d_max, theta_max, output_dir):
    """把預測的地名路徑畫在標記地圖上"""
    map_id = os.path.splitext(map_data['file'])[0]
    img_file = os.path.join(DATA_DIR, "map", "marked", f"{map_id}.png")
    if not os.path.exists(img_file):
        img_file = os.path.join(DATA_DIR, "map", f"{map_id}.png")
    if not os.path.exists(img_file):
        print(f"⚠️ 找不到地圖影像 {map_id}.png，略過視覺化")
        return
    coords = map_data['coords']
    places = run_static_greedy_extraction(list(coords), coords, d_max, theta_max)
    img = plt.imread(img_file)
    h, w = img.shape[:2]
    dpi = 150
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax.imshow(img)
    for path in places:
        xs = [coords[n][0] for n in path]
        ys = [coords[n][1] for n in path]
        ax.plot(xs, ys, color='#2ecc71', linewidth=3.5, alpha=0.85, zorder=2)
        ax.scatter(xs, ys, color='#00ffff', s=60, edgecolors='black', linewidths=1.0, zorder=3, alpha=0.9)
    ax.axis('off')
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{map_id}_baseline_A.png")
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close()
    print(f"🎨 已輸出 {out_path}")


# ─────────────────────────────────────────────
# 7. 主程式
# ─────────────────────────────────────────────
def _int_list(s):
    return [int(v) for v in s.split(',') if v.strip()]


def main():
    ap = argparse.ArgumentParser(description="基準方法 A：距離與角度門檻的規則式分群")
    ap.add_argument("--d-grid", type=_int_list, default=DEFAULT_D_GRID, help="d_max 候選值（px），逗號分隔")
    ap.add_argument("--theta-grid", type=_int_list, default=DEFAULT_THETA_GRID, help="theta_max 候選值（度），逗號分隔")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() // 2), help="網格搜尋的程序數")
    ap.add_argument("--target-map", type=str, default=None, help="要畫出預測的地圖（多張以逗號分隔，例如 15619,15814）")
    ap.add_argument("--d-max", type=int, default=None, help="與 --theta-max、--target-map 併用時跳過網格搜尋，只畫圖")
    ap.add_argument("--theta-max", type=int, default=None)
    args = ap.parse_args()

    all_files = sorted(f for f in os.listdir(config.XML_DIR) if f.endswith(".xml") and not f.startswith("B-"))
    train_files, val_files, test_files = split_maps(all_files, config)
    train_files = [f for f in train_files if f not in config.SKIP_FILES]
    test_files = [f for f in test_files if f not in config.SKIP_FILES]
    print(f"資料切分：訓練 {len(train_files)} / 驗證 {len(val_files)} / 測試 {len(test_files)}")

    # 建圖需要 scaler 與詞彙表（規則本身不使用），只用訓練切分擬合
    print("擬合正規化參數與詞彙表（訓練切分）...")
    fit_global_scaler(train_files)
    build_global_vocab(train_files)
    vis_dir = os.path.join(config.OUTPUT_DIR, "baseline_A_vis")
    targets = ([n.strip() if n.strip().endswith('.xml') else f"{n.strip()}.xml" for n in args.target_map.split(',')]
               if args.target_map else [])

    if targets and args.d_max is not None and args.theta_max is not None:
        for f in targets:
            visualize_baseline_A_map(load_gt_map(f), args.d_max, args.theta_max, vis_dir)
        return

    # ── GT 框：訓練切分上選參數，測試集評估 ──
    train_maps = [load_gt_map(f) for f in tqdm(train_files, desc="讀取訓練切分（標註）")]
    gt_d, gt_t, gt_search_f1, gt_grid = grid_search(
        train_maps, _gt_task, args.d_grid, args.theta_grid, args.workers, "GT 框網格搜尋")
    test_gt_maps = [load_gt_map(f) for f in tqdm(test_files, desc="讀取測試集（標註）")]
    gt_p, gt_r, gt_f1 = score_gt(test_gt_maps, gt_d, gt_t)

    # ── 端到端：驗證集 OCR 輸入上選參數，測試集評估 ──
    val_gt_dir = os.path.join(DATA_DIR, "xml_revised")
    val_maps = [m for m in (load_e2e_map(f, config.VAL_OCR_DIR, val_gt_dir)
                            for f in tqdm(val_files, desc="讀取驗證集（OCR）")) if m]
    e2e_d, e2e_t, e2e_search_f1, e2e_grid = grid_search(
        val_maps, _e2e_task, args.d_grid, args.theta_grid, args.workers, "端到端網格搜尋")
    ocr_files = sorted(f for f in os.listdir(config.OCR_XML_DIR) if f.endswith('.xml'))
    test_e2e_maps = [m for m in (load_e2e_map(f, config.OCR_XML_DIR, config.GT_XML_DIR)
                                 for f in tqdm(ocr_files, desc="讀取測試集（OCR）")) if m]
    e2e = score_e2e(test_e2e_maps, e2e_d, e2e_t)

    # ── 輸出 ──
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    save_heatmap(gt_grid, args.d_grid, args.theta_grid, (gt_d, gt_t),
                 'Baseline A: GT-box grid search (train split)', os.path.join(config.OUTPUT_DIR, 'baseline_A_grid_gt.png'))
    save_heatmap(e2e_grid, args.d_grid, args.theta_grid, (e2e_d, e2e_t),
                 'Baseline A: end-to-end grid search (validation set)', os.path.join(config.OUTPUT_DIR, 'baseline_A_grid_e2e.png'))
    results = {
        'split': {'train': len(train_files), 'val': len(val_maps), 'test_gt': len(test_files), 'test_e2e': len(test_e2e_maps)},
        'grid': {'d_max': args.d_grid, 'theta_max': args.theta_grid},
        'gt': {'d_max': gt_d, 'theta_max': gt_t, 'search_f1_train': gt_search_f1,
               'test': {'precision': gt_p, 'recall': gt_r, 'f1': gt_f1}},
        'e2e': {'d_max': e2e_d, 'theta_max': e2e_t, 'search_strict_f1_val': e2e_search_f1, 'test': e2e},
    }
    out_path = os.path.join(config.OUTPUT_DIR, 'baseline_A_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    c = e2e['counts']
    print("\n" + "=" * 60)
    print("基準方法 A（距離與角度門檻）測試集結果")
    print("=" * 60)
    print(f"GT 框   ：d_max={gt_d}, theta_max={gt_t}（訓練切分 F1 {gt_search_f1:.4f}）")
    print(f"  P {gt_p:.4f} / R {gt_r:.4f} / F1 {gt_f1:.4f}")
    print(f"端到端  ：d_max={e2e_d}, theta_max={e2e_t}（驗證集 Strict F1 {e2e_search_f1:.4f}）")
    for mode in ('strict', 'relaxed'):
        s = e2e[mode]
        print(f"  {mode:<8}P {s['precision']:.4f} / R {s['recall']:.4f} / F1 {s['f1']:.4f}")
    print(f"  Type A {c['TypeA']} / Type B {c['TypeB']} / Type C {c['TypeC']} / FP {c['FP']}（地名 {c['gt']}、預測 {c['pred']}）")
    print(f"結果：{out_path}")

    if targets:
        cached = {m['file']: m for m in train_maps + test_gt_maps}
        for f in targets:
            visualize_baseline_A_map(cached[f] if f in cached else load_gt_map(f), gt_d, gt_t, vis_dir)


if __name__ == "__main__":
    main()
