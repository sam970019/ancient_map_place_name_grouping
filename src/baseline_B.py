"""
baseline_B.py — 成對分類基準方法（Pairwise Classification Baseline）
====================================================================
用來回答「GNN 的訊息傳遞是否比只看兩個字的成對特徵更有效」。除了沒有訊息傳遞，
其餘一律與 GNN 相同：

  - 資料切分：data.split_maps（訓練 / 驗證 / 測試與 GNN 完全相同）
  - 候選字對與標籤：data.build_data（不設距離限制；正樣本是同一地名中閱讀順序相鄰的
    有向字對，其餘字對皆為負樣本）
  - 正規化參數與字元接續表：只用訓練圖建立
  - 訓練樣本：與 GNN 每個 epoch 相同的抽樣規則（正樣本全留、難負樣本全留（NEG_KEEP_HARD）、
    其餘負樣本抽 NEG_SAMPLE_RATIO），以 seed 固定、只抽一次
  - 特徵：兩個字的標準化 (x, y, w, h)、與 GNN 相同的邊幾何特徵（EXTRA_EDGE_FEATURES）；
    USE_TEXT_EMBED=1 時另加兩個字的 FastText 向量（各以 PCA 降到 32 維，只用訓練圖的字擬合）
    與兩者的餘弦相似度。文字相關設定跟著環境變數走，可與最終 GNN 配置一致。
  - 閾值：GT 框評估的閾值在驗證集的標註字框上選，E2E 閾值在驗證集的 OCR 上選；
    測試集只評估一次。分群規則與 GNN 相同（機率 > 閾值者相連，取連通分量，排除單字群集）。
  - 計分：GT 框層級用 metrics.calculate_exact_match_metrics，E2E 用 e2e_common.evaluate_map_iou，
    與 GNN、前人方法完全相同。

模型：
  xgb  XGBoost（hist），以驗證集抽樣字對的 logloss 做提前停止（最多 1000 棵、50 輪無改善即停）
  rf   隨機森林（100 棵、深度 12、class_weight=balanced）

輸出：DATA_DIR/baselines/baselineB_<model>_s<seed>[_<tag>]/
  results.json、feature_importance.csv、val_gt_scan.png、val_e2e_scan.png

執行方式（與 GNN 正式實驗相同的抽樣設定）：
  cd ancient_map_ai/src
  NEG_SAMPLE_RATIO=1/6 NEG_KEEP_HARD=1 python baseline_B.py --model xgb --seed 28
  NEG_SAMPLE_RATIO=1/6 NEG_KEEP_HARD=1 python baseline_B.py --model rf --seed 28
  （要拿掉文字資訊時，另加與 GNN 相同的 USE_TEXT_EMBED=0 與 EXTRA_EDGE_FEATURES=...）
小規模測試：MAP_NUM=10 VAL_NUM=2 ... python baseline_B.py --model rf --limit-test-maps 2 --tag smoke
"""

import argparse
import json
import os
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import config
import data as data_module
from data import (
    DATA_DIR, THRESHOLD_GRID, split_maps, read_xml_from, build_graph, build_data,
    build_global_vocab, load_fasttext_embeddings, fit_global_scaler, build_text_prior,
)
from e2e_common import read_xml_with_encoding_fix, parse_gt_clusters_from_xml, evaluate_map_iou
from metrics import check_gt_clusters, calculate_exact_match_metrics

PCA_DIM = 32
# 分類器以 scale_pos_weight / class_weight 加重稀少的正樣本，輸出機率整體偏高，最佳閾值常超過 0.95，
# 所以在 GNN 共用的 0.10~0.95 之上再補高段（GNN 的最佳閾值都在範圍內部，不需要）。
B_THRESHOLD_GRID = THRESHOLD_GRID + [0.96, 0.97, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99995, 0.99999]


# ─────────────────────────────────────────────
# 特徵
# ─────────────────────────────────────────────
class PairFeaturizer:
    """把一張圖的候選字對轉成特徵矩陣；文字部分依 USE_TEXT_EMBED 決定是否加入。"""

    def __init__(self, embedding_matrix, train_label_ids, seed):
        self.use_text = bool(getattr(config, 'USE_TEXT_EMBED', True))
        self.names = (['u_x', 'u_y', 'u_w', 'u_h', 'v_x', 'v_y', 'v_w', 'v_h']
                      + [f'edge_{n}' for n in data_module.edge_feature_names()])
        if self.use_text:
            from sklearn.decomposition import PCA
            ids = np.unique(train_label_ids)
            ids = ids[np.linalg.norm(embedding_matrix[ids], axis=1) > 0]   # 未收錄字（零向量）不參與擬合
            pca = PCA(n_components=PCA_DIM, random_state=seed).fit(embedding_matrix[ids])
            self.emb_pca = pca.transform(embedding_matrix).astype(np.float32)
            self.emb_pca[np.linalg.norm(embedding_matrix, axis=1) == 0] = 0.0
            norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
            self.emb_unit = np.divide(embedding_matrix, norms, out=np.zeros_like(embedding_matrix), where=norms > 0)
            self.pca_explained = float(pca.explained_variance_ratio_.sum())
            self.names += ([f'u_text_pc{i}' for i in range(PCA_DIM)] + [f'v_text_pc{i}' for i in range(PCA_DIM)]
                           + ['text_cosine'])

    def __call__(self, td, rows=None):
        idx = td.edge_label_index.numpy()
        attr = td.edge_label_attr.numpy()
        if rows is not None:
            idx, attr = idx[:, rows], attr[rows]
        x = td.x.numpy()
        u, v = idx[0], idx[1]
        parts = [x[u], x[v], attr]
        if self.use_text:
            lid = td.label_id.numpy()
            lu, lv = lid[u], lid[v]
            parts += [self.emb_pca[lu], self.emb_pca[lv],
                      np.sum(self.emb_unit[lu] * self.emb_unit[lv], axis=1, keepdims=True)]
        return np.concatenate(parts, axis=1).astype(np.float32)


def sample_rows(td, rng):
    """與 train.subsample_candidate_edges 相同的規則：正樣本全留、難負樣本全留、其餘負樣本抽 ratio。"""
    label = td.edge_label.numpy()
    hard = td.edge_label_hard.numpy()
    ratio = float(getattr(config, 'NEG_SAMPLE_RATIO', 1.0))
    keep_hard = bool(getattr(config, 'NEG_KEEP_HARD', False)) and ratio < 1.0
    pos = np.flatnonzero(label == 1)
    if keep_hard:
        neg_keep = np.flatnonzero((label == 0) & hard)
        hr = float(getattr(config, 'HARD_NEG_RATIO', 1.0))
        if hr < 1.0:
            neg_keep = rng.permutation(neg_keep)[:int(round(neg_keep.size * hr))]
        neg = np.flatnonzero((label == 0) & ~hard)
    else:
        neg_keep = np.empty(0, dtype=np.int64)
        neg = np.flatnonzero(label == 0)
    neg_sel = rng.permutation(neg)[:int(round(neg.size * ratio))]
    return np.sort(np.concatenate([pos, neg_keep, neg_sel]))


# ─────────────────────────────────────────────
# 分群
# ─────────────────────────────────────────────
def clusters_from(pairs, probs, n_nodes, threshold, text_idx):
    """機率 > 閾值的字對視為相連，取連通分量並排除單字群集（與 metrics.check_pred_clusters 相同規則）。"""
    m = probs > threshold
    if not m.any():
        return []
    adj = coo_matrix((np.ones(int(m.sum())), (pairs[0][m], pairs[1][m])), shape=(n_nodes, n_nodes))
    _, labels = connected_components(adj, directed=False, return_labels=True)
    order = np.argsort(labels)
    _, starts, counts = np.unique(labels[order], return_index=True, return_counts=True)
    out = []
    for s, c in zip(starts, counts):
        if c < 2:
            continue
        nodes = [int(n) for n in order[s:s + c]]
        chars, xs, ys = [], [], []
        for n in nodes:
            if n in text_idx:
                _, ch, x, y = text_idx[n]
                chars.append(ch); xs.append(x); ys.append(y)
            else:
                chars.append('?')
        # 連通分量只是字的集合，這裡的字序沒有意義（兩種評估都只比對集合）
        out.append({'chars': tuple(chars),
                    'centroid': (np.mean(xs) if xs else 0, np.mean(ys) if ys else 0),
                    'node_indices': nodes})
    return out


def gt_level_metrics(maps, threshold):
    """GT 框層級：所有地圖合併計分（與 GNN 的 get_place_metrics 相同），另附連線層級。"""
    preds, gts = [], []
    tp = fp = fn = 0
    for m in maps:
        preds += clusters_from(m['pairs'], m['probs'], m['n'], threshold, m['text_idx'])
        gts += m['gt']
        pb = m['probs'] > threshold
        tp += int(np.sum(pb & (m['label'] == 1)))
        fp += int(np.sum(pb & (m['label'] == 0)))
        fn += int(np.sum(~pb & (m['label'] == 1)))
    if preds and gts:
        p, r, f1, total = calculate_exact_match_metrics(preds, gts)
    else:
        p = r = f1 = 0.0
        total = len(gts)
    ep = tp / (tp + fp) if tp + fp else 0.0
    er = tp / (tp + fn) if tp + fn else 0.0
    ef = 2 * ep * er / (ep + er) if ep + er else 0.0
    return {'precision': p, 'recall': r, 'f1': f1, 'gt_total': total,
            'edge_precision': ep, 'edge_recall': er, 'edge_f1': ef}


def e2e_metrics(maps, threshold):
    """E2E：逐圖以 evaluate_map_iou 計分後合計（與 e2e_evaluate.e2e_evaluate 相同的彙總規則）。"""
    tp_s = tp_b = fn_a = fn_c = fp = n_gt = n_pred = 0
    per_map = {}
    for m in maps:
        pred = clusters_from(m['pairs'], m['probs'], m['n'], threshold, m['text_idx'])
        # 每張圖都計入（預測或 GT 為空也一樣），與 e2e_evaluate 相同
        a, b, c, d, e, _, _ = evaluate_map_iou(pred, m['gt'], m['ocr_boxes'])
        tp_s += a; tp_b += b; fn_a += c; fn_c += d; fp += e
        n_gt += len(m['gt']); n_pred += len(pred)
        n_all = len(pred) + len(m['gt'])
        f = lambda tp: (2 * tp / n_all if n_all else 0.0)   # P=tp/pred、R=tp/gt 的調和平均
        per_map[m['file']] = {'strict_f1': f(a), 'relaxed_f1': f(a + b)}

    def prf(tp):
        p = tp / n_pred if n_pred else 0.0
        r = tp / n_gt if n_gt else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)
    sp, sr, sf = prf(tp_s)
    rp, rr, rf = prf(tp_s + tp_b)
    return {'strict': {'precision': sp, 'recall': sr, 'f1': sf},
            'relaxed': {'precision': rp, 'recall': rr, 'f1': rf},
            'counts': {'TP_strict': tp_s, 'TypeB': tp_b, 'TypeA': fn_a, 'TypeC': fn_c,
                       'FP': fp, 'gt': n_gt, 'pred': n_pred},
            'per_map': per_map}


# ─────────────────────────────────────────────
# 預測
# ─────────────────────────────────────────────
def predict_gt_maps(files, xml_dir, clf, feat, keep_features=False):
    """標註字框上的地圖：全部候選字對預測一次，存下機率供閾值掃描（keep_features 時另存特徵矩陣）。"""
    out = []
    for f in files:
        data, text_idx, _, pne, ocr = build_graph(read_xml_from(f, xml_dir), None)
        td = build_data(data, pne, ocr, is_test=True)
        X = feat(td)
        out.append({'file': f, 'pairs': td.edge_label_index.numpy(), 'n': td.x.size(0),
                    'probs': clf.predict_proba(X)[:, 1], 'label': td.edge_label.numpy(),
                    'text_idx': text_idx, 'gt': check_gt_clusters(td, text_idx)})
        if keep_features:
            out[-1]['X'] = X
    return out


def permutation_importance(maps, clf, names, threshold, seed):
    """
    與 GNN 的置換重要度相同的做法：把一個特徵（或一組）在所有驗證字對之間隨機打亂，
    看字對層級 average precision 與地名層級 F1 掉多少。兩端字的同一個節點特徵（u_x 與 v_x）視為一組。
    """
    from sklearn.metrics import average_precision_score
    X = np.concatenate([m['X'] for m in maps])
    y = np.concatenate([m['label'] for m in maps])
    sizes = np.cumsum([len(m['label']) for m in maps])[:-1]
    rng = np.random.default_rng(seed)

    def score(probs):
        parts = np.split(probs, sizes)
        scored = [dict(m, probs=p) for m, p in zip(maps, parts)]
        return average_precision_score(y, probs), gt_level_metrics(scored, threshold)['f1']

    ap0, f0 = score(clf.predict_proba(X)[:, 1])
    groups = [(n, [i]) for i, n in enumerate(names) if n.startswith('edge_')]
    groups += [(f'node_{c}', [names.index(f'u_{c}'), names.index(f'v_{c}')]) for c in 'xywh']
    text_cols = [i for i, n in enumerate(names) if 'text' in n]
    if text_cols:
        groups.append(('text_all', text_cols))
    rows = []
    for g, cols in groups:
        Xp = X.copy()
        for c in cols:
            Xp[:, c] = Xp[rng.permutation(len(Xp)), c]
        ap, f1 = score(clf.predict_proba(Xp)[:, 1])
        rows.append({'feature': g, 'ap_drop': ap0 - ap, 'f1_drop': f0 - f1})
        print(f"  置換 {g:<16} AP 降 {ap0 - ap:+.4f}  地名 F1 降 {f0 - f1:+.4f}")
    return {'ap0': ap0, 'f1_0': f0, 'rows': rows}


def predict_ocr_maps(files, ocr_dir, gt_dir, clf, feat):
    """OCR 輸出上的地圖（E2E）：建圖方式與 e2e_evaluate 相同，GT 由標註 XML 解析。"""
    out = []
    for f in files:
        gt_path = os.path.join(gt_dir, f)
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"找不到 {f} 的 GT：{gt_path}")
        data, text_idx, _, pne, ocr = build_graph(read_xml_with_encoding_fix(os.path.join(ocr_dir, f)), None)
        td = build_data(data, pne, ocr, is_test=True)
        out.append({'file': f, 'pairs': td.edge_label_index.numpy(), 'n': td.x.size(0),
                    'probs': clf.predict_proba(feat(td))[:, 1], 'text_idx': text_idx,
                    'gt': parse_gt_clusters_from_xml(gt_path), 'ocr_boxes': ocr})
    return out


def plot_scan(rows, keys, title, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    ths = [r['threshold'] for r in rows]
    for k in keys:
        plt.plot(ths, [r[k] for r in rows], label=k)
    plt.xlabel('threshold'); plt.ylabel('score'); plt.title(title)
    plt.grid(alpha=0.4); plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


# ─────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Baseline B：成對分類（新協定）")
    ap.add_argument("--model", choices=("xgb", "rf"), default="xgb")
    ap.add_argument("--seed", type=int, default=config.TRAIN_SEED)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit-test-maps", type=int, default=0, help="只供小規模測試：E2E 測試圖只取前 N 張")
    ap.add_argument("--perm-importance", action="store_true", help="在驗證集上計算置換重要度（與 GNN 相同的做法）")
    ap.add_argument("--dump-test-clusters", action="store_true", help="存下測試集 E2E 每張圖預測出的地名（字框索引），供錯誤分析")
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    name = f"baselineB_{args.model}_s{args.seed}" + (f"_{args.tag}" if args.tag else "")
    out_dir = os.path.join(DATA_DIR, "baselines", name)
    os.makedirs(out_dir, exist_ok=True)

    # 1. 切分、字元接續表、正規化參數、詞彙表（與 train.py 相同的順序與範圍）
    files = sorted(f for f in os.listdir(config.XML_DIR) if f.endswith(".xml") and not f.startswith("B-"))
    train_files, val_files, test_files = split_maps(files, config)
    if not val_files:
        raise RuntimeError("沒有驗證集（VAL_NUM=0）。Baseline B 需要驗證集來選閾值與提前停止。")
    train_files = [f for f in train_files if f not in config.SKIP_FILES]
    if 'bigram_lp' in data_module._extra_edge_features():
        build_text_prior(train_files, config.TRAIN_XML_DIR)
    else:
        data_module.text_prior = None
    fit_global_scaler(train_files)
    build_global_vocab(files)
    _, _, embedding_matrix = load_fasttext_embeddings()
    print(f"[{name}] 訓練 {len(train_files)} / 驗證 {len(val_files)} / 測試 {len(test_files)} 張 | "
          f"邊特徵 {config.EDGE_FEATURE_DIM} 維 | USE_TEXT_EMBED={config.USE_TEXT_EMBED} | "
          f"NEG_SAMPLE_RATIO={config.NEG_SAMPLE_RATIO:.4f} NEG_KEEP_HARD={config.NEG_KEEP_HARD}")
    if config.NEG_SAMPLE_RATIO >= 1.0:
        print("⚠️  NEG_SAMPLE_RATIO=1：不抽樣，訓練資料約 4,000 萬組字對，記憶體與時間都會很大")

    # 2. 訓練資料：逐張圖建圖後依 GNN 的規則抽樣，只為抽到的字對產生特徵
    graphs, label_ids = [], []
    for f in train_files:
        data, _, _, pne, ocr = build_graph(read_xml_from(f, config.TRAIN_XML_DIR), None)
        td = build_data(data, pne, ocr, is_test=False)
        graphs.append(td)
        label_ids.append(td.label_id.numpy())
    feat = PairFeaturizer(embedding_matrix, np.concatenate(label_ids), args.seed)
    X_parts, y_parts = [], []
    for td in graphs:
        rows = sample_rows(td, rng)
        X_parts.append(feat(td, rows))
        y_parts.append(td.edge_label.numpy()[rows])
    del graphs
    X_train, y_train = np.concatenate(X_parts), np.concatenate(y_parts).astype(np.int8)
    del X_parts, y_parts
    n_pos, n_neg = int(y_train.sum()), int((y_train == 0).sum())
    print(f"訓練樣本 {len(y_train):,}（正 {n_pos:,} / 負 {n_neg:,}），特徵 {X_train.shape[1]} 維"
          + (f"，文字 PCA 解釋變異 {feat.pca_explained:.1%}" if feat.use_text else ""))

    # 3. 模型
    if args.model == "xgb":
        from xgboost import XGBClassifier
        # 提前停止用的驗證資料：驗證圖以同樣規則抽樣，分布與訓練資料一致
        Xv, yv = [], []
        for f in val_files:
            data, _, _, pne, ocr = build_graph(read_xml_from(f, config.TRAIN_XML_DIR), None)
            td = build_data(data, pne, ocr, is_test=False)
            rows = sample_rows(td, rng)
            Xv.append(feat(td, rows)); yv.append(td.edge_label.numpy()[rows])
        Xv, yv = np.concatenate(Xv), np.concatenate(yv).astype(np.int8)
        clf = XGBClassifier(n_estimators=1000, max_depth=6, learning_rate=0.1, subsample=0.8,
                            tree_method="hist", scale_pos_weight=n_neg / max(n_pos, 1),
                            eval_metric="logloss", early_stopping_rounds=50,
                            random_state=args.seed, n_jobs=args.n_jobs)
        clf.fit(X_train, y_train, eval_set=[(Xv, yv)], verbose=50)
        model_info = {'best_iteration': int(clf.best_iteration), 'n_estimators_max': 1000,
                      'max_depth': 6, 'learning_rate': 0.1, 'subsample': 0.8,
                      'scale_pos_weight': n_neg / max(n_pos, 1), 'early_stopping_rounds': 50}
        del Xv, yv
    else:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced",
                                     random_state=args.seed, n_jobs=args.n_jobs)
        clf.fit(X_train, y_train)
        model_info = {'n_estimators': 100, 'max_depth': 12, 'class_weight': 'balanced'}
    del X_train, y_train
    print(f"模型訓練完成（{time.time() - t0:.0f}s）：{model_info}")

    # 4. 驗證集選閾值（GT 框、E2E 各自選）
    val_gt = predict_gt_maps(val_files, config.TRAIN_XML_DIR, clf, feat, keep_features=args.perm_importance)
    gt_scan = [dict(threshold=th, **gt_level_metrics(val_gt, th)) for th in B_THRESHOLD_GRID]
    gt_best = max(gt_scan, key=lambda r: r['f1'])
    perm = None
    if args.perm_importance:
        print("計算置換重要度（驗證集）...")
        perm = permutation_importance(val_gt, clf, feat.names, gt_best['threshold'], args.seed)
    del val_gt
    val_ocr = predict_ocr_maps(val_files, config.VAL_OCR_DIR, config.XML_DIR, clf, feat)
    e2e_scan = []
    for th in B_THRESHOLD_GRID:
        m = e2e_metrics(val_ocr, th)
        e2e_scan.append({'threshold': th, 'strict_f1': m['strict']['f1'], 'relaxed_f1': m['relaxed']['f1']})
    e2e_best = max(e2e_scan, key=lambda r: r['strict_f1'])
    del val_ocr
    print(f"驗證集選定：GT 框閾值 {gt_best['threshold']:.3f}（F1 {gt_best['f1']:.4f}）| "
          f"E2E 閾值 {e2e_best['threshold']:.3f}（Strict {e2e_best['strict_f1']:.4f}）")
    edge = [k for k, b in (('GT 框', gt_best), ('E2E', e2e_best)) if b['threshold'] in (B_THRESHOLD_GRID[0], B_THRESHOLD_GRID[-1])]
    if edge:
        print(f"⚠️  {'、'.join(edge)} 的最佳閾值落在掃描範圍邊緣，真正的最佳值可能在範圍之外")

    # 5. 測試集只評估一次
    test_gt = gt_level_metrics(predict_gt_maps(test_files, config.TEST_XML_DIR, clf, feat), gt_best['threshold'])
    ocr_files = sorted(f for f in os.listdir(config.OCR_XML_DIR) if f.endswith('.xml'))
    if args.limit_test_maps:
        ocr_files = ocr_files[:args.limit_test_maps]
    test_ocr = predict_ocr_maps(ocr_files, config.OCR_XML_DIR, config.GT_XML_DIR, clf, feat)
    test_e2e = e2e_metrics(test_ocr, e2e_best['threshold'])
    if args.dump_test_clusters:
        dump = {m['file']: [c['node_indices'] for c in clusters_from(m['pairs'], m['probs'], m['n'], e2e_best['threshold'], m['text_idx'])]
                for m in test_ocr}
        with open(os.path.join(out_dir, 'test_e2e_clusters.json'), 'w', encoding='utf-8') as fh:
            json.dump({'threshold': e2e_best['threshold'], 'clusters': dump}, fh)
    del test_ocr
    c = test_e2e['counts']
    print(f"\n=== {name} 測試集 ===")
    print(f"GT 框:  P {test_gt['precision']:.4f}  R {test_gt['recall']:.4f}  F1 {test_gt['f1']:.4f}  "
          f"(Edge F1 {test_gt['edge_f1']:.4f}, 閾值 {gt_best['threshold']:.3f})")
    print(f"E2E ({len(ocr_files)} 張): Strict F1 {test_e2e['strict']['f1']:.4f}  Relaxed F1 {test_e2e['relaxed']['f1']:.4f}  "
          f"(閾值 {e2e_best['threshold']:.3f}) | Type A {c['TypeA']}  Type B {c['TypeB']}  Type C {c['TypeC']}  FP {c['FP']}")

    # 6. 輸出
    importance = sorted(zip(feat.names, clf.feature_importances_.tolist()), key=lambda t: -t[1])
    with open(os.path.join(out_dir, 'feature_importance.csv'), 'w', encoding='utf-8') as fh:
        fh.write('feature,importance\n')
        fh.writelines(f'{n},{v:.6f}\n' for n, v in importance)
    plot_scan(gt_scan, ['precision', 'recall', 'f1'], f'{name}: validation GT-box threshold scan',
              os.path.join(out_dir, 'val_gt_scan.png'))
    plot_scan(e2e_scan, ['strict_f1', 'relaxed_f1'], f'{name}: validation E2E threshold scan',
              os.path.join(out_dir, 'val_e2e_scan.png'))
    result = {
        'name': name, 'model': args.model, 'seed': args.seed, 'model_info': model_info,
        'protocol': {'train_maps': len(train_files), 'val_maps': val_files, 'test_maps': len(test_files),
                     'e2e_test_maps': len(ocr_files), 'edge_features': list(data_module.edge_feature_names()),
                     'use_text_embed': bool(config.USE_TEXT_EMBED), 'pca_dim': PCA_DIM if feat.use_text else 0,
                     'neg_sample_ratio': config.NEG_SAMPLE_RATIO, 'neg_keep_hard': bool(config.NEG_KEEP_HARD),
                     'nxn_edge_distance_limit': config.NXN_EDGE_DISTANCE_LIMIT,
                     'n_train_rows': n_pos + n_neg, 'n_train_pos': n_pos},
        'thresholds': {'gt': gt_best['threshold'], 'e2e': e2e_best['threshold'],
                       'grid': B_THRESHOLD_GRID,
                       'at_grid_edge': [b['threshold'] in (B_THRESHOLD_GRID[0], B_THRESHOLD_GRID[-1]) for b in (gt_best, e2e_best)]},
        'val_gt_scan': gt_scan, 'val_e2e_scan': e2e_scan,
        'test_gt': test_gt, 'test_e2e': test_e2e,
        'feature_importance_top20': importance[:20],
        'permutation_importance': perm,
        'seconds': round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, 'results.json'), 'w', encoding='utf-8') as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    print(f"結果已寫入 {out_dir}（共 {time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()
