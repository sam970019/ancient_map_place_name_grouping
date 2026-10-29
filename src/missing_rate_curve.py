"""
missing_rate_curve.py — 缺字率–表現退化曲線實驗
========================================================================
目的：量化「缺字率 p」對地名分群 Entity F1 的影響曲線。

方法：
  1. 使用乾淨 GT 訓練的最佳模型（不重訓，threshold 固定 config.PRED_THRESHOLD）
  2. 對 e2e 評估用的同一組 GT 地圖（GT_XML_DIR，與真實 OCR 53% 同一把尺）
     建立完美圖，再以合成節點刪除（重用 data.apply_node_dropout）模擬缺字率 p
  3. 每個 p 跑多個 random seed，報告 mean ± std

兩條評估軸（同一次推論同時計算）：
  - 曲線 B（主）：對「完整 GT」評分（IoU 匹配，與 e2e_evaluate 相同語意）
      → 含數學天花板效應，可與真實 OCR 的 53% 直接比較
  - 曲線 A（輔）：對「刪除後倖存 GT」評分（字集+中心點匹配）
      → 隔離「GNN 在倖存節點上的額外退化」（曲線版蝴蝶效應）

附加輸出：
  - 理論天花板線：mean over names of (1-p)^k（完整存活地名比例，recall 上限）
  - 真實 OCR 點 (p=0.0782, F1=0.5331) 標於圖上

執行方式：
  cd src
  python missing_rate_curve.py                          # 完整實驗
  python missing_rate_curve.py --p-grid 0,0.0782 --seeds 1 --limit-maps 3   # 冒煙測試

注意：本腳本為獨立實驗腳本，不修改任何既有檔案。
      parse_gt_clusters_from_xml / evaluate_map_iou 複製自 e2e_evaluate.py
      （該檔為頂層執行腳本，import 會觸發整條評估管線，故無法直接引用）。
"""

import argparse
import os
import re
import csv
import time
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

import config
import data as data_module
from data import (
    fit_global_scaler, build_graph, build_data, apply_node_dropout,
    build_global_vocab, load_fasttext_embeddings, DATA_DIR,
)
from metrics import check_pred_clusters, check_gt_clusters, calculate_exact_match_metrics

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


# ─────────────────────────────────────────────
# 0. 參數
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="缺字率–表現退化曲線實驗")
    parser.add_argument("--p-grid", type=str, default="0,0.02,0.05,0.0782,0.10,0.15,0.20,0.30",
                        help="缺字率網格，逗號分隔")
    parser.add_argument("--seeds", type=int, default=5,
                        help="每個 p (>0) 的隨機種子數量（p=0 無隨機性，只跑 1 次）")
    parser.add_argument("--limit-maps", type=int, default=0,
                        help="除錯用：只處理前 N 張地圖（0=全部）")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="輸出目錄（預設 config.OUTPUT_DIR）")
    parser.add_argument("--ocr-f1", type=float, default=0.5331,
                        help="真實 OCR 的 Strict F1，標於圖上")
    parser.add_argument("--ocr-p", type=float, default=0.0782,
                        help="真實 OCR 的缺字率，標於圖上")
    parser.add_argument("--no-plot", action="store_true", help="不產生曲線圖")
    return parser.parse_args()


# ─────────────────────────────────────────────
# 工具函式（複製自 e2e_evaluate.py，保持語意一致）
# ─────────────────────────────────────────────
def read_xml_with_encoding_fix(filepath: str):
    with open(filepath, 'rb') as f:
        raw = f.read()
    raw = re.sub(rb"encoding=['\"]ISO-8859-1['\"]", b"encoding='UTF-8'", raw)
    raw = re.sub(rb"encoding=['\"]iso-8859-1['\"]", b"encoding='UTF-8'", raw)
    return ET.fromstring(raw.decode('utf-8', errors='replace'))


def parse_gt_clusters_from_xml(gt_filepath: str) -> list:
    """解析 GT XML，回傳完整地名群集（複製自 e2e_evaluate.py）。"""
    try:
        tree = ET.parse(gt_filepath)
        root = tree.getroot()
    except ET.ParseError:
        with open(gt_filepath, 'rb') as f:
            raw = f.read()
        raw = re.sub(rb"encoding=['\"].*?['\"]", b"encoding='UTF-8'", raw)
        root = ET.fromstring(raw.decode('utf-8', errors='replace'))

    groups = {}
    image_elem = root.find('.//image')
    if image_elem is None:
        return []

    for box in image_elem.findall('box'):
        box_id = box.get('id', '0')
        if box_id == '0':
            continue
        label_elem = box.find('label')
        if label_elem is None or not label_elem.text:
            continue
        char = label_elem.text.strip()
        try:
            top = float(box.get('top', 0))
            left = float(box.get('left', 0))
            width = float(box.get('width', 0))
            height = float(box.get('height', 0))
        except ValueError:
            continue
        cx = left + width / 2
        cy = top + height / 2
        groups.setdefault(box_id, []).append({
            'char': char, 'cx': cx, 'cy': cy,
            'left': left, 'top': top, 'width': width, 'height': height
        })

    gt_clusters = []
    for group_id, items in groups.items():
        chars = tuple(item['char'] for item in items)
        cx = float(np.mean([item['cx'] for item in items]))
        cy = float(np.mean([item['cy'] for item in items]))
        gt_clusters.append({'chars': chars, 'centroid': (cx, cy), 'boxes': items})
    return gt_clusters


def evaluate_map_iou(pred_clusters, gt_clusters, ocr_boxes):
    """IoU 匹配的實體級評估（複製自 e2e_evaluate.py，語意完全一致）。

    回傳 (tp_strict, tp_type_b, fn_type_a, fn_type_c, fp, pred_status, gt_status)
    在合成刪除情境下：fn_type_a = 被刪到字的地名（數學上必死），
    fn_type_c = 節點完整但 GNN 分群錯誤。
    """
    total_gt = len(gt_clusters)
    total_pred = len(pred_clusters)

    tp_strict = 0
    tp_type_b = 0
    fn_type_a = 0
    fn_type_c = 0

    gt_status = ['FN_type_c'] * total_gt
    pred_status = ['FP'] * total_pred

    for i, gt_c in enumerate(gt_clusters):
        gt_c['ocr_match_indices'] = set()
        gt_c['ocr_typo'] = False
        gt_c['ocr_miss'] = False

        for gt_box in gt_c['boxes']:
            best_iou = 0
            best_ocr_idx = -1
            gl, gt_top, gw, gh = gt_box['left'], gt_box['top'], gt_box['width'], gt_box['height']

            for o_idx, o_box in enumerate(ocr_boxes):
                ol, ot, ow, oh = o_box['x'], o_box['y'], o_box['w'], o_box['h']
                inter_l = max(gl, ol)
                inter_t = max(gt_top, ot)
                inter_r = min(gl + gw, ol + ow)
                inter_b = min(gt_top + gh, ot + oh)
                inter_area = max(0, inter_r - inter_l) * max(0, inter_b - inter_t)

                if inter_area > 0:
                    gt_area = gw * gh
                    ocr_area = ow * oh
                    iou = inter_area / (gt_area + ocr_area - inter_area)
                    if iou > best_iou:
                        best_iou = iou
                        best_ocr_idx = o_idx

            if best_iou >= 0.35:
                gt_c['ocr_match_indices'].add(best_ocr_idx)
                if gt_box['char'] != ocr_boxes[best_ocr_idx]['text']:
                    gt_c['ocr_typo'] = True
            else:
                gt_c['ocr_miss'] = True

        if gt_c['ocr_miss']:
            gt_status[i] = 'FN_type_a'

    for i, gt_c in enumerate(gt_clusters):
        if gt_status[i] == 'FN_type_a':
            fn_type_a += 1
            continue

        expected_ocr_nodes = gt_c['ocr_match_indices']

        for j, pred_c in enumerate(pred_clusters):
            if pred_status[j] != 'FP':
                continue
            if set(pred_c['node_indices']) == expected_ocr_nodes:
                if gt_c['ocr_typo']:
                    pred_status[j] = 'TP_type_b'
                    gt_status[i] = 'TP_type_b'
                    tp_type_b += 1
                else:
                    pred_status[j] = 'TP_strict'
                    gt_status[i] = 'TP_strict'
                    tp_strict += 1
                break

        if gt_status[i] == 'FN_type_c':
            fn_type_c += 1

    fp = pred_status.count('FP')
    return tp_strict, tp_type_b, fn_type_a, fn_type_c, fp, pred_status, gt_status


def prf(tp, n_pred, n_gt):
    p = tp / n_pred if n_pred > 0 else 0.0
    r = tp / n_gt if n_gt > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


# ─────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────
def main():
    args = parse_args()
    p_grid = [float(x) for x in args.p_grid.split(',') if x.strip() != '']
    out_dir = args.out_dir or config.OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    start_time = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── 1. 載入模型（鏡射 e2e_evaluate.py 的載入邏輯）──
    model_path = os.path.join(config.MODEL_SAVE_DIR, f'trained_model_{config.EXPERIMENT_NAME}.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型檔案：{model_path}")

    loaded = torch.load(model_path, map_location=device, weights_only=False)

    # 關鍵參數與 bigram 表的還原（含向後相容規則）集中在 data.restore_checkpoint_config
    if data_module.restore_checkpoint_config(loaded, config) is not None:
        print("🛡️ 已載入 checkpoint 封存的關鍵架構參數。")

    # 詞彙表與 scaler 一律以 xml_revised 擬合（與訓練端相同；scaler 參數若已封存於 checkpoint 則直接沿用）
    vocab_xml_dir = os.path.join(DATA_DIR, "xml_revised")

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
    config.XML_DIR = original_xml_dir

    print("載入 FastText 嵌入...")
    embedding_dim, vocab_size, embedding_matrix = load_fasttext_embeddings()

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
    print(f"成功載入模型：{model_path}")
    threshold = getattr(config, 'PRED_THRESHOLD', 0.5)
    print(f"預測閾值：{threshold}")

    # ── 2. 載入 GT 地圖並建圖（與 e2e 同一組地圖：GT_XML_DIR 中對應 OCR 檔名者）──
    ocr_xml_files = sorted([
        f for f in os.listdir(config.OCR_XML_DIR) if f.endswith('.xml')
    ])
    if args.limit_maps > 0:
        ocr_xml_files = ocr_xml_files[:args.limit_maps]

    map_cache = []  # 每張圖: dict(file, data, text_idx, place_name_edge, ocr_results, gt_clusters)
    for fname in tqdm(ocr_xml_files, desc="載入 GT 地圖並建圖"):
        gt_path = os.path.join(config.GT_XML_DIR, fname)
        if not os.path.exists(gt_path):
            print(f"⚠️  跳過 {fname}：GT 檔不存在")
            continue
        try:
            gt_root = read_xml_with_encoding_fix(gt_path)
            data, text_idx, _, place_name_edge, ocr_results = build_graph(gt_root, model)
        except Exception as e:
            print(f"⚠️  跳過 {fname}：建圖失敗 ({e})")
            continue
        gt_clusters = parse_gt_clusters_from_xml(gt_path)
        map_cache.append({
            'file': fname, 'data': data, 'text_idx': text_idx,
            'place_name_edge': place_name_edge, 'ocr_results': ocr_results,
            'gt_clusters': gt_clusters,
        })

    if not map_cache:
        raise RuntimeError("沒有任何 GT 地圖成功載入。")
    n_names_total = sum(len(m['gt_clusters']) for m in map_cache)
    print(f"共載入 {len(map_cache)} 張 GT 地圖，完整地名 {n_names_total} 個。")

    # 理論天花板用的地名長度列表
    name_lengths = np.array([len(c['chars']) for m in map_cache for c in m['gt_clusters']])

    # ── 3. 單次評估（給定 p 與 seed）──
    def run_once(p, seed):
        """回傳 dict：曲線 B (vs 完整 GT) 與曲線 A (vs 倖存 GT) 的指標。"""
        np.random.seed(seed)
        tot = dict(tp_s=0, tp_b=0, fn_a=0, fn_c=0, fp=0, n_pred=0, n_gt=0)
        surv_tp = 0
        surv_n_pred = 0
        surv_n_gt = 0

        for m in map_cache:
            data = m['data'].clone()
            if p > 0:
                data, pne, kept_ocr = apply_node_dropout(
                    data, m['ocr_results'], m['place_name_edge'], p_drop=p
                )
            else:
                pne, kept_ocr = m['place_name_edge'], m['ocr_results']

            test_data = build_data(data, pne, kept_ocr, is_test=True)

            # 刪除後節點已重編號，重建 text_idx（格式與 build_graph 一致: idx -> (id,text,x,y)）
            new_text_idx = {
                i: (it['id'], it['text'], it['x'], it['y'])
                for i, it in enumerate(kept_ocr)
            }

            pred_clusters, _preds = check_pred_clusters(
                model, test_data, device, new_text_idx, threshold=threshold
            )

            # 曲線 B：對完整 GT（IoU 匹配，與 e2e 同語意）
            gt_clusters = [
                {'chars': c['chars'], 'centroid': c['centroid'],
                 'boxes': [dict(b) for b in c['boxes']]}
                for c in m['gt_clusters']
            ]  # 淺拷貝防止 evaluate_map_iou 在 cluster dict 上留狀態
            tp_s, tp_b, fn_a, fn_c, fp, _, _ = evaluate_map_iou(
                pred_clusters, gt_clusters, kept_ocr
            )
            tot['tp_s'] += tp_s
            tot['tp_b'] += tp_b
            tot['fn_a'] += fn_a
            tot['fn_c'] += fn_c
            tot['fp'] += fp
            tot['n_pred'] += len(pred_clusters)
            tot['n_gt'] += len(gt_clusters)

            # 曲線 A：對倖存 GT（字集+中心點匹配）
            survivor_gt = check_gt_clusters(test_data, new_text_idx)
            if pred_clusters and survivor_gt:
                sp, sr, sf1, s_total = calculate_exact_match_metrics(
                    pred_clusters, survivor_gt
                )
                # calculate_exact_match_metrics 回傳的是比例；還原成計數以便跨圖彙總
                surv_tp += sr * s_total
                surv_n_gt += s_total
                surv_n_pred += len(pred_clusters)
            else:
                surv_n_gt += len(survivor_gt)
                surv_n_pred += len(pred_clusters)

        b_p, b_r, b_f1 = prf(tot['tp_s'], tot['n_pred'], tot['n_gt'])
        rel_p, rel_r, rel_f1 = prf(tot['tp_s'] + tot['tp_b'], tot['n_pred'], tot['n_gt'])
        a_p, a_r, a_f1 = prf(surv_tp, surv_n_pred, surv_n_gt)
        return {
            'B_precision': b_p, 'B_recall': b_r, 'B_f1': b_f1,
            'B_relaxed_f1': rel_f1,
            'A_precision': a_p, 'A_recall': a_r, 'A_f1': a_f1,
            'n_type_a': tot['fn_a'], 'n_type_c': tot['fn_c'],
            'n_fp': tot['fp'], 'n_tp_strict': tot['tp_s'], 'n_gt': tot['n_gt'],
        }

    # ── 4. 掃描 (p, seed) ──
    csv_path = os.path.join(out_dir, f'missing_rate_curve_{config.EXPERIMENT_NAME}.csv')
    fieldnames = ['p', 'seed', 'B_precision', 'B_recall', 'B_f1', 'B_relaxed_f1',
                  'A_precision', 'A_recall', 'A_f1',
                  'n_type_a', 'n_type_c', 'n_fp', 'n_tp_strict', 'n_gt']
    results = []

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        runs = []
        for p in p_grid:
            seeds = [0] if p == 0 else list(range(args.seeds))
            for s in seeds:
                runs.append((p, s))

        pbar = tqdm(runs, desc="掃描 (p, seed)")
        for p, s in pbar:
            pbar.set_postfix({'p': f'{p:.4f}', 'seed': s})
            res = run_once(p, s)
            row = {'p': p, 'seed': s, **{k: round(v, 6) if isinstance(v, float) else v
                                          for k, v in res.items()}}
            results.append(row)
            writer.writerow(row)
            f.flush()
            tqdm.write(
                f"  p={p:<7.4f} seed={s}  B_F1={res['B_f1']:.4f}  A_F1={res['A_f1']:.4f}  "
                f"(TypeA={res['n_type_a']}, TypeC={res['n_type_c']})"
            )

    print(f"\n📄 結果 CSV：{csv_path}")

    # ── 5. 彙總與 sanity checks ──
    def agg(p, key):
        vals = [r[key] for r in results if r['p'] == p]
        return np.mean(vals), np.std(vals)

    print("\n==================== 彙總 ====================")
    print(f"{'p':>8} | {'B_F1 (vs完整GT)':>18} | {'A_F1 (vs倖存GT)':>18} | {'天花板(存活比例)':>14}")
    for p in p_grid:
        bm, bs = agg(p, 'B_f1')
        am, as_ = agg(p, 'A_f1')
        ceiling = float(np.mean((1 - p) ** name_lengths))
        print(f"{p:>8.4f} | {bm:>10.4f} ± {bs:.4f} | {am:>10.4f} ± {as_:.4f} | {ceiling:>14.4f}")

    p0_f1 = agg(0.0, 'B_f1')[0] if 0.0 in p_grid else None
    if p0_f1 is not None:
        print(f"\n[Sanity 1] p=0 的 B_F1 = {p0_f1:.4f}（預期接近乾淨 GT 上限 ~0.9）")
    closest_p = min(p_grid, key=lambda x: abs(x - args.ocr_p))
    if abs(closest_p - args.ocr_p) < 1e-6:
        pm = agg(closest_p, 'B_f1')[0]
        print(f"[Sanity 2] p={args.ocr_p} 的曲線 B_F1 = {pm:.4f} vs 真實 OCR {args.ocr_f1:.4f}"
              f" → 垂直落差 {pm - args.ocr_f1:+.4f} = 錯字+位置偏移+雜訊框的總貢獻")

    # ── 6. 繪圖 ──
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            ps = sorted(set(r['p'] for r in results))
            b_mean = np.array([agg(p, 'B_f1')[0] for p in ps])
            b_std = np.array([agg(p, 'B_f1')[1] for p in ps])
            a_mean = np.array([agg(p, 'A_f1')[0] for p in ps])
            a_std = np.array([agg(p, 'A_f1')[1] for p in ps])

            p_fine = np.linspace(0, max(ps), 200)
            ceiling = np.array([np.mean((1 - p) ** name_lengths) for p in p_fine])

            plt.figure(figsize=(9, 6))
            plt.plot(p_fine, ceiling, ':', color='#7f8c8d', linewidth=1.5,
                     label='Theoretical ceiling: intact-name fraction')
            plt.errorbar(ps, b_mean, yerr=b_std, marker='o', color='#2980b9',
                         linewidth=2, capsize=3,
                         label='Curve B: Strict F1 vs full GT')
            plt.errorbar(ps, a_mean, yerr=a_std, marker='s', linestyle='--',
                         color='#27ae60', linewidth=1.5, capsize=3,
                         label='Curve A: F1 vs survivor GT (GNN-only degradation)')
            plt.scatter([args.ocr_p], [args.ocr_f1], marker='*', s=250,
                        color='#e74c3c', zorder=5,
                        label=f'Real OCR (p={args.ocr_p}, F1={args.ocr_f1})')
            plt.annotate('gap = typo + offset\n+ noise boxes',
                         xy=(args.ocr_p, args.ocr_f1),
                         xytext=(args.ocr_p + 0.03, args.ocr_f1 - 0.05),
                         fontsize=9, color='#e74c3c',
                         arrowprops=dict(arrowstyle='->', color='#e74c3c', alpha=0.7))
            plt.xlabel('Node deletion rate p (synthetic missing-char rate)')
            plt.ylabel('Entity-level F1')
            plt.title(f'Missing-rate degradation curve ({config.EXPERIMENT_NAME})')
            plt.legend(fontsize=9)
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1.0)
            plt.tight_layout()
            plot_path = os.path.join(out_dir, f'missing_rate_curve_{config.EXPERIMENT_NAME}.png')
            plt.savefig(plot_path, dpi=200)
            plt.close()
            print(f"📊 曲線圖：{plot_path}")
        except Exception as e:
            print(f"（繪圖失敗，略過）：{e}")

    total = time.time() - start_time
    print(f"\n總耗時：{int(total // 3600)}h {int(total % 3600 // 60)}m {total % 60:.0f}s")


if __name__ == "__main__":
    main()
