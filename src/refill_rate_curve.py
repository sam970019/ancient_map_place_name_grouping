"""
refill_rate_curve.py — 補字率–表現恢復曲線實驗（實驗 B）
========================================================================
目的：從真實 OCR 輸入（含真實缺字、錯字、位置偏移、雜訊框）出發，
把 OCR 漏掉的字以比例 q 逐步「補回」，模擬 OCR 偵測率提升後的表現恢復曲線。

  - q=0   ：現狀，必須重現 e2e 的 53.31%（sanity check）
  - q=1.0 ：缺字全補 → 「零缺字、但錯字/偏移/雜訊框仍在」的端點
  - 缺字模式天然帶真實傾向（補的就是實際漏掉的那些字）

設計決策（已由使用者核准）：
  B1: 補回框的分組 id = 同地名已偵測兄弟框的 OCR 群 id（多數決，平手取
      幾何最近兄弟）；整個地名全缺則給全新獨立 id。
  B2: 補回內容 = GT 座標 + GT 正確文字（「完美偵測+完美辨識」的樂觀上限）。
  B3: q ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0}；per-box Bernoulli(q) 抽樣；
      中間點 5 seeds，q=0 與 q=1 各 1 次。

與實驗 A 的合體：x 軸換算為有效缺字率 p_eff = 實測殘餘缺字數 / GT 總框數，
可與 missing_rate_curve 的曲線疊在同一張圖（若找到其 CSV 會自動疊圖）。

執行方式：
  cd src
  python refill_rate_curve.py
  python refill_rate_curve.py --q-grid 0,1.0 --seeds 1 --limit-maps 3   # 冒煙測試

注意：獨立實驗腳本，不修改任何既有檔案。
      共用函式自 missing_rate_curve.py 引入（該檔有 __main__ 保護，可安全 import）。
"""

import argparse
import copy
import csv
import os
import time
import warnings
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm

import config
import data as data_module
from data import (
    fit_global_scaler, build_graph, build_data,
    build_global_vocab, load_fasttext_embeddings, DATA_DIR,
)
from metrics import check_pred_clusters
from missing_rate_curve import (
    read_xml_with_encoding_fix, parse_gt_clusters_from_xml, evaluate_map_iou, prf,
)

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

IOU_THRESHOLD = 0.35  # 與 e2e / align_ocr_dataset 一致


def parse_args():
    parser = argparse.ArgumentParser(description="補字率–表現恢復曲線實驗（實驗 B）")
    parser.add_argument("--q-grid", type=str, default="0,0.2,0.4,0.6,0.8,1.0",
                        help="補字率網格，逗號分隔")
    parser.add_argument("--seeds", type=int, default=5,
                        help="每個中間 q 的隨機種子數（q=0/1 無需多 seed，各跑 1 次）")
    parser.add_argument("--limit-maps", type=int, default=0,
                        help="除錯用：只處理前 N 張地圖（0=全部）")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="輸出目錄（預設 config.OUTPUT_DIR）")
    parser.add_argument("--no-plot", action="store_true", help="不產生曲線圖")
    return parser.parse_args()


def box_iou(gl, gt_top, gw, gh, ol, ot, ow, oh):
    inter_l = max(gl, ol)
    inter_t = max(gt_top, ot)
    inter_r = min(gl + gw, ol + ow)
    inter_b = min(gt_top + gh, ot + oh)
    inter_area = max(0.0, inter_r - inter_l) * max(0.0, inter_b - inter_t)
    if inter_area <= 0:
        return 0.0
    union = gw * gh + ow * oh - inter_area
    return inter_area / union if union > 0 else 0.0


def analyze_missing(gt_clusters, ocr_boxes_meta, fresh_id_start):
    """
    對一張地圖：找出 OCR 漏掉的 GT 框，並依決策 B1 決定每個缺字框的補回 id。

    Args:
        gt_clusters: parse_gt_clusters_from_xml 的輸出
        ocr_boxes_meta: list of dict {'left','top','width','height','id'}（原始 OCR 框）
        fresh_id_start: 全缺地名新 id 的起始值

    Returns:
        missing_list: list of dict {'char','left','top','width','height','refill_id'}
        n_gt_boxes: GT 總框數
    """
    missing_list = []
    n_gt_boxes = 0
    fresh_counter = fresh_id_start

    for gt_c in gt_clusters:
        # 每個 GT 框找最佳 IoU 的 OCR 框
        matches = []  # (gt_box, matched_ocr_id or None)
        for gt_box in gt_c['boxes']:
            n_gt_boxes += 1
            best_iou, best_idx = 0.0, -1
            for o_idx, ob in enumerate(ocr_boxes_meta):
                iou = box_iou(gt_box['left'], gt_box['top'], gt_box['width'], gt_box['height'],
                              ob['left'], ob['top'], ob['width'], ob['height'])
                if iou > best_iou:
                    best_iou, best_idx = iou, o_idx
            matched_id = ocr_boxes_meta[best_idx]['id'] if best_iou >= IOU_THRESHOLD else None
            matches.append((gt_box, matched_id))

        sibling_ids = [mid for _, mid in matches if mid is not None]
        missing_boxes = [(gb, ) for gb, mid in matches if mid is None]
        if not missing_boxes:
            continue

        if sibling_ids:
            # 多數決；平手時對每個缺字框取幾何最近的已匹配兄弟之 id
            counts = Counter(sibling_ids)
            top = counts.most_common()
            tied = [i for i, c in top if c == top[0][1]]
            matched_sibs = [(gb, mid) for gb, mid in matches if mid is not None]
            for (gt_box,) in missing_boxes:
                if len(tied) == 1:
                    refill_id = tied[0]
                else:
                    cx = gt_box['left'] + gt_box['width'] / 2
                    cy = gt_box['top'] + gt_box['height'] / 2
                    best_d, refill_id = float('inf'), tied[0]
                    for sib_box, sib_id in matched_sibs:
                        if sib_id not in tied:
                            continue
                        sx = sib_box['left'] + sib_box['width'] / 2
                        sy = sib_box['top'] + sib_box['height'] / 2
                        d = (cx - sx) ** 2 + (cy - sy) ** 2
                        if d < best_d:
                            best_d, refill_id = d, sib_id
                missing_list.append({
                    'char': gt_box['char'],
                    'left': gt_box['left'], 'top': gt_box['top'],
                    'width': gt_box['width'], 'height': gt_box['height'],
                    'refill_id': str(refill_id),
                })
        else:
            # 整個地名全缺 → 同一地名的缺字共用一個全新 id
            new_id = str(fresh_counter)
            fresh_counter += 1
            for (gt_box,) in missing_boxes:
                missing_list.append({
                    'char': gt_box['char'],
                    'left': gt_box['left'], 'top': gt_box['top'],
                    'width': gt_box['width'], 'height': gt_box['height'],
                    'refill_id': new_id,
                })

    return missing_list, n_gt_boxes


def build_refilled_root(base_root, refill_boxes):
    """深拷貝 OCR root，把補字框以 <box> 元素附加到第一個 <image> 之下。"""
    root = copy.deepcopy(base_root)
    image_elem = root.find('.//image')
    if image_elem is None:
        return root
    for rb in refill_boxes:
        box = ET.SubElement(image_elem, 'box')
        box.set('top', str(int(round(rb['top']))))
        box.set('left', str(int(round(rb['left']))))
        box.set('width', str(int(round(rb['width']))))
        box.set('height', str(int(round(rb['height']))))
        box.set('id', rb['refill_id'])
        box.set('x', 'None')
        box.set('y', 'None')
        box.set('score', '1.0')
        label = ET.SubElement(box, 'label')
        label.set('score', '1.0')
        label.text = rb['char']
        ET.SubElement(box, 'correction').text = ''
    return root


def main():
    args = parse_args()
    q_grid = [float(x) for x in args.q_grid.split(',') if x.strip() != '']
    out_dir = args.out_dir or config.OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    start_time = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── 1. 載入模型（鏡射 e2e_evaluate.py）──
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
    all_xml_files = sorted([f for f in os.listdir(config.XML_DIR)
                            if f.endswith('.xml') and not f.startswith('B-')])
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

    # ── 2. 載入 OCR 地圖 + GT，預先分析缺字集合與 refill id ──
    ocr_xml_files = sorted([f for f in os.listdir(config.OCR_XML_DIR) if f.endswith('.xml')])
    if args.limit_maps > 0:
        ocr_xml_files = ocr_xml_files[:args.limit_maps]

    map_cache = []
    total_missing_all = 0
    total_gt_boxes_all = 0
    n_oov = 0

    for fname in tqdm(ocr_xml_files, desc="載入地圖並分析缺字"):
        gt_path = os.path.join(config.GT_XML_DIR, fname)
        ocr_path = os.path.join(config.OCR_XML_DIR, fname)
        if not os.path.exists(gt_path):
            print(f"⚠️  跳過 {fname}：GT 檔不存在")
            continue
        try:
            ocr_root = read_xml_with_encoding_fix(ocr_path)
        except Exception as e:
            print(f"⚠️  跳過 {fname}：OCR 解析失敗 ({e})")
            continue

        gt_clusters = parse_gt_clusters_from_xml(gt_path)

        # 原始 OCR 框（含 id）供缺字分析
        ocr_boxes_meta = []
        max_numeric_id = 0
        for img in ocr_root.findall('.//image'):
            for box in img.findall('box'):
                try:
                    ocr_boxes_meta.append({
                        'left': float(box.get('left', 0)), 'top': float(box.get('top', 0)),
                        'width': float(box.get('width', 0)), 'height': float(box.get('height', 0)),
                        'id': box.get('id', '0'),
                    })
                    bid = box.get('id', '0')
                    if bid.isdigit():
                        max_numeric_id = max(max_numeric_id, int(bid))
                except (ValueError, TypeError):
                    continue

        missing_list, n_gt_boxes = analyze_missing(
            gt_clusters, ocr_boxes_meta, fresh_id_start=max_numeric_id + 1000
        )
        total_missing_all += len(missing_list)
        total_gt_boxes_all += n_gt_boxes
        n_oov += sum(1 for mb in missing_list if mb['char'] not in data_module.vocab)

        map_cache.append({
            'file': fname, 'ocr_root': ocr_root,
            'gt_clusters': gt_clusters, 'missing_list': missing_list,
            'n_gt_boxes': n_gt_boxes,
        })

    if not map_cache:
        raise RuntimeError("沒有任何地圖成功載入。")
    real_missing_rate = total_missing_all / total_gt_boxes_all if total_gt_boxes_all else 0.0
    print(f"共載入 {len(map_cache)} 張地圖 | GT 框 {total_gt_boxes_all} | "
          f"缺字框 {total_missing_all}（實測缺字率 {real_missing_rate:.4f}）| "
          f"缺字 OOV {n_oov} 個")

    # ── 3. 單次評估 ──
    def run_once(q, seed):
        np.random.seed(seed)
        tot = dict(tp_s=0, tp_b=0, fn_a=0, fn_c=0, fp=0, n_pred=0, n_gt=0)
        n_refilled_total = 0

        for m in map_cache:
            missing = m['missing_list']
            if q >= 1.0:
                refill = list(missing)
            elif q <= 0.0:
                refill = []
            else:
                mask = np.random.rand(len(missing)) < q
                refill = [mb for mb, keep in zip(missing, mask) if keep]
            n_refilled_total += len(refill)

            root = build_refilled_root(m['ocr_root'], refill) if refill else m['ocr_root']
            try:
                data, text_idx, _, place_name_edge, ocr_results = build_graph(root, model)
            except Exception as e:
                tqdm.write(f"⚠️  {m['file']} 建圖失敗（q={q}, seed={seed}）：{e}")
                tot['n_gt'] += len(m['gt_clusters'])
                continue
            test_data = build_data(data, place_name_edge, ocr_results, is_test=True)

            new_text_idx = {
                i: (it['id'], it['text'], it['x'], it['y'])
                for i, it in enumerate(ocr_results)
            }
            pred_clusters, _preds = check_pred_clusters(
                model, test_data, device, new_text_idx, threshold=threshold
            )

            gt_clusters = [
                {'chars': c['chars'], 'centroid': c['centroid'],
                 'boxes': [dict(b) for b in c['boxes']]}
                for c in m['gt_clusters']
            ]
            tp_s, tp_b, fn_a, fn_c, fp, _, _ = evaluate_map_iou(
                pred_clusters, gt_clusters, ocr_results
            )
            tot['tp_s'] += tp_s
            tot['tp_b'] += tp_b
            tot['fn_a'] += fn_a
            tot['fn_c'] += fn_c
            tot['fp'] += fp
            tot['n_pred'] += len(pred_clusters)
            tot['n_gt'] += len(gt_clusters)

        b_p, b_r, b_f1 = prf(tot['tp_s'], tot['n_pred'], tot['n_gt'])
        rel_p, rel_r, rel_f1 = prf(tot['tp_s'] + tot['tp_b'], tot['n_pred'], tot['n_gt'])
        p_eff = (total_missing_all - n_refilled_total) / total_gt_boxes_all
        return {
            'B_precision': b_p, 'B_recall': b_r, 'B_f1': b_f1,
            'B_relaxed_f1': rel_f1,
            'p_eff': p_eff, 'n_refilled': n_refilled_total,
            'n_type_a': tot['fn_a'], 'n_type_c': tot['fn_c'],
            'n_fp': tot['fp'], 'n_tp_strict': tot['tp_s'], 'n_gt': tot['n_gt'],
        }

    # ── 4. 掃描 (q, seed) ──
    csv_path = os.path.join(out_dir, f'refill_rate_curve_{config.EXPERIMENT_NAME}.csv')
    fieldnames = ['q', 'seed', 'p_eff', 'n_refilled',
                  'B_precision', 'B_recall', 'B_f1', 'B_relaxed_f1',
                  'n_type_a', 'n_type_c', 'n_fp', 'n_tp_strict', 'n_gt']
    results = []

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        runs = []
        for q in q_grid:
            seeds = [0] if q <= 0.0 or q >= 1.0 else list(range(args.seeds))
            for s in seeds:
                runs.append((q, s))

        pbar = tqdm(runs, desc="掃描 (q, seed)")
        for q, s in pbar:
            pbar.set_postfix({'q': f'{q:.2f}', 'seed': s})
            res = run_once(q, s)
            row = {'q': q, 'seed': s, **{k: round(v, 6) if isinstance(v, float) else v
                                          for k, v in res.items()}}
            results.append(row)
            writer.writerow(row)
            f.flush()
            tqdm.write(
                f"  q={q:<5.2f} seed={s}  p_eff={res['p_eff']:.4f}  B_F1={res['B_f1']:.4f}  "
                f"(refilled={res['n_refilled']}, TypeA={res['n_type_a']}, TypeC={res['n_type_c']})"
            )

    print(f"\n📄 結果 CSV：{csv_path}")

    # ── 5. 彙總與 sanity checks ──
    def agg(q, key):
        vals = [r[key] for r in results if r['q'] == q]
        return np.mean(vals), np.std(vals)

    print("\n==================== 彙總 ====================")
    print(f"{'q':>6} | {'p_eff':>8} | {'B_F1 (Strict)':>16} | {'Relaxed F1':>12}")
    for q in q_grid:
        pm = agg(q, 'p_eff')[0]
        bm, bs = agg(q, 'B_f1')
        rm = agg(q, 'B_relaxed_f1')[0]
        print(f"{q:>6.2f} | {pm:>8.4f} | {bm:>9.4f} ± {bs:.4f} | {rm:>12.4f}")

    if 0.0 in q_grid:
        q0 = agg(0.0, 'B_f1')[0]
        print(f"\n[Sanity 1] q=0 的 Strict F1 = {q0:.4f}（應 ≈ 0.5331；|差| > 0.005 需檢查）")
    if 1.0 in q_grid:
        q1 = agg(1.0, 'B_f1')[0]
        q1_peff = agg(1.0, 'p_eff')[0]
        print(f"[Sanity 2] q=1 的殘餘缺字率 = {q1_peff:.4f}（應為 0）")
        print(f"[Sanity 3] q=1 的 Strict F1 = {q1:.4f}"
              f"（「零缺字+真實其他噪聲」端點；預期落在 0.734 與 0.906 之間）")

    # ── 6. 疊圖（自動載入實驗 A 的 CSV）──
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            qs = sorted(set(r['q'] for r in results))
            x_b = [agg(q, 'p_eff')[0] for q in qs]
            y_b = [agg(q, 'B_f1')[0] for q in qs]
            e_b = [agg(q, 'B_f1')[1] for q in qs]

            plt.figure(figsize=(9, 6))

            # 實驗 A 曲線（若存在）
            a_csv = os.path.join(out_dir, f'missing_rate_curve_{config.EXPERIMENT_NAME}.csv')
            if os.path.exists(a_csv):
                import collections
                a_data = collections.defaultdict(list)
                with open(a_csv, encoding='utf-8-sig') as af:
                    for row in csv.DictReader(af):
                        a_data[float(row['p'])].append(float(row['B_f1']))
                a_ps = sorted(a_data.keys())
                a_mean = [np.mean(a_data[p]) for p in a_ps]
                a_std = [np.std(a_data[p]) for p in a_ps]
                plt.errorbar(a_ps, a_mean, yerr=a_std, marker='o', color='#2980b9',
                             linewidth=2, capsize=3,
                             label='Exp A: synthetic uniform deletion on clean GT')

            plt.errorbar(x_b, y_b, yerr=e_b, marker='s', color='#e67e22',
                         linewidth=2, capsize=3,
                         label='Exp B: refill on real OCR (real noise retained)')
            for q, x, y in zip(qs, x_b, y_b):
                plt.annotate(f'q={q:.1f}', (x, y), textcoords='offset points',
                             xytext=(6, -12), fontsize=8, color='#e67e22')

            plt.scatter([real_missing_rate], [agg(0.0, 'B_f1')[0] if 0.0 in q_grid else np.nan],
                        marker='*', s=250, color='#e74c3c', zorder=5,
                        label=f'Real OCR baseline (p={real_missing_rate:.4f})')
            plt.xlabel('Effective missing-char rate p')
            plt.ylabel('Entity-level Strict F1 (vs full GT)')
            plt.title(f'Missing-rate curves: synthetic vs real-noise refill ({config.EXPERIMENT_NAME})')
            plt.legend(fontsize=9)
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1.0)
            plt.tight_layout()
            plot_path = os.path.join(out_dir, f'refill_rate_curve_{config.EXPERIMENT_NAME}.png')
            plt.savefig(plot_path, dpi=200)
            plt.close()
            print(f"📊 曲線圖：{plot_path}")
        except Exception as e:
            print(f"（繪圖失敗，略過）：{e}")

    total = time.time() - start_time
    print(f"\n總耗時：{int(total // 3600)}h {int(total % 3600 // 60)}m {total % 60:.0f}s")


if __name__ == "__main__":
    main()
