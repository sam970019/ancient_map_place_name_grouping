"""
noise_ablation.py — 噪聲消融階梯實驗（實驗 C）
========================================================================
目的：在真實 OCR 輸入上逐級「修復」各類噪聲，直接量測 37.3pp 落差
（真實 OCR 0.533 vs 完整 GT 0.906）中每種噪聲的貢獻，取代殘差推論。

階梯（累積式，全部為確定性轉換，各跑一次）：
  L0: 原始 OCR                          → 基準（應與 e2e_evaluate.py 測試集結果完全相同）
  L1: + 補齊全部缺字（GT 座標+文字）      → 偵測改善天花板
  L2: + 錯字換成 GT 文字（保留 OCR 座標） → 「LLM 補字＋改字」後校正模組天花板
  L3: + 刪除雜訊框（IoU<0.35 對不上 GT）  → 雜訊框過濾的價值
  L4: + 座標貼回 GT                      → 位置偏移的價值
  L5: 完整 GT 輸入（參照組）              → L4 與 L5 之差來自 OCR 的重複框與
                                           切分方式不同的框（L4 仍保留 OCR 的框數）

匹配規則：每個 OCR 框對全部 GT 框取最佳 IoU，≥ 0.35 視為匹配
（與 e2e / align_ocr_dataset / refill_rate_curve 一致）。
閾值與計分與 e2e_evaluate.py 相同：全部階級都用 checkpoint 中在驗證集選定的 E2E 閾值，
計分函式來自 e2e_common.py。
缺字分析與補字沿用 refill_rate_curve.analyze_missing / build_refilled_root。

執行方式：
  cd src
  python noise_ablation.py
  python noise_ablation.py --limit-maps 3          # 冒煙測試
  python noise_ablation.py --levels L0,L2,L5       # 只跑部分階級

注意：獨立實驗腳本，不修改任何既有檔案。
"""

import argparse
import copy
import csv
import os
import time
import warnings

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
from data import resolve_threshold
from e2e_common import (
    read_xml_with_encoding_fix, parse_gt_clusters_from_xml, evaluate_map_iou,
)
from missing_rate_curve import prf
from refill_rate_curve import box_iou, analyze_missing, build_refilled_root, IOU_THRESHOLD

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

ALL_LEVELS = ['L0', 'L1', 'L2', 'L3', 'L4', 'L5']

LEVEL_DESC = {
    'L0': '原始 OCR（基準）',
    'L1': '+ 補齊全部缺字',
    'L2': '+ 錯字→GT 文字',
    'L3': '+ 刪除雜訊框',
    'L4': '+ 座標貼回 GT',
    'L5': '完整 GT 輸入（參照）',
}

# 各階級啟用的轉換（累積式）
LEVEL_OPS = {
    'L0': dict(refill=False, fix_text=False, drop_noise=False, snap_coords=False),
    'L1': dict(refill=True,  fix_text=False, drop_noise=False, snap_coords=False),
    'L2': dict(refill=True,  fix_text=True,  drop_noise=False, snap_coords=False),
    'L3': dict(refill=True,  fix_text=True,  drop_noise=True,  snap_coords=False),
    'L4': dict(refill=True,  fix_text=True,  drop_noise=True,  snap_coords=True),
    # L5 走另一條路：直接以 GT XML 建圖
}


def parse_args():
    parser = argparse.ArgumentParser(description="噪聲消融階梯實驗（實驗 C）")
    parser.add_argument("--levels", type=str, default=",".join(ALL_LEVELS),
                        help="要跑的階級，逗號分隔（預設全部）")
    parser.add_argument("--limit-maps", type=int, default=0,
                        help="除錯用：只處理前 N 張地圖（0=全部）")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="輸出目錄（預設 config.OUTPUT_DIR）")
    parser.add_argument("--no-plot", action="store_true", help="不產生圖")
    parser.add_argument("--sort-chains", action="store_true",
                        help="建圖前把每個 id 群內的框依幾何主軸排序（驗證亂序假說；"
                             "輸出檔加 _sorted 後綴）")
    return parser.parse_args()


def sort_group_order_in_root(base_root):
    """
    深拷貝 root，把每個 id 群內的 <box> 依幾何主軸（PCA 第一主成分投影）重新排序。
    只改變文件順序（即 build_graph 的串鏈順序），不改任何框內容。
    方向歧義（正序/反序）無妨：BIDIRECTIONAL_GNN 下反序產生相同的邊集。
    """
    root = copy.deepcopy(base_root)
    from collections import defaultdict
    for img in root.findall('.//image'):
        boxes = img.findall('box')
        groups = defaultdict(list)
        for b in boxes:
            groups[b.get('id', '0')].append(b)
        for b in boxes:
            img.remove(b)
        for gid, bs in groups.items():
            if len(bs) > 1:
                try:
                    pts = np.array([
                        [float(b.get('left', 0)) + float(b.get('width', 0)) / 2,
                         float(b.get('top', 0)) + float(b.get('height', 0)) / 2]
                        for b in bs
                    ])
                    centered = pts - pts.mean(axis=0)
                    _, _, vt = np.linalg.svd(centered, full_matrices=False)
                    proj = centered @ vt[0]
                    bs = [bs[i] for i in np.argsort(proj, kind='stable')]
                except (ValueError, TypeError):
                    pass  # 有壞框就保持原序
            for b in bs:
                img.append(b)
    return root


def match_ocr_boxes_to_gt(ocr_root, gt_clusters):
    """
    對 ocr_root 內每個 <box> 元素（依 findall 順序）找最佳 IoU 的 GT 框。

    Returns:
        matches: list，與 box 元素一一對應。元素為
                 dict {'char','left','top','width','height','iou'}（匹配成功）
                 或 None（雜訊框 / 解析失敗）。
        stats: dict {'n_ocr','n_matched','n_noise','n_typo'}
    """
    gt_boxes = []
    for c in gt_clusters:
        for b in c['boxes']:
            gt_boxes.append(b)

    matches = []
    n_matched = n_noise = n_typo = n_bad = 0
    for img in ocr_root.findall('.//image'):
        for box in img.findall('box'):
            try:
                ol = float(box.get('left', 0))
                ot = float(box.get('top', 0))
                ow = float(box.get('width', 0))
                oh = float(box.get('height', 0))
            except (ValueError, TypeError):
                matches.append(None)
                n_bad += 1
                continue
            best_iou, best_gb = 0.0, None
            for gb in gt_boxes:
                iou = box_iou(gb['left'], gb['top'], gb['width'], gb['height'],
                              ol, ot, ow, oh)
                if iou > best_iou:
                    best_iou, best_gb = iou, gb
            if best_iou >= IOU_THRESHOLD and best_gb is not None:
                matches.append({
                    'char': best_gb['char'],
                    'left': best_gb['left'], 'top': best_gb['top'],
                    'width': best_gb['width'], 'height': best_gb['height'],
                    'iou': best_iou,
                })
                n_matched += 1
                label = box.find('label')
                ocr_text = (label.text or '').strip() if label is not None else ''
                if ocr_text != best_gb['char']:
                    n_typo += 1
            else:
                matches.append(None)
                n_noise += 1

    stats = {'n_ocr': len(matches), 'n_matched': n_matched,
             'n_noise': n_noise + n_bad, 'n_typo': n_typo}
    return matches, stats


def build_ablated_root(base_root, box_matches, refill_boxes,
                       refill=False, fix_text=False, drop_noise=False, snap_coords=False):
    """依階級設定建構修復後的 XML root（深拷貝，不動原始資料）。"""
    root = copy.deepcopy(base_root)
    idx = 0
    for img in root.findall('.//image'):
        for box in list(img.findall('box')):
            m = box_matches[idx]
            idx += 1
            if m is None:
                if drop_noise:
                    img.remove(box)
                continue
            if fix_text:
                label = box.find('label')
                if label is not None:
                    label.text = m['char']
            if snap_coords:
                box.set('left', str(int(round(m['left']))))
                box.set('top', str(int(round(m['top']))))
                box.set('width', str(int(round(m['width']))))
                box.set('height', str(int(round(m['height']))))
    if refill and refill_boxes:
        # build_refilled_root 會再深拷貝一次，這裡直接把補字框附加到現有 root
        image_elem = root.find('.//image')
        if image_elem is not None:
            import xml.etree.ElementTree as ET
            for rb in refill_boxes:
                b = ET.SubElement(image_elem, 'box')
                b.set('top', str(int(round(rb['top']))))
                b.set('left', str(int(round(rb['left']))))
                b.set('width', str(int(round(rb['width']))))
                b.set('height', str(int(round(rb['height']))))
                b.set('id', rb['refill_id'])
                b.set('x', 'None')
                b.set('y', 'None')
                b.set('score', '1.0')
                label = ET.SubElement(b, 'label')
                label.set('score', '1.0')
                label.text = rb['char']
                ET.SubElement(b, 'correction').text = ''
    return root


def main():
    args = parse_args()
    levels = [x.strip() for x in args.levels.split(',') if x.strip()]
    for lv in levels:
        if lv not in ALL_LEVELS:
            raise ValueError(f"未知階級 {lv}；可用：{ALL_LEVELS}")
    out_dir = args.out_dir or config.OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    start_time = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── 1. 載入模型（鏡射 refill_rate_curve / e2e_evaluate）──
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
    threshold, threshold_src = resolve_threshold(loaded, 'e2e')
    print(f"預測閾值：{threshold}（來源：{threshold_src}）")

    # ── 2. 載入地圖 + 預先計算缺字集合與 OCR→GT 匹配 ──
    ocr_xml_files = sorted([f for f in os.listdir(config.OCR_XML_DIR) if f.endswith('.xml')])
    if args.limit_maps > 0:
        ocr_xml_files = ocr_xml_files[:args.limit_maps]

    map_cache = []
    agg_stats = {'n_ocr': 0, 'n_matched': 0, 'n_noise': 0, 'n_typo': 0,
                 'n_missing': 0, 'n_gt_boxes': 0}

    for fname in tqdm(ocr_xml_files, desc="載入地圖並分析噪聲"):
        gt_path = os.path.join(config.GT_XML_DIR, fname)
        ocr_path = os.path.join(config.OCR_XML_DIR, fname)
        if not os.path.exists(gt_path):
            print(f"⚠️  跳過 {fname}：GT 檔不存在")
            continue
        try:
            ocr_root = read_xml_with_encoding_fix(ocr_path)
            gt_root = read_xml_with_encoding_fix(gt_path)
        except Exception as e:
            print(f"⚠️  跳過 {fname}：解析失敗 ({e})")
            continue

        gt_clusters = parse_gt_clusters_from_xml(gt_path)

        # 缺字分析（沿用實驗 B）
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

        # OCR→GT 匹配（供 L2/L3/L4）
        box_matches, stats = match_ocr_boxes_to_gt(ocr_root, gt_clusters)

        for k in ['n_ocr', 'n_matched', 'n_noise', 'n_typo']:
            agg_stats[k] += stats[k]
        agg_stats['n_missing'] += len(missing_list)
        agg_stats['n_gt_boxes'] += n_gt_boxes

        map_cache.append({
            'file': fname, 'ocr_root': ocr_root, 'gt_root': gt_root,
            'gt_clusters': gt_clusters, 'missing_list': missing_list,
            'box_matches': box_matches,
        })

    if not map_cache:
        raise RuntimeError("沒有任何地圖成功載入。")

    print(f"\n共載入 {len(map_cache)} 張地圖")
    print(f"  GT 框總數        ：{agg_stats['n_gt_boxes']}")
    print(f"  OCR 框總數       ：{agg_stats['n_ocr']}")
    print(f"  匹配成功         ：{agg_stats['n_matched']}")
    print(f"  雜訊框（無 GT 對應）：{agg_stats['n_noise']} "
          f"（佔 OCR 框 {agg_stats['n_noise'] / max(agg_stats['n_ocr'], 1):.2%}）")
    print(f"  錯字（匹配但文字不同）：{agg_stats['n_typo']} "
          f"（佔匹配框 {agg_stats['n_typo'] / max(agg_stats['n_matched'], 1):.2%}）")
    print(f"  缺字（GT 無 OCR 對應）：{agg_stats['n_missing']} "
          f"（缺字率 {agg_stats['n_missing'] / max(agg_stats['n_gt_boxes'], 1):.4f}）")

    # ── 3. 單一階級評估 ──
    def run_level(level):
        tot = dict(tp_s=0, tp_b=0, fn_a=0, fn_c=0, fp=0, n_pred=0, n_gt=0)

        for m in tqdm(map_cache, desc=f"  {level}", leave=False):
            if level == 'L5':
                root = m['gt_root']
            else:
                ops = LEVEL_OPS[level]
                if not any(ops.values()):
                    root = m['ocr_root']
                else:
                    root = build_ablated_root(
                        m['ocr_root'], m['box_matches'],
                        m['missing_list'] if ops['refill'] else [],
                        **ops,
                    )
            if args.sort_chains:
                root = sort_group_order_in_root(root)
            try:
                data, text_idx, _, place_name_edge, ocr_results = build_graph(root, model)
            except Exception as e:
                tqdm.write(f"⚠️  {m['file']} 建圖失敗（{level}）：{e}")
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
        return {
            'level': level, 'desc': LEVEL_DESC[level],
            'B_precision': b_p, 'B_recall': b_r, 'B_f1': b_f1,
            'B_relaxed_f1': rel_f1,
            'n_type_a': tot['fn_a'], 'n_type_c': tot['fn_c'],
            'n_fp': tot['fp'], 'n_tp_strict': tot['tp_s'],
            'n_tp_type_b': tot['tp_b'], 'n_gt': tot['n_gt'],
        }

    # ── 4. 跑階梯 ──
    suffix = '_sorted' if args.sort_chains else ''
    csv_path = os.path.join(out_dir, f'noise_ablation_{config.EXPERIMENT_NAME}{suffix}.csv')
    fieldnames = ['level', 'desc', 'B_precision', 'B_recall', 'B_f1', 'B_relaxed_f1',
                  'n_type_a', 'n_type_c', 'n_fp', 'n_tp_strict', 'n_tp_type_b', 'n_gt']
    results = []

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for lv in levels:
            res = run_level(lv)
            row = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in res.items()}
            results.append(row)
            writer.writerow(row)
            f.flush()
            print(f"  {lv} {LEVEL_DESC[lv]:<18} Strict F1={res['B_f1']:.4f}  "
                  f"Relaxed F1={res['B_relaxed_f1']:.4f}  "
                  f"(TypeA={res['n_type_a']}, TypeC={res['n_type_c']}, FP={res['n_fp']})")

    print(f"\n📄 結果 CSV：{csv_path}")

    # ── 5. 彙總：邊際貢獻 ──
    print("\n==================== 噪聲消融階梯 ====================")
    print(f"{'階級':<4} {'內容':<22} {'Strict F1':>10} {'Δ(邊際)':>9} {'Relaxed F1':>11}")
    prev_f1 = None
    for r in results:
        delta = '' if prev_f1 is None else f"{r['B_f1'] - prev_f1:+.4f}"
        print(f"{r['level']:<4} {r['desc']:<22} {r['B_f1']:>10.4f} {delta:>9} "
              f"{r['B_relaxed_f1']:>11.4f}")
        prev_f1 = r['B_f1']

    by_level = {r['level']: r for r in results}
    if 'L0' in by_level:
        print(f"\n[Sanity] L0 Strict F1 = {by_level['L0']['B_f1']:.4f}"
              f"（應與 e2e_evaluate.py 測試集結果完全相同）")
    if 'L4' in by_level and 'L5' in by_level:
        gap = by_level['L5']['B_f1'] - by_level['L4']['B_f1']
        print(f"L5 − L4 = {gap:+.4f}（OCR 重複框與切分方式不同的框）")

    # ── 6. 階梯圖 ──
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib import font_manager

            # 嘗試使用中文字型
            for font in ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei']:
                if any(font.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
                    plt.rcParams['font.family'] = font
                    break
            plt.rcParams['axes.unicode_minus'] = False

            xs = [r['level'] for r in results]
            strict = [r['B_f1'] for r in results]
            relaxed = [r['B_relaxed_f1'] for r in results]

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(xs, strict, color='#2980b9', width=0.55, label='Strict F1')
            ax.plot(xs, relaxed, marker='o', color='#e67e22', linewidth=2,
                    label='Relaxed F1')
            for bar, r in zip(bars, results):
                ax.annotate(f"{r['B_f1']:.3f}", (bar.get_x() + bar.get_width() / 2, r['B_f1']),
                            textcoords='offset points', xytext=(0, 4),
                            ha='center', fontsize=9)
            if 'L5' in by_level:
                ax.axhline(by_level['L5']['B_f1'], color='#27ae60', linestyle='--', alpha=0.6,
                           label=f"GT input {by_level['L5']['B_f1']:.3f}")
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels([f"{r['level']}\n{r['desc']}" for r in results], fontsize=8)
            ax.set_ylabel('Entity-level F1 (vs full GT)')
            ax.set_title(f'Noise ablation ladder ({config.EXPERIMENT_NAME})')
            ax.set_ylim(0, 1.0)
            ax.grid(True, axis='y', alpha=0.3)
            ax.legend(fontsize=9)
            plt.tight_layout()
            plot_path = os.path.join(out_dir, f'noise_ablation_{config.EXPERIMENT_NAME}{suffix}.png')
            plt.savefig(plot_path, dpi=200)
            plt.close()
            print(f"📊 階梯圖：{plot_path}")
        except Exception as e:
            print(f"（繪圖失敗，略過）：{e}")

    total = time.time() - start_time
    print(f"\n總耗時：{int(total // 3600)}h {int(total % 3600 // 60)}m {total % 60:.0f}s")


if __name__ == "__main__":
    main()
