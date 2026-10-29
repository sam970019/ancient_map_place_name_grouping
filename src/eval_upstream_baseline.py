"""
eval_upstream_baseline.py — 前人分群方法的同協定 E2E 評估
==========================================================
OCR XML 中每個框的 id 就是上游（前人方法）分群的輸出：以 Delaunay 三角剖分建立候選
連線，再依規則剔除多餘的邊，剩下的連通分量即為地名。本腳本把同一 id 的框視為一個
預測地名，交給與 GNN 完全相同的 evaluate_map_iou 計分，不需要模型或 GPU。

與 GNN 評估對齊的規則：
  - 框的讀取順序、座標取整、文字 strip 與 data.build_graph 相同
  - id='0' 的框不屬於任何預測地名（前人方法的輸出中約每張圖 2 個，多為大字）
  - 單字群集不算預測（GNN 端 metrics.check_pred_clusters 也只保留 >1 個節點的群集）
  - 預測或 GT 為空的地圖不計入總數（與 e2e_evaluate.e2e_evaluate 的彙總規則相同）

執行方式：
  cd src
  python eval_upstream_baseline.py                       # 39 張 E2E 測試圖
  python eval_upstream_baseline.py --maps-tsv ../../llm_detect/handoff_astra_testset/testset_maps.tsv
  python eval_upstream_baseline.py --keep-singletons     # 參考用：保留單字群集
"""

import argparse
import csv
import json
import os

import config
from e2e_common import read_xml_with_encoding_fix, parse_gt_clusters_from_xml, evaluate_map_iou


def read_upstream_boxes(ocr_path):
    root = read_xml_with_encoding_fix(ocr_path)
    boxes = []
    for image in root.findall(".//image"):
        for box in image.findall("box"):
            boxes.append({
                "text": box.find("label").text.strip(),
                "x": int(box.get("left")),
                "y": int(box.get("top")),
                "w": int(box.get("width")),
                "h": int(box.get("height")),
                "id": box.get("id"),
            })
    return boxes


def upstream_clusters(boxes, keep_singletons=False):
    groups = {}
    for idx, b in enumerate(boxes):
        if b["id"] == "0":
            continue
        groups.setdefault(b["id"], []).append(idx)
    min_size = 1 if keep_singletons else 2
    return [{"node_indices": nodes} for nodes in groups.values() if len(nodes) >= min_size]


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="前人分群方法（OCR XML 上游 id）的同協定 E2E 評估")
    parser.add_argument("--ocr-dir", default=config.OCR_XML_DIR)
    parser.add_argument("--gt-dir", default=config.GT_XML_DIR)
    parser.add_argument("--maps-tsv", default=None,
                        help="只評估此 TSV 的 ocr_id 欄所列地圖（例如 VLM 實驗的 33 張）")
    parser.add_argument("--keep-singletons", action="store_true",
                        help="保留單字群集（預設丟棄，與 GNN 端規則一致）")
    parser.add_argument("--out", default=None, help="把結果寫成 JSON")
    args = parser.parse_args()

    files = sorted(f for f in os.listdir(args.ocr_dir) if f.endswith(".xml"))
    if args.maps_tsv:
        with open(args.maps_tsv, encoding="utf-8") as fh:
            wanted = {row["ocr_id"] + ".xml" for row in csv.DictReader(fh, delimiter="\t")}
        files = [f for f in files if f in wanted]

    tot = dict(tp_s=0, tp_b=0, fn_a=0, fn_c=0, fp=0, gt=0, pred=0)
    per_map = []
    for f in files:
        gt_path = os.path.join(args.gt_dir, f)
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"{f}：找不到 GT {gt_path}")
        boxes = read_upstream_boxes(os.path.join(args.ocr_dir, f))
        pred = upstream_clusters(boxes, args.keep_singletons)
        gt = parse_gt_clusters_from_xml(gt_path)
        # 每張圖都計入（預測或 GT 為空也一樣），與 e2e_evaluate 相同
        tp_s, tp_b, fn_a, fn_c, fp, _, _ = evaluate_map_iou(pred, gt, boxes)
        for k, v in zip(("tp_s", "tp_b", "fn_a", "fn_c", "fp"), (tp_s, tp_b, fn_a, fn_c, fp)):
            tot[k] += v
        tot["gt"] += len(gt)
        tot["pred"] += len(pred)
        s = f1(tp_s / len(pred) if pred else 0.0, tp_s / len(gt) if gt else 0.0)
        r = f1((tp_s + tp_b) / len(pred) if pred else 0.0, (tp_s + tp_b) / len(gt) if gt else 0.0)
        per_map.append({"map": f, "strict_f1": s, "relaxed_f1": r, "pred": len(pred), "gt": len(gt)})
        print(f"  {f} | Strict F1={s:.4f} (Relaxed F1={r:.4f})  (pred={len(pred)}, gt={len(gt)})")

    sp, sr = tot["tp_s"] / tot["pred"], tot["tp_s"] / tot["gt"]
    rp, rr = (tot["tp_s"] + tot["tp_b"]) / tot["pred"], (tot["tp_s"] + tot["tp_b"]) / tot["gt"]
    result = {
        "maps": len(per_map),
        "keep_singletons": args.keep_singletons,
        "strict": {"precision": sp, "recall": sr, "f1": f1(sp, sr)},
        "relaxed": {"precision": rp, "recall": rr, "f1": f1(rp, rr)},
        "counts": {"TP_strict": tot["tp_s"], "TypeB": tot["tp_b"], "TypeA": tot["fn_a"],
                   "TypeC": tot["fn_c"], "FP": tot["fp"], "gt": tot["gt"], "pred": tot["pred"]},
        "per_map": per_map,
    }

    print(f"\n=== 前人分群方法（上游 id）E2E，{len(per_map)} 張圖"
          f"{'，保留單字群集' if args.keep_singletons else ''} ===")
    print(f"Strict  : P={sp:.4f}  R={sr:.4f}  F1={f1(sp, sr):.4f}")
    print(f"Relaxed : P={rp:.4f}  R={rr:.4f}  F1={f1(rp, rr):.4f}")
    g = tot["gt"]
    print(f"Type A 缺字 {tot['fn_a']} ({tot['fn_a']/g:.1%}) | Type B 錯字 {tot['tp_b']} ({tot['tp_b']/g:.1%}) | "
          f"Type C 分群錯 {tot['fn_c']} ({tot['fn_c']/g:.1%}) | FP {tot['fp']} | GT {g} | 預測 {tot['pred']}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=1)
        print(f"已寫出 {args.out}")


if __name__ == "__main__":
    main()
