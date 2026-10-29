"""
把 llm_detect.py 的 astra 偵測結果寫成 e2e_evaluate.py 吃得下的 OCR XML，產生各實驗條件的資料夾。

條件（每個資料夾只放「有原圖」的 33 張，所有條件評估同一批圖）：
    base33        原始 OCR XML，僅篩出 33 張（新基準）
    ctrl_uniqid   原始 OCR，但每個框的 id 換成唯一值（驗證 id 不影響 e2e；應與 base33 完全相同）
    A_raw         方案 A：全部改用 astra 的框與字（含 '?'）
    A_noq         方案 A 變體：astra 框，但排除 '?'
    C_t{XX}       方案 C：OCR 框全保留，只補「與所有 OCR 框 IoU < 0.2」且 conf ≥ 0.XX 的 astra 框（保留 '?'）

用法：
    python llm_build_xml.py                       # 產生全部條件
    python llm_build_xml.py --c-thresholds 0.9 0.8 0.7
    python llm_build_xml.py --cfg t768_s1_o192_high_g1_g31pro --out-root <DATA_DIR>/llm_detect/e2e_gemini
之後：
    OCR_XML_DIR=<DATA_DIR>/llm_detect/e2e/xml_C_t090 python e2e_evaluate.py
"""
import os
import sys
import json
import shutil
import argparse
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape, quoteattr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import DATA_DIR, read_xml_from
from llm_detect import box_iou

OCR_DIR = os.path.join(DATA_DIR, "xml_ocr_output")
ORIG_DIR = os.path.join(DATA_DIR, "map", "original")
OUT_ROOT = os.path.join(DATA_DIR, "llm_detect", "e2e")
DEFAULT_CFG = "t768_s1_o192_medium_p6_astra"


def ocr_to_gt(ocr_id):
    return str(10 - int(ocr_id[0])) + ocr_id[1:]


def ocr_boxes(root):
    out = []
    for b in root.iter("box"):
        lab = b.find("label")
        out.append(dict(text=(lab.text or "").strip(), left=int(float(b.get("left"))), top=int(float(b.get("top"))),
                        width=int(float(b.get("width"))), height=int(float(b.get("height"))),
                        id=b.get("id"), score=b.get("score") or "1.0"))
    return out


def write_xml(path, image_file, boxes):
    """imglab 格式；座標寫整數（build_graph 用 int() 讀）。"""
    lines = ["<?xml version='1.0' encoding='UTF-8'?>", "<dataset>", "<name>llm_build_xml</name>", "<images>",
             f"<image file={quoteattr(image_file)}>"]
    for b in boxes:
        lines.append(
            f"<box top='{int(round(b['top']))}' left='{int(round(b['left']))}' width='{max(1, int(round(b['width'])))}' "
            f"height='{max(1, int(round(b['height'])))}' id={quoteattr(str(b['id']))} score={quoteattr(str(b['score']))}>"
            f"<label>{escape(b['text'])}</label></box>")
    lines += ["</image>", "</images>", "</dataset>"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=DEFAULT_CFG, help="llm_detect 輸出目錄名")
    ap.add_argument("--c-thresholds", type=float, nargs="+", default=[0.9, 0.8, 0.7])
    ap.add_argument("--c-iou", type=float, default=0.2, help="方案 C：與任何 OCR 框 IoU 低於此值才算缺口")
    ap.add_argument("--out-root", default=OUT_ROOT, help="輸出根目錄（換模型時請另指，避免覆寫 astra 的資料夾）")
    args = ap.parse_args()

    conds = ["base33", "ctrl_uniqid", "A_raw", "A_noq"] + [f"C_t{int(round(t * 100)):03d}" for t in args.c_thresholds]
    dirs = {c: os.path.join(args.out_root, f"xml_{c}") for c in conds}
    for d in dirs.values():
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)

    stats = {c: dict(maps=0, boxes=0, added=0) for c in conds}
    uid = 0
    for f in sorted(os.listdir(OCR_DIR)):
        if not f.endswith(".xml"):
            continue
        ocr_id = f[:-4]
        gt_id = ocr_to_gt(ocr_id)
        llm_path = os.path.join(DATA_DIR, "llm_detect", args.cfg, f"{gt_id}.json")
        if not os.path.exists(os.path.join(ORIG_DIR, f"{gt_id}.jpg")) or not os.path.exists(llm_path):
            continue
        root = read_xml_from(f, OCR_DIR)
        img = root.find(".//image")
        image_file = img.get("file") if img is not None else f"{gt_id}.jpg"
        ocr = ocr_boxes(root)
        llm = json.load(open(llm_path, encoding="utf8"))["boxes"]

        def llm_box(p):
            nonlocal uid
            uid += 1
            return dict(text=p["text"], left=p["left"], top=p["top"], width=p["width"], height=p["height"],
                        id=9_000_000 + uid, score=round(p["conf"], 4))

        out = {}
        out["base33"] = ocr
        ctrl = []
        for b in ocr:
            uid += 1
            ctrl.append(dict(b, id=8_000_000 + uid))
        out["ctrl_uniqid"] = ctrl
        out["A_raw"] = [llm_box(p) for p in llm]
        out["A_noq"] = [llm_box(p) for p in llm if p["text"] != "?"]
        gaps = [p for p in llm if all(box_iou(p, o) < args.c_iou for o in ocr)]
        for t in args.c_thresholds:
            c = f"C_t{int(round(t * 100)):03d}"
            add = [llm_box(p) for p in gaps if p["conf"] >= t]
            out[c] = ocr + add
            stats[c]["added"] += len(add)

        for c, boxes in out.items():
            write_xml(os.path.join(dirs[c], f), image_file, boxes)
            stats[c]["maps"] += 1
            stats[c]["boxes"] += len(boxes)

    print(f"{'條件':<14}{'圖數':>5}{'框數':>8}{'補入':>7}  輸出")
    for c in conds:
        s = stats[c]
        print(f"{c:<14}{s['maps']:>5}{s['boxes']:>8}{s['added'] or '':>7}  {dirs[c]}")
    json.dump(dict(cfg=args.cfg, c_iou=args.c_iou, stats=stats), open(os.path.join(args.out_root, "build_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
