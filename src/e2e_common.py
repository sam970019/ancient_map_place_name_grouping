"""
e2e_common.py — 端到端評估的共用函式（不載入模型）
====================================================
e2e_evaluate.py 在 import 時就會載入模型並解析命令列參數，其他腳本無法單獨借用
它的評估函式；這裡放純資料處理的部分，讓 GNN 與基準方法（如前人方法的分群輸出）
走完全相同的計分流程。
"""

import re
import xml.etree.ElementTree as ET

import numpy as np


def ocr_filename_to_gt(ocr_filename: str) -> str:
    """
    將 OCR 檔名轉換為對應的 GT 檔名。
    規則：首位數字 + 對應數字 = 10
      8xxxx.xml → 2xxxx.xml
      9xxxx.xml → 1xxxx.xml
    """
    first_digit = ocr_filename[0]
    gt_first = str(10 - int(first_digit))
    return gt_first + ocr_filename[1:]


def read_xml_with_encoding_fix(filepath: str):
    """
    讀取 XML 檔案，自動修正 OCR XML 常見的 ISO-8859-1 聲明但內含中文的問題。
    """
    with open(filepath, 'rb') as f:
        raw = f.read()

    # 修正錯誤的 encoding 聲明，統一改成 UTF-8
    raw = re.sub(rb"encoding=['\"]ISO-8859-1['\"]", b"encoding='UTF-8'", raw)
    raw = re.sub(rb"encoding=['\"]iso-8859-1['\"]", b"encoding='UTF-8'", raw)

    return ET.fromstring(raw.decode('utf-8', errors='replace'))


def parse_gt_clusters_from_xml(gt_filepath: str) -> list:
    """
    直接解析 GT XML 檔案，提取每個地名群集的文字與中心座標。
    GT XML 格式：<box top='' left='' width='' height='' id=''><label>字</label></box>
    id='0' 表示不屬於任何地名，跳過。

    Returns:
        list of dict: [{'chars': tuple, 'centroid': (cx, cy)}, ...]
    """
    try:
        tree = ET.parse(gt_filepath)
        root = tree.getroot()
    except ET.ParseError:
        # 嘗試修正編碼後重試
        with open(gt_filepath, 'rb') as f:
            raw = f.read()
        raw = re.sub(rb"encoding=['\"].*?['\"]", b"encoding='UTF-8'", raw)
        root = ET.fromstring(raw.decode('utf-8', errors='replace'))

    groups = {}  # {id: [(char, cx, cy), ...]}

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
            top    = float(box.get('top', 0))
            left   = float(box.get('left', 0))
            width  = float(box.get('width', 0))
            height = float(box.get('height', 0))
        except ValueError:
            continue

        cx = left + width / 2
        cy = top + height / 2

        if box_id not in groups:
            groups[box_id] = []
        groups[box_id].append({
            'char': char, 'cx': cx, 'cy': cy,
            'left': left, 'top': top, 'width': width, 'height': height
        })

    # 轉換成 metrics.py 格式，並保留 BBox 資訊供 IoU 計算
    gt_clusters = []
    for group_id, items in groups.items():
        chars = tuple(item['char'] for item in items)
        cx = float(np.mean([item['cx'] for item in items]))
        cy = float(np.mean([item['cy'] for item in items]))
        gt_clusters.append({
            'chars': chars, 
            'centroid': (cx, cy),
            'boxes': items
        })

    return gt_clusters


def evaluate_map_iou(pred_clusters, gt_clusters, ocr_boxes):
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
                inter_r = min(gl+gw, ol+ow)
                inter_b = min(gt_top+gh, ot+oh)
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
            if pred_status[j] != 'FP': continue
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
