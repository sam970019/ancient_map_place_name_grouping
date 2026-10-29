#GNN edge training method A+B
#加入文字嵌入訓練
import csv
import time
import torch
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.loader import LinkNeighborLoader
from tqdm import tqdm
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import warnings
from matplotlib.patches import Patch

# Focal Loss 函式
def focal_loss_with_logits(inputs, targets, alpha=0.25, gamma=2.0):
    """Binary Focal Loss，適用於正負樣本極度不平衡的情境"""
    bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none') #用"模型判斷其為正邊的機率"&答案(0,1)來計算loss
    pt = torch.exp(-bce_loss)          # 用Loss反推"模型猜對的機率(該題的難易度)"

    # 動態天秤：正樣本(1)給 alpha 權重，負樣本(0)給 (1-alpha) 權重
    alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
    
    focal_loss = alpha_t * (1 - pt) ** gamma * bce_loss
    return focal_loss.mean()


class EpochProfiler:
    """
    TRAIN_PROFILE=1 時記錄每個 epoch 的時間分配；關閉時每個方法立即返回。

    mark(key) 把「上一個 mark 到現在」的時間記到 key。GPU 是非同步執行，
    sync=True 的 mark 會先 cuda.synchronize，GPU 運算時間才會算在正確的項目上。
    """
    KEYS = ('loader', 'wait', 'guard', 'h2d', 'compute', 'eval_train', 'eval', 'other')

    def __init__(self, enabled, device, csv_path):
        self.enabled = enabled
        self.device = device
        self.cuda = enabled and device.type == 'cuda'
        self.csv_path = csv_path

    def _sync(self):
        if self.cuda:
            torch.cuda.synchronize(self.device)

    def _write_row(self, row):
        import csv
        new = not os.path.exists(self.csv_path)
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(['epoch', 'total_s', 'batches'] + [f'{k}_s' for k in self.KEYS] + ['gpu_peak_mib'])
            w.writerow(row)

    def setup_done(self, seconds):
        if not self.enabled:
            return
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)
        tqdm.write(f"⏱️ [profile] 啟動前處理（讀 XML、建圖、載入嵌入、建 loader）：{seconds:.1f}s")
        self._write_row([0, f'{seconds:.2f}', 0] + ['0'] * len(self.KEYS) + ['0'])

    def start_epoch(self):
        if not self.enabled:
            return
        self._sync()
        self.t = dict.fromkeys(self.KEYS, 0.0)
        self.batches = 0
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        self._start = self._last = time.perf_counter()

    def mark(self, key, sync=False):
        if not self.enabled:
            return
        if sync:
            self._sync()
        now = time.perf_counter()
        self.t[key] += now - self._last
        self._last = now

    def batch_done(self):
        if self.enabled:
            self.batches += 1

    def end_epoch(self, epoch):
        if not self.enabled:
            return
        self.mark('other', sync=True)
        total = self._last - self._start
        peak = torch.cuda.max_memory_allocated(self.device) / 2**20 if self.cuda else 0.0
        parts = ' | '.join(f"{k} {v:.1f}s ({v / total:.0%})" for k, v in self.t.items())
        tqdm.write(f"⏱️ [profile] epoch {epoch}：共 {total:.1f}s，{self.batches} batches | {parts} | GPU 峰值 {peak:.0f} MiB")
        self._write_row([epoch, f'{total:.2f}', self.batches]
                        + [f'{self.t[k]:.2f}' for k in self.KEYS] + [f'{peak:.0f}'])

# 從拆分出去的模組引入
import config
from models import Model
from data import (
    read_xml, read_xml_from, fit_global_scaler, build_graph, build_data,
    make_edge_label_attr_transform, build_global_vocab, load_fasttext_embeddings,
    vocab, vocab_count, DATA_DIR, split_maps, THRESHOLD_GRID
)
from metrics import get_place_metrics
from visualize import visualize_all_maps

# 抑制字體相關警告與日誌
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# 設置中文字體
def setup_matplotlib_fonts():
    from matplotlib import font_manager
    import platform
    from data import DATA_DIR

    # 1. 優先嘗試載入專案內的字體檔
    font_files = ['msjh.ttc', 'msjhbd.ttc', 'msjhl.ttc', 'SimHei.ttf']
    font_path = None
    for f in font_files:
        p = os.path.join(DATA_DIR, 'font', f)
        if os.path.exists(p):
            font_path = p
            break
            
    if font_path:
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = prop.get_name()
        # print(f"成功加載字體檔：{font_path}，名稱：{prop.get_name()}")
        return font_path

    # 2. 若無字體檔，依平台使用系統字體
    sys_str = platform.system()
    if sys_str == "Windows":
        plt.rcParams['font.family'] = ['Microsoft JhengHei', 'SimHei', 'sans-serif']
    elif sys_str == "Darwin":
        plt.rcParams['font.family'] = ['Heiti TC', 'sans-serif']
    else:
        plt.rcParams['font.family'] = ['Noto Sans CJK TC', 'WenQuanYi Micro Hei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    return None

font_path = setup_matplotlib_fonts()

# 繪製邊特徵可視化地圖
def draw_edges_on_map(xml_file, ocr_results, data, output_path, dataset_type='train'):
    from data import DATA_DIR
    img_path = os.path.join(DATA_DIR, "map", "original", f"{os.path.splitext(xml_file)[0]}.jpg")
    if not os.path.exists(img_path):
        # print(f"找不到原始地圖: {img_path}")
        return

    # print(f"繪製 {dataset_type} 地圖邊特徵: {xml_file}")
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)

    coords = {i: (item["x"] + item["w"] // 2, item["y"] + item["h"] // 2) for i, item in enumerate(ocr_results)}

    # 繪製預測邊（此處作為展示使用 edge_label_index 作為候選）
    edge_index = data.edge_label_index.cpu().numpy()
    edge_label = data.edge_label.cpu().numpy()

    for idx in range(edge_index.shape[1]):
        u, v = edge_index[0, idx], edge_index[1, idx]
        if u in coords and v in coords:
            pt1 = coords[u]
            pt2 = coords[v]
            color = "red" if edge_label[idx] == 1 else "blue"
            width = 3 if edge_label[idx] == 1 else 1
            draw.line([pt1, pt2], fill=color, width=width)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)


# 繪製 OCR 標記地圖
def save_text_boxes_map(xml_file, ocr_results, img_file):
    from data import DATA_DIR
    img_path = os.path.join(DATA_DIR, "map", "original", f"{os.path.splitext(xml_file)[0]}.jpg")
    if not os.path.exists(img_path):
        # print(f"找不到原始地圖: {img_path}")
        return

    if font_path:
        max_x = 3600
        max_y = 3600
        img = Image.new("RGB", (max_x, max_y), "white")
        draw = ImageDraw.Draw(img)
        for node in ocr_results:
            x, y, w, h = node["x"], node["y"], node["w"], node["h"]
            draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
            text = node["text"] if "text" in node else "?"
            font_size = 24
            font = ImageFont.truetype(font_path, font_size)
            try:
                bbox = font.getbbox(text)
                text_height = bbox[3] - bbox[1]
            except AttributeError:
                # 舊版 Pillow
                text_width, text_height = font.getsize(text)
            draw.text((x, y - text_height), text, font=font, fill="black")
        os.makedirs(os.path.join(DATA_DIR, 'map', 'marked_text'), exist_ok=True)
        output_path = os.path.join(DATA_DIR, 'map', 'marked_text', f'{os.path.splitext(xml_file)[0]}.png')
        img.save(output_path)


# ==================== 主程式開始 ====================
if __name__ == '__main__':
    start_time = time.time()
    # 訓練隨機性由 TRAIN_SEED 控制；資料切分在 split_maps 內另以 RANDOM_SEED 決定
    torch.manual_seed(config.TRAIN_SEED)
    np.random.seed(config.TRAIN_SEED)
    random.seed(config.TRAIN_SEED)
    folder_path = DATA_DIR
    # 使用 config 定義的 XML_DIR，並過濾掉不必要的檔案 (例如 B 開頭的檔案)
    xml_files = [f for f in os.listdir(config.XML_DIR) if f.endswith(".xml") and not f.startswith("B-")]
    xml_files = sorted(xml_files)

    # 全局正規化參數改在「分割訓練 / 測試集之後」再算（見下方），
    # 因為 bigram_lp 特徵需要先用訓練切分的檔案建好字元接續表。

    # 使用 config 中的設定
    epochs = config.EPOCHS
    map_num = config.MAP_NUM
    lr = config.LEARNING_RATE
    in_channels = config.IN_CHANNELS
    edge_feature_dim = config.EDGE_FEATURE_DIM

    # 在迴圈外初始化
    all_train_data_x = []
    all_train_edge_index = []
    all_train_edge_label_index = []
    all_train_edge_label = []
    all_train_edge_label_attr = []
    all_train_edge_attr = []
    all_train_label_id = []
    all_train_edge_label_hard = []   # 難負邊遮罩（NEG_KEEP_HARD 用），另存、不進 Data


    # 全局字典，用於存儲每個地圖的text_idx信息
    all_maps_text_idx = {}

    # 初始化全局字典
    train_global_text_idx = {}

    # 初始化節點偏移量
    train_node_offset = 0

    # ----------------- 建立詞彙表與 Embedding -----------------
    embedding_dim = 300  # 可調整

    # 構建全局詞彙表
    build_global_vocab(xml_files)
    # 重新從 data 模組取得最新的 vocab 和 vocab_count
    import data as data_module

    #======================fasttext===============================
    embedding_dim, vocab_size, embedding_matrix = load_fasttext_embeddings()

    # 初始化模型和優化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(in_channels=in_channels, hidden_channels=config.HIDDEN_CHANNELS, out_channels=config.OUT_CHANNELS, edge_feature_dim=edge_feature_dim, vocab_size=data_module.vocab_count, embedding_dim=embedding_dim, embedding_matrix=embedding_matrix).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # === 新增 Scheduler ===
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10 # mode='max'因為看的是F1(越高越好)、factor=0.5代表衰減為一半、patience=10代表忍受10個epoch沒進步就降LR
    )

    # 確保必要的資料夾存在
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # 1. 分割訓練集與測試集
    # 訓練 / 測試切法固定（RANDOM_SEED），驗證集再從訓練切分中抽出
    train_xml_files, val_xml_files, test_xml_files = split_maps(xml_files, config)
    config.VAL_FILES = tuple(val_xml_files)   # 隨 checkpoint 封存，評估端還原同一批驗證圖
    train_map_num = len(train_xml_files)
    test_map_num  = len(test_xml_files)
    print(f"隨機 {round(config.TRAIN_RATIO*100)}/{round((1-config.TRAIN_RATIO)*100)} 分割，"
          f"驗證集 {len(val_xml_files)} 張取自訓練切分")

    if not val_xml_files:
        print("⚠️  沒有驗證集（VAL_NUM=0 或 OCR 模式）：退回舊協定，以測試集選 checkpoint 與學習率排程")
    print(f"訓練集地圖數量: {len(train_xml_files)}")
    print(f"驗證集地圖數量: {len(val_xml_files)}" + (f"（{', '.join(os.path.splitext(f)[0] for f in val_xml_files)}）" if val_xml_files else ""))
    print(f"測試集地圖數量: {len(test_xml_files)}")
    print(f"計劃處理的訓練地圖數: {train_map_num}")
    print(f"計劃處理的測試地圖數: {test_map_num}")

    # ----------------- 字元接續表（bigram_lp 特徵） -----------------
    # 只用「實際會進訓練迴圈的檔案」建表（與下方預載入迴圈的 SKIP / STOP / 上限規則一致），
    # 測試切分的地名絕不進表；表隨 checkpoint 一起封存，評估端直接還原。
    _prior_files = []
    for _f in train_xml_files:
        if _f in config.SKIP_FILES:
            continue
        if _f == config.STOP_FILE or len(_prior_files) >= train_map_num:
            break
        _prior_files.append(_f)
    if 'bigram_lp' in data_module._extra_edge_features():
        data_module.build_text_prior(_prior_files, config.TRAIN_XML_DIR)
        print(f"📖 {data_module.text_prior.summary()}")
        # 留一圖法靠 <image file> 的主檔名（build_graph 的 map_id）對到 XML 檔名（per_file 的鍵）；
        # 兩者對不上時會靜默失效（每條正邊都查得到表），所以先檢查
        for _f in _prior_files:
            _root = data_module.read_xml_from(_f, config.TRAIN_XML_DIR)
            _ids = {os.path.splitext(im.get("file"))[0] for im in _root.findall(".//image")}
            assert _ids == {os.path.splitext(_f)[0]} and _ids <= set(data_module.text_prior.per_file),                 f"{_f}：<image file> 主檔名 {_ids} 與 XML 檔名不一致，bigram 留一圖法會失效"
    else:
        data_module.text_prior = None

    # 一行印出這次實驗的所有開關，避免環境變數沒帶到卻靜靜地跑了一版錯的設定
    print(f"⚙️  {config.EXPERIMENT_NAME}: EDGE_ANGLE_MODE={config.EDGE_ANGLE_MODE} | EXTRA={len(config.EXTRA_EDGE_FEATURES)} 個 | "
          f"NXN_LIMIT={config.NXN_EDGE_DISTANCE_LIMIT} | NEG_SAMPLE_RATIO={config.NEG_SAMPLE_RATIO:.4f} | "
          f"NEG_KEEP_HARD={config.NEG_KEEP_HARD} (ratio {config.HARD_NEG_RATIO}, lin {config.HARD_NEG_INCLUDE_LIN}) | "
          f"ORIENT_AUG={config.ORIENT_AUG} (period {config.ORIENT_AUG_PERIOD}) | EPOCHS={config.EPOCHS} | TRAIN_SEED={config.TRAIN_SEED} | "
          f"GNN_TYPE={config.GNN_TYPE} (邊特徵進注意力 {config.GNN_EDGE_IN_ATTN})")

    # 計算全局正規化參數（節點 / 候選邊 / MP 邊三個 scaler），只用實際進訓練迴圈的地圖擬合，
    # 驗證集與測試集都不參與；參數隨 checkpoint 封存，評估端直接還原
    fit_global_scaler(_prior_files)
    print(f"📐 正規化參數以 {len(_prior_files)} 張訓練圖擬合")
    print(f"🧮 邊特徵 {config.EDGE_FEATURE_DIM} 維: {', '.join(data_module.edge_feature_names())}")

    #收集訓練集資料（快取原始完美圖結構）

    train_graphs_cache = []  # 記憶體快取：每張地圖的原始完美圖結構
    train_graphs_cache_flip = []  # ORIENT_AUG：翻轉副本
    n_train = 0
    train_bar = tqdm(train_xml_files, desc="預載入訓練地圖")
    for file in train_bar:
        if file in config.SKIP_FILES:
            continue
        if file == config.STOP_FILE or n_train >= train_map_num:
            break
        n_train += 1
        train_bar.set_postfix({"預載入訓練地圖": file})
        root = read_xml_from(file, config.TRAIN_XML_DIR)
        data, text_idx, place_groups, place_name_edge, ocr_results = build_graph(root, model)
        train_graphs_cache.append((data, text_idx, place_name_edge, ocr_results))
        if config.ORIENT_AUG:
            # 直橫書翻轉副本：同一張圖沿反對角線翻轉後重建（拓樸 / 特徵全部重算，map_id 不變）
            data_f, text_idx_f, _, pne_f, ocr_f = build_graph(data_module.reflect_xml_root(root), model)
            train_graphs_cache_flip.append((data_f, text_idx_f, pne_f, ocr_f))

    # 用完美 GT 圖結構建立 combined_train（供訓練期間監控評估使用）
    for data_orig, text_idx, place_name_edge, ocr_results in train_graphs_cache:
        data = data_orig.clone()  # Clone 避免 build_data 修改快取

        for idx, info in text_idx.items():
            local_idx = idx + train_node_offset
            train_global_text_idx[local_idx] = info
        
        train_data = build_data(data, place_name_edge, ocr_results, is_test=False)
 
        all_train_data_x.append(train_data.x)
        train_edge = train_data.edge_index + train_node_offset
        all_train_edge_index.append(train_edge)
        train_edge_label_index = train_data.edge_label_index + train_node_offset
        all_train_edge_label_index.append(train_edge_label_index)
        all_train_edge_label.append(train_data.edge_label)
        all_train_edge_label_attr.append(train_data.edge_label_attr)
        all_train_edge_attr.append(train_data.edge_attr)
        all_train_label_id.append(train_data.label_id)
        all_train_edge_label_hard.append(train_data.edge_label_hard)
        
        train_node_offset += train_data.x.size(0)

    # ================== 收集評估用地圖（驗證集 / 測試集）並記錄每張地圖資訊 ==================
    def build_eval_set(files, xml_dir, max_maps, desc):
        """把一批評估地圖建成合併的 NxN 候選邊圖，回傳 (combined Data, 全局文字索引, 每張地圖資訊)。"""
        xs, eis, elis, els, elas, eas, lids = [], [], [], [], [], [], []
        global_text_idx, map_infos = {}, []
        edge_offset = node_offset = n = 0
        bar = tqdm(files, desc=desc)
        for file in bar:
            if file in config.SKIP_FILES:
                continue
            if file == config.STOP_FILE or n >= max_maps:
                break
            n += 1
            bar.set_postfix({"處理中": file})
            root = read_xml_from(file, xml_dir)
            data, text_idx, place_groups, place_name_edge, ocr_results = build_graph(root, model)

            for idx, info in text_idx.items():
                global_text_idx[idx + node_offset] = info

            d = build_data(data, place_name_edge, ocr_results, is_test=True)
            xs.append(d.x)
            eis.append(d.edge_index + node_offset)
            elis.append(d.edge_label_index + node_offset)
            els.append(d.edge_label)
            elas.append(d.edge_label_attr)
            eas.append(d.edge_attr)
            lids.append(d.label_id)

            num_edges = d.edge_label_index.shape[1]
            img_stem = os.path.splitext(file)[0]
            map_infos.append({
                "file": file,
                "ocr_results": ocr_results,
                "img_file": os.path.join(folder_path, "map/marked", f"{img_stem}.png"),
                "num_edges": num_edges,
                "edge_offset": edge_offset,
                "node_offset": node_offset,
            })
            edge_offset += num_edges
            node_offset += d.x.size(0)

        combined = Data(
            x=torch.cat(xs),
            edge_index=torch.cat(eis, dim=1),
            edge_attr=torch.cat(eas, dim=0),
            edge_label_index=torch.cat(elis, dim=1),
            edge_label=torch.cat(els),
            edge_label_attr=torch.cat(elas, dim=0),
            label_id=torch.cat(lids, dim=0)
        )
        return combined, global_text_idx, map_infos

    combined_test, test_global_text_idx, test_map_infos = build_eval_set(
        test_xml_files, config.TEST_XML_DIR, test_map_num, "處理測試地圖")
    print(f"處理的測試地圖數: {len(test_map_infos)}")

    # 驗證集取自訓練切分，所以從 TRAIN_XML_DIR 讀（GT 模式下與 TEST_XML_DIR 同為 xml_revised）
    if val_xml_files:
        combined_val, val_global_text_idx, _ = build_eval_set(
            val_xml_files, config.TRAIN_XML_DIR, len(val_xml_files), "處理驗證地圖")
        print(f"處理的驗證地圖數: {len(val_xml_files)}")
    else:
        combined_val, val_global_text_idx = combined_test, test_global_text_idx

    # 4. 合併訓練和測試集的資料
    combined_train = Data(
        x=torch.cat(all_train_data_x),
        edge_index=torch.cat(all_train_edge_index, dim=1),
        edge_attr=torch.cat(all_train_edge_attr, dim=0),
        edge_label_index=torch.cat(all_train_edge_label_index, dim=1),
        edge_label=torch.cat(all_train_edge_label),
        edge_label_attr=torch.cat(all_train_edge_label_attr,dim=0),
        label_id=torch.cat(all_train_label_id,dim=0)
    )
    combined_train_hard = torch.cat(all_train_edge_label_hard)
    # 各張圖的清單已合併進 combined_train，留著等於整份候選邊多存一份（不設距離限制時約 3 GB）
    del (all_train_data_x, all_train_edge_index, all_train_edge_attr, all_train_edge_label_index,
         all_train_edge_label, all_train_edge_label_attr, all_train_label_id, all_train_edge_label_hard)

    def assemble_train(graphs_cache):
        """把一批 (data, text_idx, place_name_edge, ocr) 組成 (combined Data, 難負邊遮罩)。"""
        _x, _ei, _eli, _el, _ela, _ea, _lid, _hard = [], [], [], [], [], [], [], []
        off = 0
        for d0, _, pne, ocr in graphs_cache:
            td = build_data(d0.clone(), pne, ocr, is_test=False)
            _x.append(td.x); _ei.append(td.edge_index + off); _eli.append(td.edge_label_index + off)
            _el.append(td.edge_label); _ela.append(td.edge_label_attr); _ea.append(td.edge_attr)
            _lid.append(td.label_id); _hard.append(td.edge_label_hard)
            off += td.x.size(0)
        return Data(x=torch.cat(_x), edge_index=torch.cat(_ei, dim=1), edge_attr=torch.cat(_ea, dim=0),
                    edge_label_index=torch.cat(_eli, dim=1), edge_label=torch.cat(_el),
                    edge_label_attr=torch.cat(_ela, dim=0), label_id=torch.cat(_lid, dim=0)), torch.cat(_hard)

    combined_train_flip, combined_train_flip_hard = (None, None)
    if config.ORIENT_AUG:
        combined_train_flip, combined_train_flip_hard = assemble_train(train_graphs_cache_flip)
        assert int(combined_train.edge_label.sum()) == int(combined_train_flip.edge_label.sum())
        assert combined_train_flip.x.size(0) == combined_train.x.size(0)

        # 方向統計：翻轉後直書 / 橫書正邊的比例應互換（用原始像素座標判斷）
        def _vert_share(cache):
            v = n = 0
            for _, _, pne, ocr in cache:
                for edges in pne.values():
                    for a, b in edges:
                        n += 1; v += abs(ocr[b]["y"] - ocr[a]["y"]) > abs(ocr[b]["x"] - ocr[a]["x"])
            return v / max(n, 1)
        print(f"🔁 直橫書翻轉增強啟用：每 {config.ORIENT_AUG_PERIOD} 個 epoch 用一次翻轉圖（epoch % {config.ORIENT_AUG_PERIOD} == 0）| "
              f"正邊直書比例 原圖 {_vert_share(train_graphs_cache)*100:.1f}% → 翻轉圖 {_vert_share(train_graphs_cache_flip)*100:.1f}%")

    # --- MP 邊建構結果摘要（sanity check：換 MP 模式後這裡的邊數應明顯變化）---
    _mp_mode = getattr(config, 'MP_EDGE_MODE', 'id')
    _tr_n, _tr_e = combined_train.x.size(0), combined_train.edge_index.size(1)
    _te_n, _te_e = combined_test.x.size(0), combined_test.edge_index.size(1)
    print(f"\n🔗 MP 邊模式 = {_mp_mode} | 拓樸座標 = {getattr(config, 'MP_ANCHOR', 'tl')}"
          + (f" | k = {getattr(config, 'MP_KNN_K', 8)}" if 'knn' in str(_mp_mode) else ""))
    print(f"   訓練集: 節點 {_tr_n} | MP邊 {_tr_e} (平均度數 {_tr_e / max(_tr_n, 1):.2f}) "
          f"| 候選邊 {combined_train.edge_label_index.size(1)}")
    print(f"   測試集: 節點 {_te_n} | MP邊 {_te_e} (平均度數 {_te_e / max(_te_n, 1):.2f}) "
          f"| 候選邊 {combined_test.edge_label_index.size(1)}")
    print(f"   訓練候選模式 = {getattr(config, 'TRAIN_SAMPLE_MODE', '?')} | "
          f"測試候選模式 = {getattr(config, 'TEST_SAMPLE_MODE', '?')}\n")

    # 鄰居取樣深度必須與 GNN 層數一致：層數 L 需要 L 跳的鄰域，
    # 否則最外層會在不完整的鄰域上聚合，且感受野實際上被採樣器截斷。
    # GNN_LAYERS=2 時為 [10, 5]，與改動前完全相同。
    _L = int(getattr(config, 'GNN_LAYERS', 2))
    NUM_NEIGHBORS = [10] + [5] * (_L - 1)
    print(f"🔁 鄰居取樣深度 num_neighbors = {NUM_NEIGHBORS}（GNN_LAYERS={_L}）")

    # 5. 創建訓練資料載入器
    if getattr(config, 'NUM_WORKERS', -1) >= 0:
        workers = config.NUM_WORKERS
    elif os.name == 'nt':
        workers = 0  # Windows 本機建議用 0，最穩且不會出錯
    else:
        workers = min(os.cpu_count() // 2, 8) #讀取系統核心數的1/2 當作讀取資料的進程數 但不得大於8
    print(f"👷 DataLoader workers = {workers}")

    def sampler_graph(combined):
        """
        交給 LinkNeighborLoader 的圖只放鄰居取樣需要的部分（節點特徵、MP 邊與其特徵、文字 id）。
        候選邊的索引與標籤另由 edge_label_index / edge_label 參數傳入，候選邊特徵由 transform 依
        input_id 取回；整份候選邊（不設距離限制時約 3 GB）不必跟著圖送進每個 worker 的共享記憶體。
        """
        return Data(x=combined.x, edge_index=combined.edge_index,
                    edge_attr=combined.edge_attr, label_id=combined.label_id)

    train_loader = LinkNeighborLoader(
        data=sampler_graph(combined_train),
        num_neighbors=NUM_NEIGHBORS,
        edge_label_index=combined_train.edge_label_index,
        edge_label=combined_train.edge_label,
        transform=make_edge_label_attr_transform(combined_train.edge_label_attr),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=workers,          # 1. 多人搬磚 (解決 CPU 抽樣瓶頸)
        persistent_workers=(workers > 0), # 2. 工人常駐不解散 (加速 Epoch 切換)
        pin_memory=True               # 3. 開啟 GPU 快速通關車道 (加速資料上傳到顯卡)
    )

    # 6. 修改評估指標的追蹤列表
    train_losses = []
    place_metrics = {
        'train': {'precision': [], 'recall': [], 'f1': [], 'edge_p': [], 'edge_r': [], 'edge_f1': []},
        'val': {'precision': [], 'recall': [], 'f1': [], 'edge_p': [], 'edge_r': [], 'edge_f1': []}
    }

    def subsample_candidate_edges(combined, ratio, seed, hard=None):
        """
        對候選邊做「正邊全留、負邊均勻抽 ratio」的子集，回傳
        (edge_label_index, edge_label, edge_label_attr) 三個對齊的張量。
        只影響訓練時進入 loss 的邊；combined 本身不動，推論仍掃全部候選邊。
        hard（NEG_KEEP_HARD）：與 edge_label 對齊的布林遮罩，為 True 的負邊全留不抽。
        """
        label = combined.edge_label
        pos = torch.nonzero(label == 1, as_tuple=False).view(-1)
        g = torch.Generator().manual_seed(int(seed))
        if hard is not None:
            neg_keep = torch.nonzero((label == 0) & hard, as_tuple=False).view(-1)
            neg = torch.nonzero((label == 0) & ~hard, as_tuple=False).view(-1)
            hr = float(getattr(config, 'HARD_NEG_RATIO', 1.0))
            if hr < 1.0:                       # 難負邊也抽，但用較高的比例
                neg_keep = neg_keep[torch.randperm(neg_keep.numel(), generator=g)[:int(round(neg_keep.numel() * hr))]]
        else:
            neg_keep = pos.new_empty(0)
            neg = torch.nonzero(label == 0, as_tuple=False).view(-1)
        k = int(round(neg.numel() * ratio))
        neg_sel = neg[torch.randperm(neg.numel(), generator=g)[:k]]
        sel = torch.cat([pos, neg_keep, neg_sel])
        sel = sel[torch.randperm(sel.numel(), generator=g)]   # 打散正負順序
        return (combined.edge_label_index[:, sel],
                combined.edge_label[sel],
                combined.edge_label_attr[sel])

    _NEG_RATIO = float(getattr(config, 'NEG_SAMPLE_RATIO', 1.0))
    if not (0.0 < _NEG_RATIO <= 1.0):
        raise ValueError(f"NEG_SAMPLE_RATIO 必須在 (0, 1]，目前為 {_NEG_RATIO}")
    _KEEP_HARD = bool(getattr(config, 'NEG_KEEP_HARD', False)) and _NEG_RATIO < 1.0
    if _NEG_RATIO < 1.0:
        _n_pos = int((combined_train.edge_label == 1).sum())
        _n_neg = int((combined_train.edge_label == 0).sum())
        _n_hard = int(((combined_train.edge_label == 0) & combined_train_hard).sum())
        _HR = float(getattr(config, 'HARD_NEG_RATIO', 1.0))
        _n_use = (int(round(_n_hard * _HR)) + int(round((_n_neg - _n_hard) * _NEG_RATIO))) if _KEEP_HARD else int(round(_n_neg * _NEG_RATIO))
        print(f"🎲 負邊抽樣啟用：NEG_SAMPLE_RATIO={_NEG_RATIO:.4f}，每個 epoch 重抽 | "
              f"正邊 {_n_pos:,} 全留，負邊 {_n_neg:,} → 約 {_n_use:,} "
              f"(正:負 1:{_n_neg/_n_pos:.0f} → 1:{_n_use/_n_pos:.0f})")
        if _KEEP_HARD:
            print(f"🎯 難負邊全留（NEG_KEEP_HARD）：大字(平均字級>{config.HARD_NEG_SIZE_REL}×中位)"
                  f"{' 或 線性通名' if config.HARD_NEG_INCLUDE_LIN else ''} 的負邊 "
                  f"{_n_hard:,} 條（佔負邊 {_n_hard/_n_neg*100:.1f}%）"
                  + ("不抽樣" if _HR >= 1.0 else f"以 {_HR:.3f} 抽樣（其餘負邊 {_NEG_RATIO:.3f}）"))

    _BATCH_GUARD = os.environ.get("BATCH_GUARD", "1") != "0"

    def check_batch(b, vocab_size, epoch, it):
        """
        CPU 端檢查這個 batch 所有會被拿去索引的張量，在它上 GPU 之前攔下越界。

        背景：訓練曾多次在隨機 epoch 出現 CUDA 端的 index-out-of-bounds assert
        （index_select / ScatterGather 皆有），CUDA 為非同步，traceback 指不到案發處。
        這裡先在 CPU 驗證；若攔到 → 是 sampler/資料問題且 batch 已存檔可離線重現；
        若這裡從未攔到而 CUDA 仍 assert → 索引是在 GPU 上才壞的，指向硬體/驅動。
        成本：每 batch 幾個 max()，可忽略。BATCH_GUARD=0 可關閉。
        """
        n = int(b.x.size(0))
        problems = []

        def _rng(name, t, hi):
            if t.numel() == 0:
                return
            lo_, hi_ = int(t.min()), int(t.max())
            if lo_ < 0 or hi_ >= hi:
                problems.append(f"{name}: 範圍 [{lo_}, {hi_}]，允許 [0, {hi - 1}]")

        _rng("label_id",         b.label_id,         vocab_size)
        _rng("edge_index",       b.edge_index,       n)
        _rng("edge_label_index", b.edge_label_index, n)
        for name in ("x", "edge_attr", "edge_label_attr"):
            t = getattr(b, name, None)
            if t is not None and t.numel() and not torch.isfinite(t).all():
                problems.append(f"{name}: 含 NaN/Inf")
        if b.edge_label_attr.size(0) != b.edge_label_index.size(1):
            problems.append(f"edge_label_attr 列數 {b.edge_label_attr.size(0)} != 邊數 {b.edge_label_index.size(1)}")
        if problems:
            dump = os.path.join(config.MODEL_SAVE_DIR,
                                f"bad_batch_{config.EXPERIMENT_NAME}_ep{epoch}_it{it}.pt")
            torch.save(b.cpu(), dump)
            raise RuntimeError(
                "\n".join([f"[BATCH_GUARD] epoch {epoch} batch {it} 索引/數值異常，已存至 {dump}",
                           f"  節點 {n}，MP 邊 {b.edge_index.size(1)}，候選邊 {b.edge_label_index.size(1)}，"
                           f"vocab {vocab_size}"] + [f"  - {q}" for q in problems]))

    def train_model(model, train_loader, combined_train, combined_val, epochs):
        global train_losses, place_metrics
        best_val_f1 = 0.0
        train_eval_every = int(getattr(config, 'TRAIN_EVAL_EVERY', 0))
        model_path = os.path.join(config.MODEL_SAVE_DIR, f'trained_model_{config.EXPERIMENT_NAME}.pth')
        prof = EpochProfiler(getattr(config, 'TRAIN_PROFILE', False), device,
                             os.path.join(config.OUTPUT_DIR, f'profile_{config.EXPERIMENT_NAME}.csv'))
        prof.setup_done(time.time() - start_time)

        # 逐 epoch 紀錄（loss、學習率、驗證集分數），每個 epoch 結束就寫入，訓練中斷也保得住
        history_path = os.path.join(config.OUTPUT_DIR, f'history_{config.EXPERIMENT_NAME}.csv')
        with open(history_path, 'w', newline='', encoding='utf-8') as fh:
            csv.writer(fh).writerow(['epoch', 'loss', 'lr', 'val_p', 'val_r', 'val_f1',
                                     'val_edge_p', 'val_edge_r', 'val_edge_f1', 'train_f1',
                                     'is_best', 'epoch_seconds'])

        # 訓練循環
        pbar = tqdm(range(1, epochs + 1), desc="Training Progress")
        for epoch in pbar:
            prof.start_epoch()
            epoch_t0 = time.time()
            # === 每個 Epoch 是否需要重建 loader ===
            #   負邊抽樣：圖不動，只重抽進入 loss 的負邊
            #   翻轉增強：每 ORIENT_AUG_PERIOD 個 epoch 換用翻轉圖
            # 兩者都關閉時沿用預先建好的 train_loader。
            _use_flip = bool(config.ORIENT_AUG) and (epoch % config.ORIENT_AUG_PERIOD == 0)   # 每 PERIOD 個 epoch 用一次翻轉圖
            if _NEG_RATIO < 1.0 or config.ORIENT_AUG:
                if _use_flip:
                    combined_train_epoch, hard_epoch = combined_train_flip, combined_train_flip_hard
                else:
                    combined_train_epoch, hard_epoch = combined_train, combined_train_hard
                if _NEG_RATIO < 1.0:
                    _eli, _el, _ela = subsample_candidate_edges(
                        combined_train_epoch, _NEG_RATIO, config.TRAIN_SEED * 1000 + epoch,
                        hard=hard_epoch if _KEEP_HARD else None)
                else:
                    _eli, _el, _ela = (combined_train_epoch.edge_label_index,
                                       combined_train_epoch.edge_label,
                                       combined_train_epoch.edge_label_attr)
                # 每個 epoch 重建一次，worker 用完即結束（常駐只會讓舊 worker 與其共享記憶體殘留）
                epoch_loader = LinkNeighborLoader(
                    data=sampler_graph(combined_train_epoch),
                    num_neighbors=NUM_NEIGHBORS,
                    edge_label_index=_eli,
                    edge_label=_el,
                    transform=make_edge_label_attr_transform(_ela),
                    batch_size=config.BATCH_SIZE,
                    shuffle=True,
                    num_workers=workers,
                    persistent_workers=False,
                    pin_memory=True
                )
            else:
                epoch_loader = train_loader
            prof.mark('loader')

            model.train()
            total_loss = 0

            batch_bar = tqdm(epoch_loader, desc=f"  Epoch {epoch}", leave=False)
            for _it, sampled_data in enumerate(batch_bar):
                prof.mark('wait')
                optimizer.zero_grad()
                if _BATCH_GUARD:
                    check_batch(sampled_data, model.embedding.num_embeddings, epoch, _it)
                prof.mark('guard')
                sampled_data = sampled_data.to(device)
                prof.mark('h2d', sync=True)
                pred = model(sampled_data)
                if getattr(config, 'USE_FOCAL_LOSS', False):
                    loss = focal_loss_with_logits(
                        pred,
                        sampled_data.edge_label.float(),
                        alpha=config.FOCAL_ALPHA,
                        gamma=config.FOCAL_GAMMA
                    )
                else:
                    loss = F.binary_cross_entropy_with_logits(pred, sampled_data.edge_label.float())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batch_bar.set_postfix({'batch_loss': f'{loss.item():.6f}'})
                prof.mark('compute', sync=True)
                prof.batch_done()
            prof.mark('wait')

            avg_loss = total_loss / len(epoch_loader)
            train_losses.append(avg_loss)
            
            # 評估訓練集（只為了畫學習曲線，計算量大）：TRAIN_EVAL_EVERY=N 時每 N 個 epoch 算一次，其餘沿用前次值
            if train_eval_every > 0:
                if epoch == 1 or epoch % train_eval_every == 0 or epoch == epochs:
                    train_place_p, train_place_r, train_place_f1, train_edge_p, train_edge_r, train_edge_f1, _, _ = get_place_metrics(model, combined_train, device, train_global_text_idx)
                else:
                    train_place_p = place_metrics['train']['precision'][-1]
                    train_place_r = place_metrics['train']['recall'][-1]
                    train_place_f1 = place_metrics['train']['f1'][-1]
                    train_edge_p = place_metrics['train']['edge_p'][-1]
                    train_edge_r = place_metrics['train']['edge_r'][-1]
                    train_edge_f1 = place_metrics['train']['edge_f1'][-1]
                place_metrics['train']['precision'].append(train_place_p)
                place_metrics['train']['recall'].append(train_place_r)
                place_metrics['train']['f1'].append(train_place_f1)
                place_metrics['train']['edge_p'].append(train_edge_p)
                place_metrics['train']['edge_r'].append(train_edge_r)
                place_metrics['train']['edge_f1'].append(train_edge_f1)
            prof.mark('eval_train', sync=True)

            # 每個 epoch 評估驗證集：選最佳 checkpoint、驅動學習率排程（測試集只在訓練結束後評估一次）
            # 閾值用固定的 config.PRED_THRESHOLD；最終閾值在訓練後另外於驗證集上掃描決定
            val_place_p, val_place_r, val_place_f1, val_edge_p, val_edge_r, val_edge_f1, _, _ = get_place_metrics(model, combined_val, device, val_global_text_idx)
            prof.mark('eval', sync=True)
            place_metrics['val']['precision'].append(val_place_p)
            place_metrics['val']['recall'].append(val_place_r)
            place_metrics['val']['f1'].append(val_place_f1)
            place_metrics['val']['edge_p'].append(val_edge_p)
            place_metrics['val']['edge_r'].append(val_edge_r)
            place_metrics['val']['edge_f1'].append(val_edge_f1)

            # === 新增 Scheduler.step ===
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_place_f1)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < current_lr:
                tqdm.write(f"📉 [Epoch {epoch}] F1 進入瓶頸，Learning Rate 自動調降為: {new_lr:.6f}")
            
            # Checkpoint: 儲存最佳模型 (包含 config 參數)
            is_best = val_place_f1 > best_val_f1
            if is_best:
                best_val_f1 = val_place_f1
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    # 名稱帶 KEY / ENDPOINT / DEPLOYMENT 的設定（API 金鑰、端點與部署名稱）不寫進 checkpoint
                    'config': {k: getattr(config, k) for k in dir(config) if not k.startswith('__') and 'KEY' not in k.upper() and 'ENDPOINT' not in k.upper() and 'DEPLOYMENT' not in k.upper() and not callable(getattr(config, k)) and not str(getattr(config, k)).startswith('<module')},
                    # 詞彙表（字 → 索引）；評估端以 data.restore_vocab 還原，不必依賴 xml_revised 與建表時相同
                    'vocab': dict(data_module.vocab),
                    # bigram_lp 的字元接續表（只含訓練切分的地名）；沒用到該特徵時為 None
                    'text_prior': data_module.text_prior.state_dict() if data_module.text_prior is not None else None,
                    'best_epoch': epoch,
                    'scalers': data_module.scalers_state(),
                }
                torch.save(checkpoint, model_path)
                
            # 將關鍵指標即時更新到進度條後方
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                **({'Train_F1': f'{train_place_f1:.4f}'} if train_eval_every > 0 else {}),
                'Val_F1': f'{val_place_f1:.4f}',
                'Best_Val_F1': f'{best_val_f1:.4f}'
            })
            with open(history_path, 'a', newline='', encoding='utf-8') as fh:
                csv.writer(fh).writerow([
                    epoch, f'{avg_loss:.6f}', f'{current_lr:.6g}',
                    f'{val_place_p:.4f}', f'{val_place_r:.4f}', f'{val_place_f1:.4f}',
                    f'{val_edge_p:.4f}', f'{val_edge_r:.4f}', f'{val_edge_f1:.4f}',
                    f'{train_place_f1:.4f}' if train_eval_every > 0 else '',
                    int(is_best), f'{time.time() - epoch_t0:.1f}'])
            prof.end_epoch(epoch)

    embedding_before = model.embedding.weight.clone().detach().cpu().numpy()
    train_model(model, train_loader, combined_train, combined_val, epochs)

    # 訓練結束，讀取歷史最佳模型進行最終評估
    model_path = os.path.join(config.MODEL_SAVE_DIR, f'trained_model_{config.EXPERIMENT_NAME}.pth')
    loaded = torch.load(model_path, weights_only=False)
    if isinstance(loaded, dict) and 'model_state_dict' in loaded:
        model.load_state_dict(loaded['model_state_dict'])
    else:
        model = loaded
    print(f"已載入最佳模型 (檔案: {model_path})")

    embedding_after = model.embedding.weight.clone().detach().cpu().numpy()
    print("權重差異:", np.abs(embedding_after - embedding_before).sum())

    # 在驗證集上選 GT 框評估的閾值（E2E 閾值另以 e2e_evaluate.py --split val 選定），寫回 checkpoint
    val_scan = []
    if val_xml_files:
        print("在驗證集上掃描 GT 框閾值...")
        for th in THRESHOLD_GRID:
            vp, vr, vf1, *_ = get_place_metrics(model, combined_val, device, val_global_text_idx, threshold=th)
            val_scan.append((th, vp, vr, vf1))
        gt_threshold, _, _, best_val_scan_f1 = max(val_scan, key=lambda r: r[3])
        print(f"驗證集選定 GT 框閾值: {gt_threshold:.2f}（驗證集 Entity F1 = {best_val_scan_f1:.4f}）")
        if isinstance(loaded, dict):
            loaded['thresholds'] = {'gt': gt_threshold, 'e2e': None}
            torch.save(loaded, model_path)
    else:
        gt_threshold = config.PRED_THRESHOLD
        print(f"沒有驗證集，沿用 config.PRED_THRESHOLD = {gt_threshold}")

    # 最終測試評估（測試集只在這裡評估一次）
    print(f"進行最佳模型的最終評估（閾值 {gt_threshold:.2f}）...")
    final_place_p, final_place_r, final_place_f1, final_edge_p, final_edge_r, final_edge_f1, tt, _ = get_place_metrics(model, combined_test, device, test_global_text_idx, threshold=gt_threshold)

    print("=====================================")
    print("最終測試結果:")
    print("地名級別評估 (Entity-level, 無 TN):")
    print(f"Precision : {final_place_p:.4f}")
    print(f"Recall    : {final_place_r:.4f}")
    print(f"F1-Score  : {final_place_f1:.4f}")
    print("---")
    print("連線級別評估 (Edge-level):")
    print(f"Edge P    : {final_edge_p:.4f}")
    print(f"Edge R    : {final_edge_r:.4f}")
    print(f"Edge F1   : {final_edge_f1:.4f}")
    print(f"測試集 GT 地名數: {tt}")
    print("=====================================")

    # 完整的可視化代碼
    def plot_metrics(place_metrics, train_losses):
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        plt.plot(range(1, len(train_losses) + 1), train_losses, color='#5DA5DA')
        plt.title('訓練損失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(range(1, len(place_metrics['train']['f1']) + 1), place_metrics['train']['f1'], '-', color='#60BD68', label='Entity 訓練集')
        plt.plot(range(1, len(place_metrics['train']['edge_f1']) + 1), place_metrics['train']['edge_f1'], '--', color='#60BD68', alpha=0.6, label='Edge 訓練集')
        val_epochs = range(1, len(place_metrics['val']['f1']) + 1)
        if len(place_metrics['val']['f1']) > 0:
            plt.plot(val_epochs, place_metrics['val']['f1'], '-', color='#FAA43A', label='Entity 驗證集')
            plt.plot(val_epochs, place_metrics['val']['edge_f1'], '--', color='#FAA43A', alpha=0.6, label='Edge 驗證集')
        plt.title('F1 分數 (Entity vs Edge)')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.plot(range(1, len(place_metrics['train']['precision']) + 1), place_metrics['train']['precision'], '-', color='#60BD68', label='Entity 訓練集')
        plt.plot(range(1, len(place_metrics['train']['edge_p']) + 1), place_metrics['train']['edge_p'], '--', color='#60BD68', alpha=0.6, label='Edge 訓練集')
        if len(place_metrics['val']['precision']) > 0:
            plt.plot(val_epochs, place_metrics['val']['precision'], '-', color='#FAA43A', label='Entity 驗證集')
            plt.plot(val_epochs, place_metrics['val']['edge_p'], '--', color='#FAA43A', alpha=0.6, label='Edge 驗證集')
        plt.title('精確率 Precision (Entity vs Edge)')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.plot(range(1, len(place_metrics['train']['recall']) + 1), place_metrics['train']['recall'], '-', color='#60BD68', label='Entity 訓練集')
        plt.plot(range(1, len(place_metrics['train']['edge_r']) + 1), place_metrics['train']['edge_r'], '--', color='#60BD68', alpha=0.6, label='Edge 訓練集')
        if len(place_metrics['val']['recall']) > 0:
            plt.plot(val_epochs, place_metrics['val']['recall'], '-', color='#FAA43A', label='Entity 驗證集')
            plt.plot(val_epochs, place_metrics['val']['edge_r'], '--', color='#FAA43A', alpha=0.6, label='Edge 驗證集')
        plt.title('召回率 Recall (Entity vs Edge)')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        metrics_path = os.path.join(config.OUTPUT_DIR, f'place_name_metrics_{config.EXPERIMENT_NAME}.png')
        plt.savefig(metrics_path)
        plt.close()

    plot_metrics(place_metrics, train_losses)
    print("訓練完成，地名識別指標圖已保存")

    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = f"{int(hours)}小時 {int(minutes)}分鐘 {seconds:.2f}秒"

    # 將最終成績存為文字檔
    report_path = os.path.join(config.OUTPUT_DIR, f'final_metrics_{config.EXPERIMENT_NAME}.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"=== 最終模型測試結果 ({config.EXPERIMENT_NAME}) ===\n")
        f.write(f"總訓練時間: {time_str}\n\n")
        f.write("[評估協定]\n")
        if val_xml_files:
            f.write(f"驗證集 {len(val_xml_files)} 張（取自訓練切分）: {', '.join(os.path.splitext(v)[0] for v in val_xml_files)}\n")
            f.write(f"最佳 checkpoint: epoch {loaded.get('best_epoch', '?') if isinstance(loaded, dict) else '?'}（以驗證集 Entity F1 選定，閾值 {config.PRED_THRESHOLD}）\n")
            f.write(f"GT 框閾值: {gt_threshold:.2f}（驗證集掃描選定，測試集只評估一次）\n")
            f.write("E2E 閾值: 尚未選定，請在本機執行 e2e_evaluate.py --split val --scan-threshold --save-threshold\n")
            f.write(f"逐 epoch 紀錄: history_{config.EXPERIMENT_NAME}.csv\n\n")
            f.write("[驗證集 GT 框閾值掃描]\n")
            for th, vp, vr, vf1 in val_scan:
                f.write(f"  {th:.2f} | P {vp:.4f}  R {vr:.4f}  F1 {vf1:.4f}\n")
            f.write("\n")
        else:
            f.write(f"舊協定（無驗證集）：以測試集選 checkpoint，閾值 {gt_threshold}\n\n")
        f.write("[實體級別 Entity-Level]\n")
        f.write(f"Precision : {final_place_p:.4f}\n")
        f.write(f"Recall    : {final_place_r:.4f}\n")
        f.write(f"F1-Score  : {final_place_f1:.4f}\n")
        f.write(f"測試集 GT 地名數: {tt}\n\n")
        
        f.write("[連線級別 Edge-Level]\n")
        f.write(f"Edge P    : {final_edge_p:.4f}\n")
        f.write(f"Edge R    : {final_edge_r:.4f}\n")
        f.write(f"Edge F1   : {final_edge_f1:.4f}\n")
        
        f.write("\n=== 訓練時設定的超參數 ===\n")
        
        # 讀取 config.py 原始碼，過濾掉一般程式碼，僅保留有標題註解和變數的部分
        config_py_path = os.path.join(os.path.dirname(__file__), 'config.py')
        if os.path.exists(config_py_path):
            with open(config_py_path, 'r', encoding='utf-8') as cfg_f:
                for line in cfg_f:
                    line_stripped = line.strip()
                    if line_stripped.startswith('# ---') or line_stripped.startswith('# ==="'):
                        f.write(f"\n{line_stripped}\n")
                    elif '=' in line_stripped and not line_stripped.startswith('#'):
                        var_name = line_stripped.split('=')[0].strip()
                        if hasattr(config, var_name) and not var_name.startswith('__') and 'KEY' not in var_name.upper():
                            f.write(f"{var_name} = {getattr(config, var_name)}\n")

    print(f"最終模型成績已儲存至: {report_path}")
    print("=====================================")
    print(f"程式總運行時間: {int(hours)}小時 {int(minutes)}分鐘 {seconds:.2f}秒")
    print("=====================================")

    end_time = time.time()

    # ==================== 視覺化每張測試地圖的預測結果（VISUALIZE_TEST=1 才執行）====================
    if getattr(config, 'VISUALIZE_TEST', False):
        print("\n開始視覺化測試集地圖預測結果...")
        visualize_all_maps(
            model, combined_test, test_map_infos,
            threshold=gt_threshold,
        )
        print("所有測試集地圖的預測結果已視覺化完成！")

        visualize_time = time.time() - end_time
        hours, remainder = divmod(visualize_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print("=====================================")
        print(f"視覺化時間: {int(hours)}小時 {int(minutes)}分鐘 {seconds:.2f}秒")
        print("=====================================")
    else:
        print("略過測試集預測圖（需要時設 VISUALIZE_TEST=1）")
