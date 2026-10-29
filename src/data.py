import xml.etree.ElementTree as ET
import torch
from torch_geometric.data import Data
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from gensim.models import KeyedVectors

# 資料根目錄：環境變數 DATA_DIR 優先；否則為本 repo 的上一層資料夾
# （資料與 repo 並排放置，換機器、換帳號都不必改程式）
DATA_DIR = os.environ.get("DATA_DIR") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

# 全局變量
scaler = None
# edge_scaler：候選邊 (edge_label_attr) 的正規化器。
#              fit 方式維持原樣（以 id 鏈的邊特徵 fit），因此在所有 MP 模式間
#              完全一致 —— LinkPredictor 的輸入正規化不會隨 MP 策略改變。
edge_scaler = None
# mp_scaler：訊息傳遞邊 (edge_attr) 的正規化器，依 config.MP_EDGE_MODE
#            在該模式自己的邊分布上 fit，讓每種幾何法都拿到公平的正規化。
#            MP_EDGE_MODE="id" 時與 edge_scaler fit 在相同資料上，
#            故可位元級重現改動前的行為。
mp_scaler = None
# 三個 scaler 是否已由 checkpoint 還原。是的話 fit_global_scaler 不再重新擬合，
# 評估端才會用到與訓練時完全相同的正規化參數（訓練時只用訓練圖擬合）。
_scalers_from_checkpoint = False
_scaler_fit_files = []

# ----------------- 詞彙表相關 -----------------
vocab = {}
vocab_count = 0

def read_xml_from(filename, xml_dir):
    """
    從指定目錄讀取 XML，並修復 OCR XML 常見的 ISO-8859-1 編碼宣告問題。
    這是 read_xml 的通用版本，讓 train.py 能分別從
    TRAIN_XML_DIR 和 TEST_XML_DIR 讀取不同資料集。
    """
    import re as _re
    filepath = os.path.join(xml_dir, filename)
    with open(filepath, 'rb') as f:
        raw = f.read()
    raw = _re.sub(rb"encoding=['\"]ISO-8859-1['\"]", b"encoding='UTF-8'", raw)
    raw = _re.sub(rb"encoding=['\"]iso-8859-1['\"]", b"encoding='UTF-8'", raw)
    return ET.fromstring(raw.decode('utf-8', errors='replace'))


def read_xml(filename):
    """
    讀取 XML 檔案，路徑由 config.XML_DIR 決定（支援 xml_revised 或 xml_ocr_aligned）。
    同時防禦性地修復 OCR XML 常見的 ISO-8859-1 宣告但內含中文的編碼問題。
    """
    import re as _re
    from config import XML_DIR
    filepath = os.path.join(XML_DIR, filename)
    with open(filepath, 'rb') as f:
        raw = f.read()
    # 修正可能的錯誤 encoding 宣告（OCR XML 原始檔常見問題）
    raw = _re.sub(rb"encoding=['\"]ISO-8859-1['\"]", b"encoding='UTF-8'", raw)
    raw = _re.sub(rb"encoding=['\"]iso-8859-1['\"]", b"encoding='UTF-8'", raw)
    root = ET.fromstring(raw.decode('utf-8', errors='replace'))
    return root


# ==================== 訊息傳遞邊 (MP 邊) 建構 ====================
# 設計要點：
#   * 標籤 (place_name_edge) 永遠來自 id 鏈，不受 MP 模式影響。
#   * 拓樸座標由 config.MP_ANCHOR 決定（"tl" / "center"）。
#   * 邊特徵值一律以左上角 (x, y) 計算，與原實作一致，
#     使「拓樸」成為唯一變動的變數。
#   * 各模式回傳「無向配對集合」；是否補反向邊由 config.BIDIRECTIONAL_GNN 決定，
#     補法與原實作相同（正向、反向交錯排列）。

VALID_MP_MODES = ("id", "none", "empty", "delaunay", "gabriel", "knn", "dirnn")


def _mp_topology_coords(ocr_results, anchor):
    """拓樸判定用的座標。anchor="center" 時用框中心，否則用左上角。"""
    if anchor == "center":
        return np.array([[o["x"] + o["w"] / 2.0, o["y"] + o["h"] / 2.0]
                         for o in ocr_results], dtype=np.float64)
    return np.array([[o["x"], o["y"]] for o in ocr_results], dtype=np.float64)


def _mp_pairs_delaunay(coords):
    """Delaunay 三角化的邊集。少於 3 點或退化時退回全連接。"""
    from itertools import combinations
    n = len(coords)
    if n < 3:
        return {(i, j) for i in range(n) for j in range(i + 1, n)}
    try:
        from scipy.spatial import Delaunay
    except ImportError as exc:
        raise ImportError(
            "MP_EDGE_MODE=delaunay/gabriel 需要 scipy（scipy.spatial.Delaunay）。"
            "請在執行環境安裝 scipy 後重試。"
        ) from exc
    try:
        tri = Delaunay(coords, qhull_options="QJ")  # QJ: joggle，避免共線退化
    except Exception:
        # 全部共線等極端情況：退回全連接，避免整張圖失去邊
        return {(i, j) for i in range(n) for j in range(i + 1, n)}
    E = set()
    for s in tri.simplices:
        for a, b in combinations(s, 2):
            E.add((min(int(a), int(b)), max(int(a), int(b))))
    return E


def _mp_pairs_gabriel(coords):
    """
    Gabriel graph：邊 (p,q) 保留 iff 以 pq 為直徑的圓內沒有第三點。
    Gabriel ⊆ Delaunay，等價於 β=1 的 β-skeleton（FormNet / GVdoc 採用）。
    """
    dela = _mp_pairs_delaunay(coords)
    P = np.asarray(coords, dtype=np.float64)
    E = set()
    for i, j in dela:
        m = (P[i] + P[j]) / 2.0
        r2 = ((P[i] - P[j]) ** 2).sum() / 4.0
        d2 = ((P - m) ** 2).sum(axis=1)
        d2[i] = d2[j] = np.inf
        if (d2 >= r2 - 1e-9).all():
            E.add((i, j))
    return E


def _mp_pairs_knn(coords, k):
    """每節點連向最近的 k 個鄰居，並對稱化（union）。"""
    P = np.asarray(coords, dtype=np.float64)
    n = len(P)
    if n < 2:
        return set()
    diff = P[:, None, :] - P[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    kk = min(int(k), n - 1)
    nn = np.argsort(dist, axis=1)[:, :kk]
    E = set()
    for i in range(n):
        for j in nn[i]:
            j = int(j)
            E.add((min(i, j), max(i, j)))
    return E


def _mp_pairs_dirnn(coords):
    """
    上/下/左/右四個方向各取最近鄰（GLAM 式異向性建圖）。
    以 |dx| 與 |dy| 誰較大決定該候選落在水平或垂直扇區；
    影像座標 y 向下遞增。
    """
    P = np.asarray(coords, dtype=np.float64)
    n = len(P)
    if n < 2:
        return set()
    E = set()
    for i in range(n):
        dx = P[:, 0] - P[i, 0]
        dy = P[:, 1] - P[i, 1]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        dist[i] = np.inf
        horiz = np.abs(dx) >= np.abs(dy)
        sectors = (
            horiz & (dx > 0),    # 右
            horiz & (dx < 0),    # 左
            ~horiz & (dy > 0),   # 下
            ~horiz & (dy < 0),   # 上
        )
        for sec in sectors:
            cand = np.where(sec & np.isfinite(dist))[0]
            if cand.size == 0:
                continue
            j = int(cand[np.argmin(dist[cand])])
            E.add((min(i, j), max(i, j)))
    return E


# ==================== 邊幾何特徵 ====================
# MP 邊 (edge_attr) 與候選邊 (edge_label_attr) 共用同一套公式，
# 保證兩者永遠不會分歧。
#
# 基礎維（依 config.EDGE_ANGLE_MODE）：
#   0. distance   : 左上角歐氏距離
#   1. delta_x    : dst.x - src.x
#   2. delta_y    : dst.y - src.y
#   3. angle      : arctan2(dy, dx)                      ← EDGE_ANGLE_MODE="atan2"（舊）
#   3. cos, 4. sin: 方向的單位向量；dist=0 時為 (0, 0)   ← EDGE_ANGLE_MODE="cossin"（預設）
#      atan2 的不連續點 ±π 剛好落在最常見的「右→左橫書」方向上，模型學會忽略它；
#      cos/sin 連續且有界。
#
# 追加維（皆對 (src, dst) 交換對稱；欄位順序固定為 _EXTRA_FEATURE_ORDER）：
#   rel_dist   : distance / min(src 最近鄰距, dst 最近鄰距)，剪至 20
#                同一地名的相鄰字互為最近鄰 -> ~1；分屬兩個地名的字各自有更近的夥伴 -> >1。
#   size_ratio : max(字級) / min(字級)，字級 = sqrt(w*h)，剪至 10
#                取 sqrt(w*h) 而非單看高度，使其不受直書/橫書影響。
#   align_ovl  : max(x 軸投影重疊率, y 軸投影重疊率)，值域 [0, 1]
#                同一地名的字沿書寫軸排列，在垂直於該軸的方向高度重疊。
#   lin_any    : 兩端任一字屬 config.LINEAR_FEATURE_CHARS（河/路/鉄…）-> 1，否則 0
#                線性地物的字沿線分散、字距大，是分距離段篩選中唯一有「救回方向」訊號的特徵。
#   size_rel   : 兩端平均字級 / 該圖字級中位數，剪至 10
#                與 size_ratio 不同：size_ratio 比兩端，size_rel 比「這對字和整張圖」。
#   dist_norm  : distance / 兩端平均字級，剪至 20
#                以字寬為單位的距離；大字 300px = 3 字寬，可借用小字在 3 字寬學到的東西。
#   bigram_lp  : max(log P(cj|ci), log P(ci|cj))，剪至 [-12, 0]
#                由訓練切分的 GT 地名學得的字元接續機率（add-k 平滑）。
#                亂連的字對是「地名裡不會出現的組合」（路–歐、縣–撫），AUC 0.71；
#                對該連沒連的字對沒有作用（0.54），故只往「不亂連」的方向推。
#                訓練圖以留一圖法扣掉自己的地名（否則每條正邊都查得到，模型會背答案）；
#                表存進 checkpoint，評估時直接載入，測試切分的地名絕不進表。
#                兩端任一字不在表裡 -> 給「中性值」（正邊平均與隨機字對平均的中點）。
#   gap_norm   : 框與框的最短間距 / 兩端平均字級，重疊 = 0，剪至 20
#                GVdoc / FormNet 的框間距；字級不同時中心距會誤導，框間距不會。
#   rank_max   : dst 在 src 的距離排序中的名次與反向名次取較大者（0 = 最近鄰），剪至 30
#                亂連的邊中位名次 1，正確的 0。
#
# 前面的距離類特徵一律以左上角 (x, y) 計算，與改動前完全一致；rel_dist / rank_max
# 的最近鄰同樣用左上角距離。align_ovl / gap_norm 使用真實框範圍，不受 MP_ANCHOR 影響。
#
# 追加特徵由 config.EXTRA_EDGE_FEATURES 逐項開關，特徵消融不需要改任何程式碼。

_REL_DIST_CLIP = 20.0    # 超過局部間距 20 倍一律視為「絕不相鄰」，避免離群值壓垮 StandardScaler
_SIZE_RATIO_CLIP = 10.0
_SIZE_REL_CLIP = 10.0
_DIST_NORM_CLIP = 20.0
_GAP_NORM_CLIP = 20.0
_RANK_CLIP = 30.0
_BIGRAM_LP_MIN = -12.0


# 追加特徵的正規順序。實際啟用哪些由 config.EXTRA_EDGE_FEATURES 決定，
# 但輸出的欄位順序永遠依此排列，與設定的書寫順序無關 —— 這樣
# checkpoint 裡的權重才不會因為設定字串的順序不同而錯位。
# 只能在尾端追加，不能重排，否則舊 checkpoint 的欄位會錯位。
_EXTRA_FEATURE_ORDER = ("rel_dist", "size_ratio", "align_ovl",
                        "lin_any", "size_rel", "dist_norm",
                        "bigram_lp", "gap_norm", "rank_max")


def _extra_edge_features():
    """回傳本次啟用的追加特徵名稱（依 _EXTRA_FEATURE_ORDER 排序）。"""
    import config
    sel = getattr(config, "EXTRA_EDGE_FEATURES", None)
    if sel is None:
        # 沒有這個設定的舊 checkpoint：退回布林總開關
        return _EXTRA_FEATURE_ORDER[:3] if getattr(config, "USE_EXTRA_EDGE_FEATURES", False) else ()
    if isinstance(sel, str):
        sel = sel.split(",")
    sel = {str(x).strip() for x in sel if str(x).strip()}
    unknown = sel - set(_EXTRA_FEATURE_ORDER)
    if unknown:
        raise ValueError(
            f"EXTRA_EDGE_FEATURES 含未知特徵 {sorted(unknown)}；"
            f"可用: {', '.join(_EXTRA_FEATURE_ORDER)}")
    return tuple(f for f in _EXTRA_FEATURE_ORDER if f in sel)


def _angle_mode():
    import config
    return str(getattr(config, "EDGE_ANGLE_MODE", "atan2")).strip()


def base_edge_feature_names():
    """基礎維的名稱（依 EDGE_ANGLE_MODE）。"""
    if _angle_mode() == "cossin":
        return ("distance", "delta_x", "delta_y", "cos", "sin")
    return ("distance", "delta_x", "delta_y", "angle")


def edge_feature_names():
    return base_edge_feature_names() + _extra_edge_features()


def edge_feature_dim():
    """目前設定下的邊特徵維度；供建立空張量與自我檢查使用。"""
    return len(base_edge_feature_names()) + len(_extra_edge_features())


# ----------------- 字元接續先驗（bigram_lp 用） -----------------
class TextPrior:
    """
    由 GT 地名學得的字元 bigram 表，供 bigram_lp 特徵查詢。

    洩漏防護（兩層）：
      1. 只用訓練切分的檔案建表（train.py 在分割之後才呼叫 build_text_prior），
         測試切分 / e2e 的地名絕不進表。
      2. 每個訓練檔各自的貢獻另外記錄（per_file）。計算某張訓練圖的特徵時，
         先扣掉它自己的貢獻（留一圖法），所以模型看不到「這條正邊在表裡」這種答案，
         只看得到從其他 152 張圖學來的通用接續規律 —— 和一個小型語言模型一樣。
         扣除以 map_id 為鍵，與 ocr_results 是否被 node dropout 過濾無關。
    表以 state_dict() 存進 checkpoint，評估端 from_state() 還原，不重算。
    """

    def __init__(self, k=0.5):
        self.k = float(k)
        self.chars = []            # idx -> 字
        self.char2idx = {}
        self.C = None              # (V, V) int32：C[a, b] = a 後接 b 的次數
        self.per_file = {}         # map_id -> {(a_idx, b_idx): n}
        self.n_names = 0
        self.neutral = 0.0         # 未知字對的中性值
        self._pos_mean = 0.0
        self._rand_mean = 0.0

    # ---- 建表 ----
    @staticmethod
    def chains_from_root(root):
        """從 GT XML 依 id 分組取出每個地名的字序列（順序 = XML 列出順序，與 build_graph 一致）。"""
        groups = {}
        for image in root.findall(".//image"):
            for box in image.findall("box"):
                pid = box.get("id")
                if pid == "0":
                    continue
                lab = box.find("label")
                text = (lab.text or "").strip() if lab is not None else ""
                groups.setdefault(pid, []).append(text)
        return list(groups.values())

    def fit(self, xml_files, xml_dir):
        pairs_by_file = {}
        for f in xml_files:
            root = read_xml_from(f, xml_dir)
            map_id = os.path.splitext(f)[0]
            pairs = []
            for chain in self.chains_from_root(root):
                self.n_names += 1
                for a, b in zip(chain, chain[1:]):
                    pairs.append((a, b))
            pairs_by_file[map_id] = pairs
        vocab = sorted({c for ps in pairs_by_file.values() for ab in ps for c in ab})
        self.chars = vocab
        self.char2idx = {c: i for i, c in enumerate(vocab)}
        V = len(vocab)
        self.C = np.zeros((V, V), dtype=np.int32)
        self.per_file = {}
        for map_id, pairs in pairs_by_file.items():
            cnt = {}
            for a, b in pairs:
                key = (self.char2idx[a], self.char2idx[b])
                cnt[key] = cnt.get(key, 0) + 1
                self.C[key] += 1
            self.per_file[map_id] = cnt
        self._finalize()
        return self

    def _finalize(self):
        V = self.C.shape[0]
        k = self.k
        row = self.C.sum(axis=1).astype(np.float64)
        logp = np.log((self.C + k) / (row[:, None] + k * V))
        w = self.C.astype(np.float64)
        self._pos_mean = float((logp * w).sum() / max(w.sum(), 1.0))   # 正邊（依出現次數加權）平均
        self._rand_mean = float(logp.mean())                            # 隨機字對平均
        self.neutral = 0.5 * (self._pos_mean + self._rand_mean)

    # ---- 查詢 ----
    def tables_for(self, map_id):
        """回傳 (C, row) —— 若 map_id 是建表檔之一，先扣掉它自己的貢獻。"""
        C, row = self.C, self.C.sum(axis=1).astype(np.float64)
        own = self.per_file.get(map_id)
        if own:
            C = C.copy()
            row = row.copy()
            for (a, b), n in own.items():
                C[a, b] -= n
                row[a] -= n
        return C, row

    def logp(self, a_idx, b_idx, tables):
        """向量化 log P(b|a)；a_idx/b_idx 為 -1 代表未知字 -> 中性值。"""
        C, row = tables
        V = C.shape[0]
        a = np.asarray(a_idx, dtype=np.int64)
        b = np.asarray(b_idx, dtype=np.int64)
        known = (a >= 0) & (b >= 0)
        out = np.full(a.shape, self.neutral, dtype=np.float64)
        if known.any():
            aa, bb = a[known], b[known]
            out[known] = np.log((C[aa, bb] + self.k) / (row[aa] + self.k * V))
        return out

    def index_of(self, texts):
        return np.array([self.char2idx.get(t, -1) for t in texts], dtype=np.int64)

    # ---- 序列化 ----
    def state_dict(self):
        return {"k": self.k, "chars": list(self.chars),
                "C": self.C.copy(),
                "per_file": {m: {f"{a},{b}": n for (a, b), n in cnt.items()} for m, cnt in self.per_file.items()},
                "n_names": self.n_names}

    @classmethod
    def from_state(cls, st):
        tp = cls(k=st["k"])
        tp.chars = list(st["chars"])
        tp.char2idx = {c: i for i, c in enumerate(tp.chars)}
        tp.C = np.asarray(st["C"], dtype=np.int32)
        tp.per_file = {m: {tuple(int(x) for x in key.split(",")): int(n) for key, n in cnt.items()}
                       for m, cnt in st["per_file"].items()}
        tp.n_names = int(st.get("n_names", 0))
        tp._finalize()
        return tp

    def summary(self):
        V = len(self.chars)
        return (f"bigram 表：{len(self.per_file)} 張圖、{self.n_names} 個地名、字種 {V}、"
                f"bigram 種類 {int((self.C > 0).sum())}；中性值 {self.neutral:.2f}"
                f"（正邊平均 {self._pos_mean:.2f} / 隨機字對平均 {self._rand_mean:.2f}）")


text_prior = None   # TextPrior 或 None；train.py 建表、評估端由 checkpoint 還原


def build_text_prior(xml_files, xml_dir):
    """用訓練切分的檔案建 bigram 表並設為全域 text_prior。"""
    global text_prior
    import config
    text_prior = TextPrior(k=getattr(config, "BIGRAM_SMOOTHING", 0.5)).fit(xml_files, xml_dir)
    return text_prior


class _NodeGeom:
    """
    每張圖只需算一次的節點層級中間量（供 _edge_geom_features 使用）。

      xy       : (N, 2) 左上角座標
      wh       : (N, 2) 寬高
      size     : (N,)   字級 = sqrt(w*h)
      nn       : (N,)   到最近鄰節點的左上角距離（下限 1.0 px）
      size_med : 該圖字級中位數
      lin      : (N,)   是否為線性地物通名字（lin_any 用）
      map_id   : 這張圖的識別字串（bigram 留一圖法用；沒有則 None）
    以下為延遲計算：
      rank     : (N, N) 依左上角距離的鄰居名次（rank_max 用）
      char_idx / bigram_tables（bigram_lp 用）
    """

    def __init__(self, ocr_results):
        import config
        self.ocr_results = ocr_results
        xy = np.array([[o["x"], o["y"]] for o in ocr_results], dtype=np.float64)
        wh = np.array([[o["w"], o["h"]] for o in ocr_results], dtype=np.float64)
        if len(xy) == 0:
            xy, wh = xy.reshape(0, 2), wh.reshape(0, 2)
        self.xy, self.wh = xy, wh
        self.size = np.sqrt(np.maximum(wh[:, 0], 1.0) * np.maximum(wh[:, 1], 1.0))
        n = len(xy)
        if n < 2:
            nn = np.ones(n, dtype=np.float64)
        else:
            try:
                from scipy.spatial import cKDTree
                # k=2：第一個鄰居是自己，第二個才是最近的其他點
                d, _ = cKDTree(xy).query(xy, k=2)
                nn = np.asarray(d)[:, 1]
            except ImportError:
                d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
                np.fill_diagonal(d, np.inf)
                nn = d.min(axis=1)
            nn = np.maximum(nn, 1.0)   # 完全重疊的框距離為 0，箝到 1 px
        self.nn = nn
        self.size_med = float(np.median(self.size)) if n else 1.0
        lin_chars = set(getattr(config, "LINEAR_FEATURE_CHARS", ""))
        self.texts = [str(o.get("text", "")) for o in ocr_results]
        self.lin = np.array([any(ch in lin_chars for ch in t) for t in self.texts], dtype=np.float64)
        self.map_id = ocr_results[0].get("map_id") if n else None
        self._rank = None
        self._char_idx = None
        self._bigram_tables = None

    # 舊介面：允許 xy, wh, size, nn = ng
    def __iter__(self):
        return iter((self.xy, self.wh, self.size, self.nn))

    @property
    def rank(self):
        if self._rank is None:
            n = len(self.xy)
            if n < 2:
                self._rank = np.zeros((n, n), dtype=np.int32)
            else:
                d = np.linalg.norm(self.xy[:, None, :] - self.xy[None, :, :], axis=-1)
                np.fill_diagonal(d, np.inf)             # 自己排最後，不佔名次
                self._rank = np.argsort(np.argsort(d, axis=1, kind="stable"), axis=1).astype(np.int32)
        return self._rank

    @property
    def char_idx(self):
        if self._char_idx is None:
            self._char_idx = text_prior.index_of(self.texts)
        return self._char_idx

    @property
    def bigram_tables(self):
        if self._bigram_tables is None:
            self._bigram_tables = text_prior.tables_for(self.map_id)
        return self._bigram_tables


def _node_geom(ocr_results):
    """每張圖只需算一次的節點層級中間量；回傳 _NodeGeom（可解包成 xy, wh, size, nn）。"""
    return _NodeGeom(ocr_results)


def _edge_geom_features(ocr_results, src, dst, ng=None):
    """
    向量化計算一批邊的「未正規化」特徵，回傳 (E, edge_feature_dim())。

    Args:
        src, dst: 節點索引的一維陣列（list 亦可）
        ng      : _node_geom 的回傳值；同一張圖多次呼叫時傳入可免除重算
    """
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    if ng is None:
        ng = _node_geom(ocr_results)
    xy, wh, size, nn = ng.xy, ng.wh, ng.size, ng.nn

    dx = xy[dst, 0] - xy[src, 0]
    dy = xy[dst, 1] - xy[src, 1]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    if _angle_mode() == "cossin":
        safe = np.where(dist > 0, dist, 1.0)
        cos = np.where(dist > 0, dx / safe, 0.0)
        sin = np.where(dist > 0, dy / safe, 0.0)
        cols = [dist, dx, dy, cos, sin]
    else:
        angle = np.arctan2(dy, dx)
        angle[dist == 0] = 0.0
        cols = [dist, dx, dy, angle]
    extras = _extra_edge_features()
    if not extras:
        return np.stack(cols, axis=1)

    msize = (size[src] + size[dst]) / 2.0

    def _ovl(lo_a, len_a, lo_b, len_b):
        """兩個區間的重疊長度 / 較短者長度，值域 [0, 1]。"""
        inter = np.minimum(lo_a + len_a, lo_b + len_b) - np.maximum(lo_a, lo_b)
        denom = np.maximum(np.minimum(len_a, len_b), 1e-6)
        return np.clip(inter / denom, 0.0, 1.0)

    def _rel_dist():
        # 分母取兩端最近鄰距的「較小者」
        # -> 只要任一端有更近的夥伴，比值就會拉大（等價於 max(d/nn_src, d/nn_dst)）
        return np.clip(dist / np.minimum(nn[src], nn[dst]), 0.0, _REL_DIST_CLIP)

    def _size_ratio():
        s_max = np.maximum(size[src], size[dst])
        s_min = np.maximum(np.minimum(size[src], size[dst]), 1e-6)
        return np.clip(s_max / s_min, 1.0, _SIZE_RATIO_CLIP)   # 永遠 >= 1

    def _align_ovl():
        # 兩軸各算一次取較大者。分母用較短邊，
        # 使「小字完全落在大字的投影範圍內」也能得到 1.0。
        return np.maximum(
            _ovl(xy[src, 0], wh[src, 0], xy[dst, 0], wh[dst, 0]),
            _ovl(xy[src, 1], wh[src, 1], xy[dst, 1], wh[dst, 1]),
        )

    def _lin_any():
        return np.maximum(ng.lin[src], ng.lin[dst])

    def _size_rel():
        return np.clip(msize / max(ng.size_med, 1e-6), 0.0, _SIZE_REL_CLIP)

    def _dist_norm():
        return np.clip(dist / np.maximum(msize, 1e-6), 0.0, _DIST_NORM_CLIP)

    def _bigram_lp():
        if text_prior is None:
            raise RuntimeError(
                "EXTRA_EDGE_FEATURES 含 bigram_lp，但 data.text_prior 尚未建立：\n"
                "  訓練端請在分割後呼叫 build_text_prior(train_files, TRAIN_XML_DIR)；\n"
                "  評估端請以 restore_checkpoint_config() 從 checkpoint 還原；\n"
                "  不需要這個特徵的腳本請把它從 EXTRA_EDGE_FEATURES 拿掉。")
        ci, tabs = ng.char_idx, ng.bigram_tables
        fwd = text_prior.logp(ci[src], ci[dst], tabs)
        bwd = text_prior.logp(ci[dst], ci[src], tabs)
        return np.clip(np.maximum(fwd, bwd), _BIGRAM_LP_MIN, 0.0)

    def _gap_norm():
        gx = np.maximum(xy[src, 0], xy[dst, 0]) - np.minimum(xy[src, 0] + wh[src, 0], xy[dst, 0] + wh[dst, 0])
        gy = np.maximum(xy[src, 1], xy[dst, 1]) - np.minimum(xy[src, 1] + wh[src, 1], xy[dst, 1] + wh[dst, 1])
        gap = np.hypot(np.maximum(gx, 0.0), np.maximum(gy, 0.0))    # 任一軸重疊 -> 該軸 0；兩軸皆重疊 -> 0
        return np.clip(gap / np.maximum(msize, 1e-6), 0.0, _GAP_NORM_CLIP)

    def _rank_max():
        r = ng.rank
        return np.clip(np.maximum(r[src, dst], r[dst, src]).astype(np.float64), 0.0, _RANK_CLIP)

    _builders = {"rel_dist": _rel_dist, "size_ratio": _size_ratio, "align_ovl": _align_ovl,
                 "lin_any": _lin_any, "size_rel": _size_rel, "dist_norm": _dist_norm,
                 "bigram_lp": _bigram_lp, "gap_norm": _gap_norm, "rank_max": _rank_max}
    return np.stack(cols + [_builders[f]() for f in extras], axis=1)


# ----------------- 資料切分（訓練 / 驗證 / 測試，各腳本共用） -----------------
# GT 框與 E2E 的閾值掃描共用同一組候選值：0.10 ~ 0.95，間隔 0.05
THRESHOLD_GRID = [i / 100.0 for i in range(10, 100, 5)]


def split_maps(xml_files, cfg):
    """
    回傳 (train, val, test) 三份地圖檔名清單。

    train / test 與改動前完全相同：sorted 後取前 MAP_NUM 張，以全域 random.seed(RANDOM_SEED)
    打亂，前 TRAIN_RATIO 為訓練、其餘為測試（會改動全域 random 狀態，與改動前一致）。
    驗證集從訓練切分中抽出，只抽有原始 OCR 輸出的地圖（VAL_OCR_DIR），驗證集才能做 E2E：
      - cfg.VAL_FILES 非空（評估端由 checkpoint 還原）→ 直接沿用，不重抽
      - 否則以獨立的 random.Random(VAL_SEED) 抽 VAL_NUM 張；VAL_NUM=0 時不切驗證集
    """
    import random
    random.seed(cfg.RANDOM_SEED)
    shuffled = sorted(xml_files)[:cfg.MAP_NUM]
    random.shuffle(shuffled)
    n_train = int(len(shuffled) * cfg.TRAIN_RATIO)
    train, test = shuffled[:n_train], shuffled[n_train:]

    val = list(getattr(cfg, 'VAL_FILES', ()) or ())
    if not val and getattr(cfg, 'VAL_NUM', 0) > 0:
        if not os.path.isdir(cfg.VAL_OCR_DIR):
            raise FileNotFoundError(
                f"驗證集需要訓練切分的原始 OCR 輸出，找不到資料夾：{cfg.VAL_OCR_DIR}\n"
                f"請把本機的 xml_ocr_output_train/ocr 同步到這台機器，或設 VAL_NUM=0 退回舊協定。")
        ocr_available = set(os.listdir(cfg.VAL_OCR_DIR))
        candidates = sorted(f for f in train if f in ocr_available and f not in cfg.SKIP_FILES)
        if len(candidates) < cfg.VAL_NUM:
            raise ValueError(f"有 OCR 的訓練地圖只有 {len(candidates)} 張，不足 VAL_NUM={cfg.VAL_NUM}")
        val = sorted(random.Random(cfg.VAL_SEED).sample(candidates, cfg.VAL_NUM))
    missing = [f for f in val if f not in train]
    if missing:
        raise ValueError(f"驗證集地圖不在訓練切分中：{missing}")
    val_set = set(val)
    train = [f for f in train if f not in val_set]
    assert not (set(train) & set(test) or val_set & set(test)), "訓練、驗證、測試集有重疊的地圖"
    return train, val, test


def resolve_threshold(loaded, kind):
    """
    決定評估用的閾值，回傳 (threshold, 來源說明)。kind 為 'gt'（GT 框評估）或 'e2e'。
    優先序：環境變數 PRED_THRESHOLD > checkpoint 中在驗證集選定的閾值 > config 預設值。
    """
    import config
    if 'PRED_THRESHOLD' in os.environ:
        return float(os.environ['PRED_THRESHOLD']), "環境變數 PRED_THRESHOLD"
    saved = (loaded.get('thresholds') or {}) if isinstance(loaded, dict) else {}
    if saved.get(kind) is not None:
        return float(saved[kind]), f"checkpoint（驗證集選定的 {kind} 閾值）"
    return float(config.PRED_THRESHOLD), f"config 預設值（checkpoint 中沒有驗證集選定的 {kind} 閾值）"


# ----------------- checkpoint 還原（評估端共用） -----------------
CHECKPOINT_CRITICAL_KEYS = (
    'IN_CHANNELS', 'EDGE_FEATURE_DIM', 'USE_EXTRA_EDGE_FEATURES', 'EXTRA_EDGE_FEATURES',
    'EDGE_ANGLE_MODE', 'LINEAR_FEATURE_CHARS', 'HIDDEN_CHANNELS', 'OUT_CHANNELS',
    'BIDIRECTIONAL_GNN', 'TRAIN_SAMPLE_MODE', 'MAP_NUM', 'RANDOM_SEED', 'TRAIN_RATIO',
    'GNN_LAYERS', 'VAL_NUM', 'VAL_SEED', 'VAL_FILES', 'USE_TEXT_EMBED',
    'GNN_TYPE', 'GNN_EDGE_IN_ATTN', 'GNN_EDGE_IN_PRED',
)

# 建圖相關、評估時也常用環境變數指定的參數：環境變數有明確設定時以它為準（例如
# ablate_mp_edges.py 會故意換 MP 模式），沒設時才從 checkpoint 還原。
# 以前這幾個不還原，評估時忘記設環境變數就會靜默地用錯建圖方式。
CHECKPOINT_ENV_OVERRIDABLE_KEYS = ('MP_EDGE_MODE', 'MP_KNN_K', 'MP_ANCHOR', 'NXN_EDGE_DISTANCE_LIMIT')


def restore_checkpoint_config(loaded, config):
    """
    把 checkpoint 封存的關鍵架構 / 資料切分參數寫回 config，並還原 bigram 表。
    evaluate.py / e2e_evaluate.py 與各分析腳本共用，避免五份複製的還原邏輯彼此脫節。

    向後相容規則：
      - 舊 checkpoint（4 維邊特徵時代）沒存 EXTRA_EDGE_FEATURES：由 EDGE_FEATURE_DIM==4 推回空集合
      - 沒存 EDGE_ANGLE_MODE 的 checkpoint 一律是 atan2 時代
      - FULL_GRAPH_INFERENCE -> TRAIN_SAMPLE_MODE
    回傳 saved_config（dict）或 None（不是含 config 的 checkpoint）。
    """
    global text_prior, scaler, edge_scaler, mp_scaler, _scalers_from_checkpoint
    _scalers_from_checkpoint = False
    if not (isinstance(loaded, dict) and 'config' in loaded):
        return None
    saved = loaded['config']
    for k in CHECKPOINT_CRITICAL_KEYS:
        if k in saved:
            setattr(config, k, saved[k])
    if 'EXTRA_EDGE_FEATURES' not in saved and saved.get('EDGE_FEATURE_DIM') == 4:
        config.EXTRA_EDGE_FEATURES = ()
    if 'EDGE_ANGLE_MODE' not in saved:
        config.EDGE_ANGLE_MODE = "atan2"
    if 'FULL_GRAPH_INFERENCE' in saved and 'TRAIN_SAMPLE_MODE' not in saved:
        config.TRAIN_SAMPLE_MODE = "NxN" if saved['FULL_GRAPH_INFERENCE'] else "1:1"
    # 驗證集協定之前的 checkpoint：沒有驗證集（測試集不受影響，切法不變）
    if 'VAL_NUM' not in saved:
        config.VAL_NUM = 0
    if 'VAL_FILES' not in saved:
        config.VAL_FILES = ()
    # 文字嵌入開關之前的 checkpoint 都有使用 FastText
    if 'USE_TEXT_EMBED' not in saved:
        config.USE_TEXT_EMBED = True
    # 架構開關之前的 checkpoint 都是邊特徵進入注意力的 GATv2
    if 'GNN_TYPE' not in saved:
        config.GNN_TYPE = "gatv2"
    if 'GNN_EDGE_IN_ATTN' not in saved:
        config.GNN_EDGE_IN_ATTN = True
    if 'GNN_EDGE_IN_PRED' not in saved:
        config.GNN_EDGE_IN_PRED = True
    # 幾何建圖之前（commit 423912b）的 checkpoint 一律是 id 鏈
    if 'MP_EDGE_MODE' not in saved and 'MP_EDGE_MODE' not in os.environ:
        config.MP_EDGE_MODE = "id"
    for k in CHECKPOINT_ENV_OVERRIDABLE_KEYS:
        if k not in saved:
            continue
        if k in os.environ:
            if str(getattr(config, k)) != str(saved[k]):
                print(f"⚠️  {k}：環境變數 {getattr(config, k)} 與 checkpoint 訓練時的 {saved[k]} 不同，以環境變數為準")
        else:
            setattr(config, k, saved[k])
    # 正規化參數：新 checkpoint 存有訓練時（只用訓練圖）擬合的結果，直接還原；
    # 舊 checkpoint 沒有，評估端照舊用全部地圖重新擬合（與它們訓練時的做法一致）
    st_sc = loaded.get('scalers')
    if st_sc:
        scaler = _scaler_from_state(st_sc['node'])
        edge_scaler = _scaler_from_state(st_sc['edge'])
        mp_scaler = _scaler_from_state(st_sc['mp'])
        _scalers_from_checkpoint = True
        if 'MP_EDGE_MODE' in os.environ and str(os.environ['MP_EDGE_MODE']) != str(saved.get('MP_EDGE_MODE')):
            print("⚠️  MP 模式與訓練時不同，但 mp_scaler 仍是依訓練時的 MP 模式擬合")
    # bigram 表：有就還原；需要卻沒有就直接報錯，不讓特徵靜靜地退化成常數
    st = loaded.get('text_prior')
    text_prior = TextPrior.from_state(st) if st is not None else None
    if 'bigram_lp' in _extra_edge_features() and text_prior is None:
        raise RuntimeError("checkpoint 的 EXTRA_EDGE_FEATURES 含 bigram_lp，但沒有封存 text_prior，無法重現訓練時的特徵")
    _D = edge_feature_dim()
    if _D != config.EDGE_FEATURE_DIM:
        raise RuntimeError(
            f"checkpoint 的 EDGE_FEATURE_DIM={config.EDGE_FEATURE_DIM} 與依其設定算出的維度 {_D} 不符"
            f"（EDGE_ANGLE_MODE={config.EDGE_ANGLE_MODE}, EXTRA_EDGE_FEATURES={config.EXTRA_EDGE_FEATURES}）")
    return saved


def _mp_attr_for(ocr_results, edge_index, ng=None):
    """把 [[src, dst], ...] 轉成未正規化特徵清單，順序與 edge_index 一一對應。"""
    if not edge_index:
        return []
    arr = np.asarray(edge_index, dtype=np.int64)
    return _edge_geom_features(ocr_results, arr[:, 0], arr[:, 1], ng).tolist()


def _mp_geometric_pairs(ocr_results, mode, anchor, k):
    """依模式回傳無向配對集合（不含 id / none / empty 這些特殊模式）。"""
    coords = _mp_topology_coords(ocr_results, anchor)
    if mode == "delaunay":
        return _mp_pairs_delaunay(coords)
    if mode == "gabriel":
        return _mp_pairs_gabriel(coords)
    if mode == "knn":
        return _mp_pairs_knn(coords, k)
    if mode == "dirnn":
        return _mp_pairs_dirnn(coords)
    raise ValueError(f"未知的 MP 幾何模式: {mode}")


def build_mp_edges(ocr_results, place_groups, node_indices, mode=None,
                   anchor=None, k=None, bidirectional=None):
    """
    建構 GNN 訊息傳遞邊。

    Args:
        ocr_results : build_graph 收集的文字框列表（順序即節點索引順序）
        place_groups: 依 id 分組的文字框（僅 mode="id" 需要）
        node_indices: (id, text, x, y) -> 節點索引（僅 mode="id" 需要）
        mode/anchor/k/bidirectional: 省略時取自 config

    Returns:
        (edge_index_list, raw_attr_list)
        raw_attr_list 為「未正規化」的邊特徵，由呼叫方套用 mp_scaler。
    """
    import config
    if mode is None:
        mode = getattr(config, "MP_EDGE_MODE", "id")
    if anchor is None:
        anchor = getattr(config, "MP_ANCHOR", "tl")
    if k is None:
        k = getattr(config, "MP_KNN_K", 8)
    if bidirectional is None:
        bidirectional = getattr(config, "BIDIRECTIONAL_GNN", False)

    mode = str(mode).strip()
    n = len(ocr_results)

    # ---- 特殊模式 ----
    if mode == "empty":
        return [], []
    if mode == "none":
        # 每節點一條自環邊，原始特徵全零（與 ablate_mp_edges.py 的退路一致）
        _D = edge_feature_dim()
        return [[i, i] for i in range(n)], [[0.0] * _D for _ in range(n)]

    # ---- 對照組：原樣的 id 鏈（嚴格保留原本的迭代順序與正/反向交錯）----
    if mode == "id":
        edge_index = []
        for place_id, words in place_groups.items():
            for i in range(len(words) - 1):
                src = node_indices[(place_id, words[i]["text"], words[i]["x"], words[i]["y"])]
                dst = node_indices[(place_id, words[i + 1]["text"], words[i + 1]["x"], words[i + 1]["y"])]
                edge_index.append([src, dst])
                if bidirectional:
                    edge_index.append([dst, src])
        # 反向邊直接以 (dst, src) 代入公式，與原本手寫的 [dist, -dx, -dy, rev_angle] 等價
        return edge_index, _mp_attr_for(ocr_results, edge_index)

    # ---- 幾何模式（含 union 聯集）----
    if mode.startswith("union:"):
        sub = [m.strip() for m in mode[len("union:"):].split("+") if m.strip()]
        if not sub:
            raise ValueError(f"union 模式未指定子模式: {mode}")
        pairs = set()
        for m in sub:
            pairs |= _mp_geometric_pairs(ocr_results, m, anchor, k)
    else:
        if mode not in VALID_MP_MODES:
            raise ValueError(
                f"未知的 MP_EDGE_MODE: {mode}（可用: {', '.join(VALID_MP_MODES)}，"
                f"或 union:a+b）"
            )
        pairs = _mp_geometric_pairs(ocr_results, mode, anchor, k)

    edge_index = []
    for src, dst in sorted(pairs):          # 排序確保結果可重現
        edge_index.append([src, dst])
        if bidirectional:
            edge_index.append([dst, src])
    return edge_index, _mp_attr_for(ocr_results, edge_index)


def _scaler_to_state(s):
    return {'mean': s.mean_.tolist(), 'scale': s.scale_.tolist(), 'var': s.var_.tolist(),
            'n_samples_seen': np.asarray(s.n_samples_seen_).tolist()}


def _scaler_from_state(st):
    s = StandardScaler()
    s.mean_ = np.asarray(st['mean'], dtype=np.float64)
    s.scale_ = np.asarray(st['scale'], dtype=np.float64)
    s.var_ = np.asarray(st['var'], dtype=np.float64)
    s.n_samples_seen_ = np.asarray(st['n_samples_seen'])
    s.n_features_in_ = s.mean_.shape[0]
    return s


def scalers_state():
    """三個 scaler 的參數（純數值清單）與擬合用的地圖，隨 checkpoint 封存。"""
    return {'node': _scaler_to_state(scaler), 'edge': _scaler_to_state(edge_scaler),
            'mp': _scaler_to_state(mp_scaler), 'fit_files': list(_scaler_fit_files)}


def fit_global_scaler(xml_files):
    global scaler, edge_scaler, mp_scaler, _scaler_fit_files
    import config
    if _scalers_from_checkpoint:
        print("📐 使用 checkpoint 中的正規化參數（訓練時只用訓練圖擬合），不重新擬合")
        return
    _scaler_fit_files = [f for f in xml_files if f not in config.SKIP_FILES]
    _mp_mode = str(getattr(config, "MP_EDGE_MODE", "id")).strip()
    # 下列三種模式的 mp_scaler 直接沿用 id 鏈的邊分布：
    #   id    : 本來就是 id 鏈，如此可位元級重現改動前的行為
    #   none  : 原始特徵全零，沿用 id 鏈的 fit，與 ablate_mp_edges.py 的退路一致
    #   empty : 沒有邊，scaler 不會被用到，但仍需一個可用的物件
    _mp_needs_own_fit = _mp_mode not in ("id", "none", "empty")
    global_features = []
    global_edge_features = []
    global_mp_features = []
    for file in xml_files:
        from config import SKIP_FILES, STOP_FILE
        if file in SKIP_FILES:
            continue
        elif file == STOP_FILE:
            break
        root = read_xml(file)

        ocr_results = []
        for image in root.findall(".//image"):
            map_id = os.path.splitext(image.get("file") or file)[0]
            for box in image.findall("box"):
                x = int(box.get("left"))
                y = int(box.get("top"))
                w = int(box.get("width"))
                h = int(box.get("height"))
                place_id = box.get("id")
                lab = box.find("label")
                text = (lab.text or "").strip() if lab is not None else ""
                #收集節點特徵 
                global_features.append([x, y, w, h])
                # 收集 OCR 資訊 提供後續計算邊特徵（text / map_id 供 lin_any、bigram_lp 使用）
                ocr_results.append({"x": x, "y": y, "w": w, "h": h, "id": place_id,
                                    "text": text, "map_id": map_id})

        place_groups = {}
        for item in ocr_results:
            if item["id"] == "0":
                continue
            if item["id"] not in place_groups:
                place_groups[item["id"]] = []
            place_groups[item["id"]].append(item)

        # place_groups 內的 dict 就是 ocr_results 裡的同一批物件，
        # 以物件身分反查索引，才能餵給共用的 _edge_geom_features。
        _idx_of = {id(o): i for i, o in enumerate(ocr_results)}
        _chain = []
        for place_id, words in place_groups.items():
            # words = sorted(words, key=lambda x: x["x"])  # 按 x 座標排序
            for i in range(len(words) - 1):
                _chain.append([_idx_of[id(words[i])], _idx_of[id(words[i + 1])]])
        if _chain:
            _chain = np.asarray(_chain, dtype=np.int64)
            global_edge_features.extend(
                _edge_geom_features(ocr_results, _chain[:, 0], _chain[:, 1]).tolist()
            )

        # 收集當前 MP 模式自己的邊特徵（供 mp_scaler 使用）。
        # 與 edge_scaler 保持一致：只 fit 正向邊（反向邊由 scaler 外推）。
        if _mp_needs_own_fit:
            _, _mp_attr = build_mp_edges(
                ocr_results, place_groups, None,
                mode=_mp_mode, bidirectional=False,
            )
            global_mp_features.extend(_mp_attr)
    # 節點特徵正規化
    global_features = np.array(global_features)
    scaler = StandardScaler()
    scaler.fit(global_features)
    # print(f"全局正規化參數 - 均值: {scaler.mean_}, 標準差: {scaler.scale_}")
    # 邊特徵正規化（候選邊 / edge_label_attr 用；fit 方式維持原樣）
    global_edge_features = np.array(global_edge_features)
    _D = edge_feature_dim()
    if global_edge_features.ndim != 2 or global_edge_features.shape[1] != _D:
        raise RuntimeError(
            f"邊特徵維度 {global_edge_features.shape} 與 edge_feature_dim()={_D} 不符")
    if _D != getattr(config, 'EDGE_FEATURE_DIM', _D):
        raise RuntimeError(
            f"config.EDGE_FEATURE_DIM={config.EDGE_FEATURE_DIM} 與實際產生的邊特徵維度 {_D} 不符，"
            f"請檢查 EXTRA_EDGE_FEATURES 設定")
    edge_scaler = StandardScaler()
    edge_scaler.fit(global_edge_features)
    # print(f"邊全局正規化參數 - 均值: {edge_scaler.mean_}, 標準差: {edge_scaler.scale_}")
    # MP 邊正規化（edge_attr 用）
    if _mp_needs_own_fit and len(global_mp_features) > 0:
        mp_scaler = StandardScaler()
        mp_scaler.fit(np.array(global_mp_features))
    else:
        # id / none / empty，或該模式在此資料集上完全沒有邊 → 沿用 id 鏈的 fit
        mp_scaler = StandardScaler()
        mp_scaler.fit(global_edge_features)
    print(f"🔗 MP 邊模式: {_mp_mode} | 拓樸座標: {getattr(config, 'MP_ANCHOR', 'tl')}"
          + (f" | k={getattr(config, 'MP_KNN_K', 8)}" if "knn" in _mp_mode else "")
          + f" | mp_scaler fit 樣本數: "
            f"{len(global_mp_features) if (_mp_needs_own_fit and global_mp_features) else len(global_edge_features)}")

def build_graph(root, model):
    global vocab, vocab_count
    # 1. 提取 OCR 資訊
    ocr_results = []
    for image in root.findall(".//image"):
            image_file = image.get("file")  # Get the file name
            # print(f"抓取Image: {image_file}內的資料 生成Data")
            map_id = os.path.splitext(image_file)[0]  # 獲取地圖ID
            
            for box in image.findall("box"):
                label = (box.find("label").text or "").strip()
                x = int(box.get("left"))
                y = int(box.get("top"))
                w = int(box.get("width"))
                h = int(box.get("height"))

                label_id = vocab.get(label, 0)

                place_id = box.get("id")  # 地名 ID
                ocr_results.append({
                    "text": label, 
                    "x": x, 
                    "y": y, 
                    "w": w, 
                    "h": h,
                    "id": place_id, 
                    'label_id': label_id,
                    'map_id': map_id  # 添加地圖ID
                })

    # 2. 按照 id 分組
    place_groups = {}
    for item in ocr_results:
        if item["id"] == "0":
            continue
        if item["id"] not in place_groups:
            place_groups[item["id"]] = []
        place_groups[item["id"]].append(item)

    # 3. 構造 PyG 節點特徵
    node_features = []
    label_ids = []
    node_indices = {}  # 儲存每個字對應的索引
    idx = 0

    for item in ocr_results:
        node_features.append([item['x'], item['y'], item['w'], item['h']])
        label_ids.append(item['label_id'])
        _key = (item['id'], item['text'], item["x"], item["y"])
        # 節點以 (地名 id, 字, 左上角座標) 為鍵；同一張圖出現完全相同的框會互相覆蓋，直接報錯
        if _key in node_indices:
            raise ValueError(f"{item['map_id']}：重複的字框 id={_key[0]} 字={_key[1]!r} 座標=({_key[2]}, {_key[3]})")
        node_indices[_key] = idx
        idx += 1
    
    node_features = scaler.transform(node_features)
    # node_features = np.hstack([node_features, label_ids])
    node_features = torch.tensor(node_features, dtype=torch.float)
    # label_ids = torch.tensor(label_ids, dtype=torch.long)

    # 4a. 構造「連線預測的正樣本標籤」——永遠來自 id 鏈，不受 MP 模式影響
    place_name_edge = {}  # 每個地名包含的邊（= 正樣本）
    for place_id, words in place_groups.items():
        # words = sorted(words, key=lambda x: x["x"])  # 確保是"台 → 北 → 市"順序
        for i in range(len(words) - 1):
            src = node_indices[(place_id, words[i]["text"], words[i]["x"], words[i]["y"])]
            dst = node_indices[(place_id, words[i + 1]["text"], words[i + 1]["x"], words[i + 1]["y"])]
            if words[i]["id"] not in place_name_edge:
                place_name_edge[words[i]["id"]] = []
            place_name_edge[words[i]["id"]].append([src, dst])

    # 4b. 構造「GNN 訊息傳遞邊」——依 config.MP_EDGE_MODE 決定
    #     MP_EDGE_MODE="id" 時與改動前完全等價（含正/反向交錯順序）。
    edge_index, edge_attr = build_mp_edges(ocr_results, place_groups, node_indices)

    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()  # 轉置成 (2, num_edges)
        # 以 mp_scaler 正規化訊息傳遞邊的特徵
        edge_attr = torch.tensor(mp_scaler.transform(edge_attr), dtype=torch.float)
    else:
        # MP_EDGE_MODE="empty"：真正的空邊集
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, edge_feature_dim()), dtype=torch.float)

    # 5. 建構 PyG Graph
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, label_id=torch.tensor(label_ids, dtype=torch.long))

    #整理要回傳的資料
    text_idx = {node_indices[item] : item for item in node_indices}#每個文字框對應的id
    place_groups = {item : "".join([i['text'] for i in place_groups[item]]) for item in place_groups}#每個地名id對應的地名

    return data, text_idx, place_groups, place_name_edge, ocr_results

def apply_node_dropout(data, ocr_results, place_name_edge, p_drop=0.0782):
    """
    動態隨機節點丟棄對抗訓練與局部傳遞閉包重構演算法
    --------------------------------------------------
    1. 隨機產生 bool mask 決定保留的節點。
    2. 切片保留節點特徵 (x) 與 label_id，並更新節點新舊索引對照。
    3. 過濾 OCR 文字框列表，依據 group_id 重構賸餘節點的地名鏈結（排序並重新首尾相連）。
    4. 重新構造 GNN 消息傳遞所需的邊 (edge_index) 與幾何邊特徵 (edge_attr) 並進行 scaler 正規化。
    """
    N = len(ocr_results)
    if N < 2:
        return data, place_name_edge, ocr_results
        
    # 1. 隨機生成保留 Mask
    keep_mask = np.random.rand(N) > p_drop
    if not np.any(keep_mask) or np.sum(keep_mask) < 2:
        # 防禦性設計：若全部被丟棄或剩下少於 2 個節點，則退回原完美圖
        return data, place_name_edge, ocr_results
        
    kept_indices = np.where(keep_mask)[0]
    
    # 2. 過濾節點特徵與文字標籤
    data.x = data.x[kept_indices]
    if hasattr(data, 'label_id') and data.label_id is not None:
        data.label_id = data.label_id[kept_indices]
        
    # 過濾 OCR 文字框列表
    kept_ocr_results = [ocr_results[i] for i in kept_indices]
    
    # 重新建立節點新索引映射表
    node_indices = {}
    for idx, item in enumerate(kept_ocr_results):
        node_indices[(item['id'], item['text'], item["x"], item["y"])] = idx
        
    # 3. 同地名實體的傳遞閉包重構 (localized Transitive Label Reconstruction)
    # 將剩下的文字框按地名 group_id 分組 (排除背景字 '0')
    kept_place_groups = {}
    for item in kept_ocr_results:
        pid = item["id"]
        if pid == "0":
            continue
        if pid not in kept_place_groups:
            kept_place_groups[pid] = []
        kept_place_groups[pid].append(item)
        
    # 組內兩兩重新首尾相連 (kept_ocr_results 保持了原始順序，故 words 天然遞增對齊)
    new_place_name_edge = {}
    for pid, words in kept_place_groups.items():
        for i in range(len(words) - 1):
            src = node_indices[(pid, words[i]["text"], words[i]["x"], words[i]["y"])]
            dst = node_indices[(pid, words[i + 1]["text"], words[i + 1]["x"], words[i + 1]["y"])]
            
            if pid not in new_place_name_edge:
                new_place_name_edge[pid] = []
            new_place_name_edge[pid].append([src, dst])
            
    # 4. 重新構造 GNN 消息傳遞的邊與正規化幾何屬性
    #    與 build_graph 一致：拓樸由 config.MP_EDGE_MODE 決定，
    #    上方 new_place_name_edge（標籤）則永遠來自 id 鏈。
    edge_index, edge_attr = build_mp_edges(
        kept_ocr_results, kept_place_groups, node_indices
    )

    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.tensor(mp_scaler.transform(edge_attr), dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, edge_feature_dim()), dtype=torch.float)
        
    data.edge_index = edge_index
    data.edge_attr = edge_attr
    
    return data, new_place_name_edge, kept_ocr_results

def reflect_xml_root(root):
    """
    直橫書翻轉增強：回傳沿反對角線翻轉的 XML 副本（原 root 不動）。
      (x, y) → (S − (y + h), S − (x + w))，w ↔ h，S = 該圖所有框的最大座標（保持座標為正）
    右→左的橫書字序變成上→下的直書字序，反之亦然；id / label / 順序全部不變。
    """
    import copy
    root2 = copy.deepcopy(root)
    boxes = list(root2.iter("box"))
    if not boxes:
        return root2
    S = max(max(int(b.get("left")) + int(b.get("width")), int(b.get("top")) + int(b.get("height"))) for b in boxes)
    for b in boxes:
        l, t, w, h = int(b.get("left")), int(b.get("top")), int(b.get("width")), int(b.get("height"))
        b.set("left", str(S - (t + h)))
        b.set("top", str(S - (l + w)))
        b.set("width", str(h))
        b.set("height", str(w))
    return root2


def hard_negative_mask(ocr_results, src, dst, ng=None):
    """
    「難負邊」遮罩（train.py 的 NEG_KEEP_HARD 用）：兩端平均字級 > HARD_NEG_SIZE_REL × 該圖中位；
    HARD_NEG_INCLUDE_LIN=1 時再加上任一端是線性地物通名的對。
    直接由幾何 / 文字算，不依賴 EXTRA_EDGE_FEATURES 有沒有開對應特徵。
    """
    import config
    if ng is None:
        ng = _node_geom(ocr_results)
    src = np.asarray(src, dtype=np.int64); dst = np.asarray(dst, dtype=np.int64)
    msize = (ng.size[src] + ng.size[dst]) / 2.0
    hard = msize > float(getattr(config, "HARD_NEG_SIZE_REL", 1.5)) * max(ng.size_med, 1e-6)
    if getattr(config, "HARD_NEG_INCLUDE_LIN", False):
        hard = hard | (np.maximum(ng.lin[src], ng.lin[dst]) > 0)
    return hard


def build_data(data, place_name_edge, ocr_results, is_test=False):
    import config
    idx = len(data.x)
    
    # 所有地名都作為正樣本
    all_places = list(place_name_edge.keys())
    all_edges = [edge for place in all_places for edge in place_name_edge[place]]
    pos_set = set(map(tuple, all_edges))
    
    # 轉換為張量
    # 沒有任何正邊時（例如每框 id 皆唯一的 LLM 偵測輸入）保持 (2, 0) 形狀，後面的 cat 才接得上
    pos_edges_tensor = (torch.tensor(all_edges, dtype=torch.long).T if all_edges
                        else torch.empty((2, 0), dtype=torch.long))
    
    # --- 向量化計算全部正樣本邊特徵 (src 用於 Link Predictor 考卷) ---
    coords = np.array([[r["x"], r["y"]] for r in ocr_results], dtype=np.float32)
    
    # 節點層級中間量（最近鄰距等）每張圖只算一次，正/負樣本共用
    node_geom = _node_geom(ocr_results)

    pos_arr = np.array(all_edges, dtype=np.int64).reshape(-1, 2)  # (E, 2)
    pos_raw_attr = _edge_geom_features(ocr_results, pos_arr[:, 0], pos_arr[:, 1], node_geom)
    pos_edge_attr_np = (edge_scaler.transform(pos_raw_attr) if len(pos_arr)
                        else np.zeros((0, edge_scaler.n_features_in_), dtype=np.float64))
    pos_edge_attr_tensor = torch.tensor(pos_edge_attr_np, dtype=torch.float)

    # =======================================================
    # 模式切換： 依據 is_test 決定使用 NxN 或 1:1 取樣
    # =======================================================
    mode = getattr(config, 'TEST_SAMPLE_MODE', 'NxN') if is_test else getattr(config, 'TRAIN_SAMPLE_MODE', 'NxN')
    if mode == "NxN":
        # --- 全圖 NxN 向量化生成 ---
        N = idx
        i_idx, j_idx = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        mask_self = i_idx != j_idx  # 排除自環
        all_i = i_idx[mask_self]
        all_j = j_idx[mask_self]

        # 根據 NXN_EDGE_DISTANCE_LIMIT 過濾過長的候選負邊
        limit = getattr(config, 'NXN_EDGE_DISTANCE_LIMIT', 0)
        if limit > 0:
            src_coords = coords[all_i]
            dst_coords = coords[all_j]
            dx = dst_coords[:, 0] - src_coords[:, 0]
            dy = dst_coords[:, 1] - src_coords[:, 1]
            dist = np.sqrt(dx**2 + dy**2)
            valid_mask = dist <= limit
            all_i = all_i[valid_mask]
            all_j = all_j[valid_mask]

        # 排除正樣本邊
        # 向量化：把 (i, j) 編碼成 i*N+j 的單一整數鍵，用 np.isin 一次過濾，
        # 取代原本對數千萬組候選對逐一 hash 的 Python 迴圈。
        # 輸出的順序與內容與原實作完全一致。
        cand_keys = all_i.astype(np.int64) * N + all_j.astype(np.int64)
        if pos_arr.size > 0:
            pos_keys = pos_arr[:, 0].astype(np.int64) * N + pos_arr[:, 1].astype(np.int64)
            keep_mask = ~np.isin(cand_keys, pos_keys)
        else:
            keep_mask = np.ones(cand_keys.shape[0], dtype=bool)
        neg_pairs = np.stack([all_i[keep_mask], all_j[keep_mask]], axis=1).astype(np.int64)

        # 向量化計算所有負樣本的邊特徵
        if len(neg_pairs) > 0:
            neg_raw_attr = _edge_geom_features(
                ocr_results, neg_pairs[:, 0], neg_pairs[:, 1], node_geom)
            neg_edge_attr_np = edge_scaler.transform(neg_raw_attr)
            neg_edges_tensor = torch.tensor(neg_pairs.T, dtype=torch.long)
            neg_edge_attr_tensor = torch.tensor(neg_edge_attr_np, dtype=torch.float)
        else:
            neg_edges_tensor = torch.empty((2, 0), dtype=torch.long)
            neg_edge_attr_tensor = torch.empty((0, edge_feature_dim()), dtype=torch.float)
    else:
        # --- 原始 1:1 隨機取樣 ---
        num_neg = pos_edges_tensor.shape[1]
        neg_edges_list = []
        t = 0
        while len(neg_edges_list) < num_neg and t < 1000000:
            t += 1
            i, j = np.random.randint(0, idx), np.random.randint(0, idx)
            if i == j or (i, j) in pos_set or (j, i) in pos_set:
                continue
            d_x = coords[j, 0] - coords[i, 0]
            d_y = coords[j, 1] - coords[i, 1]
            d = np.sqrt(d_x**2 + d_y**2)
            dist_limit = getattr(config, 'NEG_EDGE_DISTANCE_LIMIT', 300)
            if d < dist_limit:
                neg_edges_list.append((i, j))
        if t >= 1000000:
            print("Warning: 無法產生足夠的負樣本")
        if len(neg_edges_list) > 0:
            _neg_arr = np.asarray(neg_edges_list, dtype=np.int64).reshape(-1, 2)
            neg_edge_attr_np = edge_scaler.transform(
                _edge_geom_features(ocr_results, _neg_arr[:, 0], _neg_arr[:, 1], node_geom))
            neg_edges_tensor = torch.tensor(neg_edges_list, dtype=torch.long).T
            neg_edge_attr_tensor = torch.tensor(neg_edge_attr_np, dtype=torch.float)
        else:
            # 完全找不到負樣本（節點極少 or 全在 pos_set 中）
            neg_edges_tensor = torch.empty((2, 0), dtype=torch.long)
            neg_edge_attr_tensor = torch.empty((0, edge_feature_dim()), dtype=torch.float)

    # 建構最終 edge_label_index 和考卷標籤
    edge_label_index = torch.cat([pos_edges_tensor, neg_edges_tensor], dim=1)
    edge_label_attr = torch.cat([pos_edge_attr_tensor, neg_edge_attr_tensor], dim=0)
    edge_label = torch.cat([
        torch.ones(pos_edges_tensor.shape[1]),
        torch.zeros(neg_edges_tensor.shape[1])
    ])

    data.edge_label_index = edge_label_index
    data.edge_label = edge_label
    data.edge_label_attr = edge_label_attr
    # 難負邊遮罩（與 edge_label 對齊；正邊一律 False）。train.py 會把它取出來另存，
    # 不留在餵給 LinkNeighborLoader 的 Data 裡。
    _hard = np.zeros(edge_label.numel(), dtype=bool)
    if neg_edges_tensor.shape[1] > 0:
        _ne = neg_edges_tensor.numpy()
        _hard[pos_edges_tensor.shape[1]:] = hard_negative_mask(ocr_results, _ne[0], _ne[1], node_geom)
    data.edge_label_hard = torch.from_numpy(_hard)

    return data.contiguous()

class EdgeLabelAttrTransform:
    """
    LinkNeighborLoader 用的 transform，把候選邊特徵正確對齊到該批抽到的邊。

    為什麼需要它
    ------------
    PyG 依「第一維大小」推斷屬性是節點層級還是邊層級。edge_label_attr 的
    第一維是候選邊總數，既不等於節點數也不等於 edge_index 的邊數，PyG 無法
    歸類，因此**不會切片**——每個 batch 都會原封不動收到整包全域陣列。

    舊版 keep_edge_label_attr 以 batch 內的位置 i 去索引該全域陣列
    (`data.edge_label_attr[i]`)，等於永遠取回全域的前 N 列，與這批實際抽到
    的邊無關。又因 build_data 先接正樣本再接負樣本，那前 N 列全是正邊特徵，
    等於不論正負樣本都被餵入「看起來像正邊」的短距離特徵；而評估路徑
    (metrics.check_pred_clusters) 是正確對齊的，形成訓練/評估的分布不一致。

    正確做法是用 LinkNeighborLoader 在 batch 上附帶的 input_id，
    即這批種子邊在原始 edge_label_index 中的索引。

    為什麼是類別而不是閉包
    ----------------------
    DataLoader 以 spawn / forkserver 啟動 worker 時需要 pickle transform，
    而巢狀定義的局部函式無法序列化（Can't pickle local object）。
    類別實例可以，因此 num_workers > 0 時必須用這個形式。
    """

    def __init__(self, global_edge_label_attr):
        self.global_edge_label_attr = global_edge_label_attr

    def __call__(self, data):
        input_id = getattr(data, 'input_id', None)
        if input_id is None:
            raise RuntimeError(
                "batch 缺少 input_id，無法對齊 edge_label_attr。"
                "請確認使用的是 LinkNeighborLoader（NodeLoader 不提供 input_id）。"
            )
        data.edge_label_attr = self.global_edge_label_attr[input_id]
        n_edge = data.edge_label_index.size(1)
        if data.edge_label_attr.size(0) != n_edge:
            raise RuntimeError(
                f"edge_label_attr 列數 {data.edge_label_attr.size(0)} "
                f"與 edge_label_index 邊數 {n_edge} 不符"
            )
        return data


def make_edge_label_attr_transform(global_edge_label_attr):
    """回傳可 pickle 的 transform 實例（見 EdgeLabelAttrTransform）。"""
    return EdgeLabelAttrTransform(global_edge_label_attr)


def keep_edge_label_attr(data):
    """已停用。舊實作會把 batch 的第 i 條邊配上全域候選陣列的第 i 列，
    導致訓練時的邊特徵與該批的邊無關。請改用
    make_edge_label_attr_transform(combined.edge_label_attr)。"""
    raise RuntimeError(
        "keep_edge_label_attr 已停用（索引錯位，見 docstring）。"
        "請改用 make_edge_label_attr_transform(<全域 edge_label_attr>)。"
    )


# 在主程式開始前，預先構建詞彙表
def build_global_vocab(xml_files):
    global vocab, vocab_count
    vocab = {}
    vocab_count = 1
    n = 0
    from config import SKIP_FILES, STOP_FILE
    for file in xml_files:
        if file in SKIP_FILES:
            continue
        elif file == STOP_FILE:
            break
        n += 1
        # 讀檔失敗直接報錯（不跳過），避免詞彙表悄悄少了一張圖的字
        root = read_xml(file)
        for image in root.findall(".//image"):
            for box in image.findall("box"):
                label = box.find("label").text.strip()
                if label not in vocab:
                    vocab[label] = vocab_count
                    vocab_count += 1
    print(f"詞彙表大小: {len(vocab)}")  # 打印詞彙表大小


def restore_vocab(loaded):
    """checkpoint 存有詞彙表時直接沿用，評估端不必依賴 xml_revised 的內容與建表時完全相同。
    舊 checkpoint 沒有存詞彙表，維持 build_global_vocab 重建的結果。回傳是否有還原。"""
    global vocab, vocab_count
    saved = loaded.get('vocab') if isinstance(loaded, dict) else None
    if not saved:
        return False
    vocab = dict(saved)
    vocab_count = max(vocab.values()) + 1
    print(f"詞彙表改用 checkpoint 內存的版本（{len(vocab)} 字）")
    return True

def load_fasttext_embeddings(ft_model_path=None):
    """載入 FastText 模型並建立 embedding_matrix"""
    global vocab, vocab_count
    import config
    if not getattr(config, 'USE_TEXT_EMBED', True):
        # 消融用：架構不變，只把每個字的文字輸入換成零向量（嵌入層凍結，訓練中維持為零）
        vocab_size = len(vocab) + 1
        print("文字嵌入已關閉（USE_TEXT_EMBED=0）：所有字的文字輸入皆為零向量，不載入 FastText")
        return 300, vocab_size, np.zeros((vocab_size, 300), dtype=np.float32)
    if ft_model_path is None:
        ft_model_path = os.path.join(DATA_DIR, 'cc.zh.300.kv')
    
    print("載入模型")
    ft_model = KeyedVectors.load(ft_model_path)

    print("初始化模型")
    embedding_dim = ft_model.vector_size  # 例如300
    vocab_size = len(vocab) + 1  # +1是給UNK用
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)

    # FastText 沒收錄的字（約 150 字、0.8% 字框）維持零向量，與詞彙表外的字（索引 0）相同。
    # 以前用未設種子的隨機向量，訓練與評估是不同的兩個行程，同一個字會拿到不同的向量。
    print("初始化嵌入")
    for word, idx in vocab.items():
        if word in ft_model:
            embedding_matrix[idx] = ft_model[word]
    print("fasttext完成!")
    
    return embedding_dim, vocab_size, embedding_matrix
