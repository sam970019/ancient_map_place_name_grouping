"""
LLM 獨立偵測（Stage 0）：把整張古地圖切塊丟給 Azure OpenAI 視覺模型，
回傳「一字一框」的候選字框，還原到全圖座標並在重疊區去重。

流程：原圖 → 切塊(overlap) → (可選)放大 → VLM → 塊內座標 → 除以放大倍率 + 塊偏移 → 全圖字框 → IoU 去重

輸出（DATA_DIR/llm_detect/<cfg>/）：
    <gt_id>/tiles/<x0>_<y0>.json   每塊的原始回覆與解析結果（永久快取；重跑先讀這裡，不重花錢）
    <gt_id>.json                   合併去重後的全圖字框 [{text,left,top,width,height,conf,tile}]
    <gt_id>_overlay.png            (--draw) 字框畫回原圖，方便肉眼檢查

用法：
    python llm_detect.py --maps 25147 15692 --tile 768 --scale 1 --overlap 192 --workers 8
    python llm_detect.py --maps 15692 --tile 384 --scale 2 --overlap 96
    python llm_detect.py --maps 25147 --region 3584 0 1536 1536   # 只跑一小塊區域做 smoke test
    python llm_detect.py --maps 25147 --eval                        # 對 xml_revised 的 GT 算字框 P/R（偵測層）

設計依據（前期探路實驗的結論）：
    - detail=high 會把圖縮到短邊 768 再切 512 塊，所以送進去的圖 **短邊不能超過 768**，
      否則模型看到的座標系和我們的不一致（回傳座標會在 768/1024 之間漂移）。
    - 難圖（細筆畫、等高線穿字）用 tile=384、scale=2 放大送，字正確率明顯較好，但塊數 ×4。
    - reasoning_effort 預設 medium；low 會多幻覺（把不確定的字補成合理地名）。
"""
import os
import re
import sys
import json
import time
import base64
import random
import argparse
import threading
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data import DATA_DIR

Image.MAX_IMAGE_PIXELS = None

ORIGINAL_DIR = os.path.join(DATA_DIR, "map", "original")
GT_DIR = os.path.join(DATA_DIR, "xml_revised")
OUT_ROOT = os.path.join(DATA_DIR, "llm_detect")
FONT_PATH = os.path.join(DATA_DIR, "map", "font", "msjhbd.ttc")

PROMPT_VERSION = "p6"
PROMPT_TEMPLATE = (
    "你是一個古地圖中文字偵測器。輸入是一張 {W}×{H} 像素的地圖圖塊（民國初年，含直書與橫書，橫書由右至左）。\n"
    "任務：找出圖中**每一個**印刷中文字，並給出其外接矩形框。\n"
    "規則：\n"
    "1. 一個字一個框，不要把整個地名合成一框。\n"
    "2. 座標用這張圖的像素座標，左上角為原點，欄位 left/top/width/height 皆為整數。\n"
    "3. 只輸出圖上實際看得到的字；看不清楚或不確定的字，text 填 \"?\"，不要憑地名常識猜字。\n"
    "4. 忽略阿拉伯數字、拉丁字母與地圖符號；直書、橫書、旋轉的字都要。\n"
    "5. 不要分組、不要排序、不要解釋。\n"
    "6. 照圖上的字形輸出，不要改寫成標準字、通用字或做簡繁轉換（例如圖上是 鉄、窑、岺 就寫 鉄、窑、岺，不要寫成 鐵、窯、嶺）。\n"
    "每個框另加 \"conf\"（0–1，對「這個位置有字且文字正確」的把握）。\n"
    "輸出格式（純 JSON 陣列，不要 markdown）："
    "[{{\"text\":\"馮\",\"left\":412,\"top\":140,\"width\":39,\"height\":33,\"conf\":0.9}}, ...]"
)


# ---------------------------------------------------------------- 幾何工具
def box_iou(a, b):
    ax2, ay2 = a["left"] + a["width"], a["top"] + a["height"]
    bx2, by2 = b["left"] + b["width"], b["top"] + b["height"]
    iw = max(0, min(ax2, bx2) - max(a["left"], b["left"]))
    ih = max(0, min(ay2, by2) - max(a["top"], b["top"]))
    inter = iw * ih
    union = a["width"] * a["height"] + b["width"] * b["height"] - inter
    return inter / union if union > 0 else 0.0


def tile_origins(size, tile, overlap):
    """沿一個軸產生圖塊起點：固定步距，最後一塊貼齊邊緣（不會超出圖外）。"""
    step = tile - overlap
    if size <= tile:
        return [0]
    xs = list(range(0, size - tile, step))
    xs.append(size - tile)
    return xs


def make_tiles(W, H, tile, overlap, region=None):
    """回傳 [(x0, y0)]；region=(rx, ry, rw, rh) 時只保留與該區域相交的塊（smoke test 用）。"""
    tiles = [(x, y) for y in tile_origins(H, tile, overlap) for x in tile_origins(W, tile, overlap)]
    if region:
        rx, ry, rw, rh = region
        tiles = [(x, y) for (x, y) in tiles if x < rx + rw and x + tile > rx and y < ry + rh and y + tile > ry]
    return tiles


# ---------------------------------------------------------------- 呼叫與解析
_client_lock = threading.Lock()
_client = None


def get_client():
    global _client
    with _client_lock:
        if _client is None:
            from openai import AzureOpenAI
            _client = AzureOpenAI(api_key=config.AZURE_API_KEY, azure_endpoint=config.AZURE_ENDPOINT,
                                  api_version="2024-12-01-preview", timeout=300.0)
    return _client


def parse_boxes(txt):
    """把模型回覆解析成 [{text,left,top,width,height,conf}]；容忍 markdown 圍欄與前後雜訊。"""
    if not txt:
        return []
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt.strip(), flags=re.M)
    data = None
    try:
        data = json.loads(s)
    except Exception:
        m = re.search(r"\[.*\]", s, re.S)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                data = None
    if not isinstance(data, list):
        return []
    out = []
    for p in data:
        if not isinstance(p, dict):
            continue
        try:
            b = dict(text=str(p.get("text", "?")).strip() or "?",
                     left=float(p["left"]), top=float(p["top"]),
                     width=float(p["width"]), height=float(p["height"]),
                     conf=float(p.get("conf", 1.0)))
        except (KeyError, TypeError, ValueError):
            continue
        out.append(b)
    return out


# Gemini：原生座標格式是 box_2d [ymin, xmin, ymax, xmax]、歸一化到 0–1000（官方文件）。
# 實測要求像素座標時它仍回傳 0–1000 的值，所以改用原生格式、合併時再換算回像素。
GEMINI_PROMPT_VERSION = "g1"
GEMINI_PROMPT = (
    "你是一個古地圖中文字偵測器。輸入是一張地圖圖塊（民國初年，含直書與橫書，橫書由右至左）。\n"
    "任務：找出圖中每一個印刷中文字，一個字一個框。\n"
    "規則：只輸出圖上實際看得到的字；看不清楚或不確定的字，text 填 \"?\"，不要憑地名常識猜字。"
    "忽略阿拉伯數字、拉丁字母與地圖符號；直書、橫書都要。照圖上字形輸出，不要改寫成標準字或簡繁轉換。\n"
    "box_2d 用 [ymin, xmin, ymax, xmax]，數值歸一化到 0–1000。每個框附 conf（0–1）。\n"
    "輸出 JSON 陣列：[{\"text\":\"馮\",\"box_2d\":[ymin,xmin,ymax,xmax],\"conf\":0.9}, ...]"
)


def is_gemini(deployment):
    return bool(deployment) and deployment.startswith("gemini")


def parse_box2d(txt, size):
    """Gemini 的 box_2d（0–1000 歸一化）→ 與 parse_boxes 相同格式的像素框；size = 送出圖的邊長。"""
    if not txt:
        return []
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt.strip(), flags=re.M)
    try:
        data = json.loads(s)
    except Exception:
        m = re.search(r"\[.*\]", s, re.S)
        try:
            data = json.loads(m.group(0)) if m else None
        except Exception:
            data = None
    out = []
    for p in data if isinstance(data, list) else []:
        try:
            ymin, xmin, ymax, xmax = [float(v) for v in p["box_2d"]]
        except (KeyError, TypeError, ValueError):
            continue
        l, t = xmin / 1000 * size, ymin / 1000 * size
        out.append(dict(text=str(p.get("text", "?")).strip() or "?", left=l, top=t,
                        width=(xmax - xmin) / 1000 * size, height=(ymax - ymin) / 1000 * size,
                        conf=float(p.get("conf", 1.0))))
    return out


def call_gemini(img, prompt, effort, max_tokens, max_retries, backoff, deployment):
    """Gemini REST 單塊呼叫（金鑰放 header）；effort 對應 thinkingLevel。回傳格式同 call_vlm。"""
    import urllib.request
    import urllib.error
    buf = BytesIO()
    img.save(buf, format="PNG")
    gen = {"maxOutputTokens": max_tokens, "responseMimeType": "application/json"}
    if effort:
        gen["thinkingConfig"] = {"thinkingLevel": effort}
    body = json.dumps({"contents": [{"parts": [
        {"text": prompt},
        {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(buf.getvalue()).decode()}}]}],
        "generationConfig": gen}).encode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{deployment}:generateContent"
    last_err = None
    for attempt in range(max_retries):
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json",
                                                              "x-goog-api-key": config.GEMINI_API_KEY})
        try:
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.load(r)
            c = (d.get("candidates") or [{}])[0]
            raw = "".join(p.get("text", "") for p in c.get("content", {}).get("parts", []) if not p.get("thought"))
            u = d.get("usageMetadata", {})
            usage = dict(prompt_tokens=u.get("promptTokenCount", 0),
                         completion_tokens=u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0),
                         reasoning_tokens=u.get("thoughtsTokenCount"), finish=c.get("finishReason"),
                         model=d.get("modelVersion"))
            return raw, usage, time.time() - t0
        except urllib.error.HTTPError as e:
            last_err = RuntimeError(f"HTTP {e.code}: {e.read().decode()[:300]}")
            if e.code not in (429, 500, 502, 503, 504) or attempt == max_retries - 1:
                raise last_err
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = e
            if attempt == max_retries - 1:
                raise
        time.sleep((backoff ** attempt) + random.uniform(1, 3))
    raise last_err


# Claude（Azure Foundry）：官方文件建議要求絕對像素座標 [x1, y1, x2, y2]，歸一化座標效果差。
# 2026-10-17 小測：c1（bbox）比 p6（left/top/width/height）召回高 11%；768 圖塊在高解析度級內，不會被縮。
CLAUDE_PROMPT_VERSION = "c1"
CLAUDE_PROMPT_TEMPLATE = PROMPT_TEMPLATE.replace(
    "2. 座標用這張圖的像素座標，左上角為原點，欄位 left/top/width/height 皆為整數。",
    "2. 座標用這張圖的像素座標，左上角為原點；bbox 為 [x1, y1, x2, y2]（左上角與右下角），皆為整數。").replace(
    "[{{\"text\":\"馮\",\"left\":412,\"top\":140,\"width\":39,\"height\":33,\"conf\":0.9}}, ...]",
    "[{{\"text\":\"馮\",\"bbox\":[412,140,451,173],\"conf\":0.9}}, ...]")
assert "bbox" in CLAUDE_PROMPT_TEMPLATE and "left/top" not in CLAUDE_PROMPT_TEMPLATE


def is_claude(deployment):
    return bool(deployment) and deployment.startswith("claude")


def parse_bbox(txt):
    """Claude 的 bbox [x1, y1, x2, y2]（像素）→ 與 parse_boxes 相同格式。"""
    if not txt:
        return []
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt.strip(), flags=re.M)
    try:
        data = json.loads(s)
    except Exception:
        m = re.search(r"\[.*\]", s, re.S)
        try:
            data = json.loads(m.group(0)) if m else None
        except Exception:
            data = None
    out = []
    for p in data if isinstance(data, list) else []:
        try:
            x1, y1, x2, y2 = [float(v) for v in p["bbox"]]
        except (KeyError, TypeError, ValueError):
            continue
        out.append(dict(text=str(p.get("text", "?")).strip() or "?", left=x1, top=y1,
                        width=x2 - x1, height=y2 - y1, conf=float(p.get("conf", 1.0))))
    return out


_claude_client = None


def get_claude_client(max_retries):
    """金鑰：環境變數 ANTHROPIC_API_KEY；app 未重啟時讀 Windows 使用者環境變數。資源名取自 CLAUDE_ENDPOINT 的主機名。"""
    global _claude_client
    with _client_lock:
        if _claude_client is None:
            from urllib.parse import urlparse
            from anthropic import AnthropicFoundry

            def env(name):
                if os.environ.get(name):
                    return os.environ[name]
                import winreg
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                    return winreg.QueryValueEx(k, name)[0]
            resource = urlparse(env("CLAUDE_ENDPOINT")).hostname.split(".")[0]
            _claude_client = AnthropicFoundry(api_key=env("ANTHROPIC_API_KEY"), resource=resource,
                                              max_retries=max_retries, timeout=900)
        return _claude_client


def call_claude(img, prompt, effort, max_tokens, max_retries, backoff, deployment):
    """Claude 單塊呼叫（串流取最終訊息；429/5xx 由 SDK 退避重試）。回傳格式同 call_vlm。"""
    buf = BytesIO()
    img.save(buf, format="PNG")
    params = dict(model=deployment, max_tokens=max_tokens, messages=[{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                     "data": base64.standard_b64encode(buf.getvalue()).decode()},
         "transformations": {"oversized_image": "error"}},   # 被縮圖會讓座標漂移，寧可報錯
        {"type": "text", "text": prompt}]}])
    if effort:
        params["output_config"] = {"effort": effort}
    t0 = time.time()
    with get_claude_client(max_retries).messages.stream(**params) as st:
        r = st.get_final_message()
    raw = "".join(b.text for b in r.content if b.type == "text")
    usage = dict(prompt_tokens=r.usage.input_tokens, completion_tokens=r.usage.output_tokens,
                 reasoning_tokens=None, finish=r.stop_reason, model=r.model)
    return raw, usage, time.time() - t0


def call_vlm(img, prompt, effort, max_tokens, max_retries, backoff, deployment=None):
    """單塊呼叫；429 / 暫時性錯誤指數退避。回傳 (raw_text, usage_dict, seconds)。"""
    if is_gemini(deployment):
        return call_gemini(img, prompt, effort, max_tokens, max_retries, backoff, deployment)
    if is_claude(deployment):
        return call_claude(img, prompt, effort, max_tokens, max_retries, backoff, deployment)
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
    ]}]
    params = dict(model=deployment or config.DEPLOYMENT_NAME, messages=messages, max_completion_tokens=max_tokens)
    if effort:
        params["reasoning_effort"] = effort
    last_err = None
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            r = get_client().chat.completions.create(**params)
            u = r.usage
            usage = dict(prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens,
                         reasoning_tokens=getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", None),
                         finish=r.choices[0].finish_reason, model=r.model)
            return r.choices[0].message.content, usage, time.time() - t0
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            transient = any(k in msg for k in ("429", "rate limit", "too many requests", "timeout", "timed out",
                                                 "502", "503", "504", "connection", "overloaded"))
            if not transient or attempt == max_retries - 1:
                raise
            time.sleep((backoff ** attempt) + random.uniform(1, 3))
    raise last_err


# ---------------------------------------------------------------- 單張圖
def model_tag(deployment=None):
    """deployment 名 → 目錄用短名，例如 gpt-5.6-sol → sol、gpt-6-astra → astra、gemini-3.1-pro-preview → g31pro。"""
    d = deployment or config.DEPLOYMENT_NAME
    if is_gemini(d):
        return d.replace("gemini-", "g").replace("-preview", "").replace(".", "").replace("-", "")
    if is_claude(d):
        return d.replace("claude-", "").replace("-", "")   # claude-opus-5-5 → opus55
    return d.split("-")[-1]


def prompt_version(deployment=None):
    if is_gemini(deployment):
        return GEMINI_PROMPT_VERSION
    return CLAUDE_PROMPT_VERSION if is_claude(deployment) else PROMPT_VERSION


def cfg_name(tile, scale, overlap, effort, deployment=None):
    return f"t{tile}_s{scale}_o{overlap}_{effort or 'default'}_{prompt_version(deployment)}_{model_tag(deployment)}"


def process_tile(img, x0, y0, tile, scale, cache_path, args):
    """裁一塊 → (放大) → 呼叫 → 存快取（只存原始回覆與用量）。已有快取則直接讀。"""
    if os.path.exists(cache_path) and not args.force:
        with open(cache_path, encoding="utf8") as f:
            return json.load(f), True
    crop = img.crop((x0, y0, x0 + tile, y0 + tile))
    send = crop.resize((tile * scale, tile * scale), Image.LANCZOS) if scale != 1 else crop
    gem = is_gemini(getattr(args, "deployment", None))
    cld = is_claude(getattr(args, "deployment", None))
    if gem:
        prompt = GEMINI_PROMPT
    else:
        prompt = (CLAUDE_PROMPT_TEMPLATE if cld else PROMPT_TEMPLATE).format(W=send.width, H=send.height)
    if args.dry_run:
        raw, usage, secs = "[]", {}, 0.0
    else:
        raw, usage, secs = call_vlm(send, prompt, args.effort, args.max_tokens, args.max_retries, args.backoff,
                                    getattr(args, "deployment", None))
    rec = dict(tile=[x0, y0], size=tile, scale=scale, seconds=round(secs, 1), usage=usage, raw=raw)
    if cld:
        rec["fmt"] = "bbox"
    if gem:
        rec["fmt"] = "box2d"   # 合併時依此換算座標
        # Gemini 偶爾整塊退化成「框都對、字全是 ?」或沒正常結束：只依輸出格式判斷（不看答案）重試，
        # 每次嘗試的原始回覆都留在 raw_attempts，事後可分別評估「只用第一次」與「重試後」。
        def degenerate(r, u):
            bs = parse_box2d(r, tile * scale)
            q = sum(1 for b in bs if b["text"] == "?")
            return (len(bs) >= 5 and q / len(bs) >= 0.8) or (u.get("finish") not in (None, "STOP"))
        if not args.dry_run and degenerate(raw, usage):
            attempts = [dict(raw=raw, usage=usage, seconds=round(secs, 1))]
            for _ in range(2):
                raw, usage, secs = call_vlm(send, prompt, args.effort, args.max_tokens, args.max_retries,
                                            args.backoff, args.deployment)
                attempts.append(dict(raw=raw, usage=usage, seconds=round(secs, 1)))
                if not degenerate(raw, usage):
                    break
            rec.update(raw=raw, usage=usage, raw_attempts=attempts,
                       seconds=round(sum(a["seconds"] for a in attempts), 1))
    if args.dry_run:
        return rec, False
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp = cache_path + ".tmp"
    with open(tmp, "w", encoding="utf8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    os.replace(tmp, cache_path)
    return rec, False


def tile_boxes(rec, img_size):
    """快取的原始回覆 → 全圖座標字框。解析與過濾都在這裡做，改規則不用重打 API。"""
    x0, y0 = rec["tile"]
    tile, scale = rec["size"], rec["scale"]
    W_img, H_img = img_size
    boxes = []
    fmt = rec.get("fmt")
    parsed = (parse_box2d(rec["raw"], tile * scale) if fmt == "box2d" else
              parse_bbox(rec["raw"]) if fmt == "bbox" else parse_boxes(rec["raw"]))
    for b in parsed:
        # 塊內(已放大)座標 → 原尺寸塊內座標；丟掉尺寸明顯不合理的框
        l, t, w, h = b["left"] / scale, b["top"] / scale, b["width"] / scale, b["height"] / scale
        if w < 4 or h < 4 or w > tile / 2 or h > tile / 2:
            continue
        l = max(0.0, min(l, tile)); t = max(0.0, min(t, tile))
        w = min(w, tile - l); h = min(h, tile - t)
        if w < 4 or h < 4:
            continue
        # 貼著塊邊的框多半是被切到一半的字；只要那條邊不是整張圖的邊界，鄰塊一定有完整版，這裡丟掉
        touches = ((l <= 1 and x0 > 0) or (t <= 1 and y0 > 0) or
                   (l + w >= tile - 1 and x0 + tile < W_img) or (t + h >= tile - 1 and y0 + tile < H_img))
        if touches:
            continue
        boxes.append(dict(text=b["text"], left=round(l + x0, 1), top=round(t + y0, 1),
                          width=round(w, 1), height=round(h, 1), conf=b["conf"], tile=[x0, y0]))
    return boxes


def dedup(boxes, tile, iou_th=0.3):
    """重疊區去重：同一字被兩塊都抓到時，保留離所屬塊中心較近的那個（塊邊緣的框較不準）。"""
    def margin(b):
        cx, cy = b["left"] + b["width"] / 2, b["top"] + b["height"] / 2
        tx, ty = b["tile"]
        return min(cx - tx, tx + tile - cx, cy - ty, ty + tile - cy)
    ordered = sorted(boxes, key=margin, reverse=True)
    kept = []
    for b in ordered:
        if any(box_iou(b, k) >= iou_th for k in kept):
            continue
        kept.append(b)
    return kept


def draw_overlay(img_path, boxes, out_png, gt_boxes=None):
    img = Image.open(img_path).convert("RGB")
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, 18)
    if gt_boxes:
        for g in gt_boxes:
            d.rectangle([g["left"], g["top"], g["left"] + g["width"], g["top"] + g["height"]], outline=(0, 170, 0), width=2)
    for b in boxes:
        x1, y1 = b["left"], b["top"]
        d.rectangle([x1, y1, x1 + b["width"], y1 + b["height"]], outline=(255, 0, 0), width=2)
        d.text((x1, max(0, y1 - 20)), b["text"], fill=(200, 0, 0), font=font)
    img.save(out_png)


def run_map(gt_id, args):
    img_path = os.path.join(ORIGINAL_DIR, f"{gt_id}.jpg")
    if not os.path.exists(img_path):
        print(f"[{gt_id}] 沒有原圖，跳過：{img_path}")
        return None
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    tiles = make_tiles(W, H, args.tile, args.overlap, args.region)
    out_dir = os.path.join(OUT_ROOT, cfg_name(args.tile, args.scale, args.overlap, args.effort, args.deployment))
    tile_dir = os.path.join(out_dir, gt_id, "tiles")
    print(f"[{gt_id}] {W}x{H}, tile={args.tile} scale={args.scale} overlap={args.overlap} → {len(tiles)} 塊, workers={args.workers}")

    all_boxes, stats = [], dict(calls=0, cached=0, seconds=0.0, prompt_tokens=0, completion_tokens=0, errors=0)
    t_start = time.time()

    def job(xy):
        x0, y0 = xy
        return process_tile(img, x0, y0, args.tile, args.scale, os.path.join(tile_dir, f"{x0}_{y0}.json"), args)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(job, xy): xy for xy in tiles}
        done = 0
        for fut in as_completed(futs):
            xy = futs[fut]
            done += 1
            try:
                rec, from_cache = fut.result()
            except Exception as e:
                stats["errors"] += 1
                print(f"  [{gt_id}] 塊 {xy} 失敗：{e}")
                continue
            all_boxes.extend(tile_boxes(rec, img.size))
            if from_cache:
                stats["cached"] += 1
            else:
                stats["calls"] += 1
                stats["seconds"] += rec["seconds"]
                stats["prompt_tokens"] += rec["usage"].get("prompt_tokens") or 0
                stats["completion_tokens"] += rec["usage"].get("completion_tokens") or 0
            if done % 10 == 0 or done == len(tiles):
                print(f"  [{gt_id}] {done}/{len(tiles)} 塊  框數 {len(all_boxes)}  已用 {time.time() - t_start:.0f}s")

    kept = dedup(all_boxes, args.tile, args.dedup_iou)
    result = dict(map=gt_id, image_size=[W, H], cfg=cfg_name(args.tile, args.scale, args.overlap, args.effort, args.deployment),
                  tile=args.tile, scale=args.scale, overlap=args.overlap, effort=args.effort,
                  prompt_version=prompt_version(args.deployment), region=args.region,
                  n_tiles=len(tiles), n_raw_boxes=len(all_boxes), n_boxes=len(kept), stats=stats, boxes=kept)
    suffix = "_region" if args.region else ""
    if not args.dry_run:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{gt_id}{suffix}.json"), "w", encoding="utf8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"[{gt_id}] 完成：{len(all_boxes)} 框 → 去重後 {len(kept)}；API 呼叫 {stats['calls']}（快取 {stats['cached']}、失敗 {stats['errors']}）"
          f"，tokens {stats['prompt_tokens']}+{stats['completion_tokens']}，wall {time.time() - t_start:.0f}s")
    if args.draw and not args.dry_run:
        gt = load_gt_boxes(gt_id) if args.eval else None
        draw_overlay(img_path, kept, os.path.join(out_dir, f"{gt_id}{suffix}_overlay.png"), gt)
    return result


# ---------------------------------------------------------------- 偵測層評估（對 GT 字框）
def load_gt_boxes(gt_id):
    import xml.etree.ElementTree as ET
    root = ET.parse(os.path.join(GT_DIR, f"{gt_id}.xml")).getroot()
    out = []
    for b in root.iter("box"):
        out.append(dict(text=b.find("label").text, left=int(b.get("left")), top=int(b.get("top")),
                        width=int(b.get("width")), height=int(b.get("height")), id=b.get("id")))
    return out


def match_boxes(pred, gt, iou_th=0.35):
    """貪婪一對一配對（GT 依序找 IoU 最大的未用 pred）。回傳 [(gt, pred_or_None, iou)]。"""
    used = set()
    rows = []
    for g in gt:
        best, bi = 0.0, None
        for i, p in enumerate(pred):
            if i in used:
                continue
            v = box_iou(p, g)
            if v > best:
                best, bi = v, i
        if bi is not None and best >= iou_th:
            used.add(bi)
            rows.append((g, pred[bi], best))
        else:
            rows.append((g, None, best))
    return rows


def eval_detection(result, gt_boxes, region=None):
    pred = result["boxes"]
    gt = gt_boxes
    if region:
        rx, ry, rw, rh = region
        inside = lambda b: b["left"] >= rx and b["top"] >= ry and b["left"] + b["width"] <= rx + rw and b["top"] + b["height"] <= ry + rh
        gt = [g for g in gt if inside(g)]
        pred = [p for p in pred if inside(p)]
    out = {}
    for th in (0.35, 0.5):
        rows = match_boxes(pred, gt, th)
        m = [r for r in rows if r[1] is not None]
        txt_ok = sum(1 for g, p, _ in m if p["text"] == g["text"])
        out[f"iou{th}"] = dict(n_gt=len(gt), n_pred=len(pred), matched=len(m),
                               recall=round(len(m) / len(gt), 4) if gt else None,
                               precision=round(len(m) / len(pred), 4) if pred else None,
                               text_acc=round(txt_ok / len(m), 4) if m else None,
                               mean_iou=round(sum(r[2] for r in m) / len(m), 4) if m else None)
    return out


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", nargs="+", required=True, help="gt_id 列表（xml_revised / map/original 的檔名）")
    ap.add_argument("--tile", type=int, default=768, help="原生裁切邊長（px）")
    ap.add_argument("--scale", type=int, default=1, help="送出前放大倍率；tile*scale 短邊不可超過 768")
    ap.add_argument("--overlap", type=int, default=None, help="重疊像素，預設 tile 的 25%%")
    ap.add_argument("--effort", default="medium", help="reasoning_effort（medium/high/low；'none' 表示不傳）")
    ap.add_argument("--deployment", default=None, help="Azure deployment 名，預設用 config.DEPLOYMENT_NAME")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--max-retries", type=int, default=getattr(config, "LLM_MAX_RETRIES", 10))
    ap.add_argument("--backoff", type=float, default=getattr(config, "LLM_BACKOFF_FACTOR", 2))
    ap.add_argument("--dedup-iou", type=float, default=0.3, help="跨塊去重 IoU 門檻（同字跨塊抖動實測 0.3–0.5）")
    ap.add_argument("--region", type=int, nargs=4, metavar=("X", "Y", "W", "H"), help="只跑與此區域相交的塊（smoke test）")
    ap.add_argument("--force", action="store_true", help="忽略快取重新呼叫")
    ap.add_argument("--dry-run", action="store_true", help="不呼叫 API，只列切塊數")
    ap.add_argument("--draw", action="store_true", help="輸出 overlay PNG")
    ap.add_argument("--eval", action="store_true", help="對 xml_revised 的 GT 算偵測層 P/R")
    args = ap.parse_args()
    if args.overlap is None:
        args.overlap = args.tile // 4
    if args.effort in ("none", "", None):
        args.effort = None
    if args.tile * args.scale > 768:
        print(f"警告：tile*scale = {args.tile * args.scale} > 768，API 會縮圖，座標系可能不一致。")

    summary = []
    for gt_id in args.maps:
        res = run_map(gt_id, args)
        if res is None:
            continue
        if args.eval:
            ev = eval_detection(res, load_gt_boxes(gt_id), args.region)
            res["eval"] = ev
            print(f"[{gt_id}] 偵測層 @0.35: R={ev['iou0.35']['recall']} P={ev['iou0.35']['precision']} "
                  f"txt={ev['iou0.35']['text_acc']} IoU={ev['iou0.35']['mean_iou']} "
                  f"| @0.5: R={ev['iou0.5']['recall']} P={ev['iou0.5']['precision']} txt={ev['iou0.5']['text_acc']}")
            summary.append((gt_id, ev))
    if summary:
        out_dir = os.path.join(OUT_ROOT, cfg_name(args.tile, args.scale, args.overlap, args.effort, args.deployment))
        with open(os.path.join(out_dir, "eval_summary.json"), "w", encoding="utf8") as f:
            json.dump({k: v for k, v in summary}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
