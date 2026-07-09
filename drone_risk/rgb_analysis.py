"""일반(RGB) 사진 결함 분석 — AI 제안 + 사람 확정, 그리고 상시 감시(변화 감지).

★ AI 등급은 '참고용 제안'이다. 단순 영상처리로는 진짜 균열과 창틀·줄눈을
   완벽히 구분할 수 없으므로, AI가 대략적 제안과 의심 지점만 주고 사람이 확정한다.

상시 감시(monitor): 고정 카메라의 '정상 기준' 대비 새로 생긴 이상만 잡는다.
   → 항상 있던 창틀·줄눈은 기준에 포함돼 무시되고, 진짜 변화만 경보된다.

재사용: 연결요소 라벨링(imaging), A~E 등급 매핑(risk_engine.to_grade).
"""
from __future__ import annotations
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .config import RGB
from .imaging import label_components
from .contracts import RiskScore, EngineMeta
from .risk_engine import to_grade, _clamp


def _norm(x, ref):
    return max(0.0, min(1.0, x / ref)) if ref > 0 else 0.0


def _pca_elong(cells) -> float:
    pts = np.array(list(cells), dtype=float)
    if len(pts) < 5:
        return 1.0
    pts -= pts.mean(axis=0)
    ev = np.linalg.eigvalsh(np.cov(pts.T))
    return (max(ev[1], 1e-9) / max(ev[0], 1e-9)) ** 0.5


def _detect(image_bytes: bytes):
    """공통 탐지: 이미지 → (PIL RGB, W, H, 후보목록, 품질).

    후보 = 길고·길쭉하고·진한 어두운 선(균열형). 진하고 긴 순 상위 N개.
    """
    cfg = RGB
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    scale = cfg["max_dim"] / max(w, h)
    if scale < 1:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    W, H = img.size

    gray_img = img.convert("L")
    gray_raw = np.asarray(gray_img, dtype=float)
    gray = np.asarray(gray_img.filter(ImageFilter.GaussianBlur(1)), dtype=float)
    r = max(2, int(min(W, H) * cfg["blur_radius_frac"]))
    bg = np.asarray(gray_img.filter(ImageFilter.GaussianBlur(r)), dtype=float)
    dark = bg - gray
    mask = dark >= cfg["dark_thresh"]

    comps = label_components(mask, min_px=cfg["min_comp_px"])
    diag = (W * W + H * H) ** 0.5
    min_len = cfg["min_len_frac"] * diag

    cands = []
    for c in comps:
        r0, c0, bh, bw = c["bbox"]
        length = (bh * bh + bw * bw) ** 0.5
        if length < min_len or _pca_elong(c["cells"]) < cfg["crack_elong"]:
            continue
        idx = np.array(list(c["cells"]))
        dmean = float(dark[idx[:, 0], idx[:, 1]].mean())
        cands.append({"bbox": (r0, c0, bh, bw), "prom": length * dmean,
                      "promn": (length / diag) * (dmean / 255.0)})
    cands.sort(key=lambda x: -x["prom"])
    cands = cands[:cfg["max_candidates"]]
    quality = "low" if float(gray_raw.std()) < cfg["min_contrast"] else "ok"
    return img, W, H, cands, quality


def analyze_photo(image_bytes: bytes) -> dict:
    """단발 분석: 의심 지점 표시 + AI 제안 등급(사람이 확정)."""
    img, W, H, cands, quality = _detect(image_bytes)
    s = RGB["suggest"]
    top = max((c["promn"] for c in cands), default=0.0)
    concern = _clamp(s["w_top"] * _norm(top, s["top_ref"])
                     + s["w_cnt"] * _norm(len(cands), s["cnt_ref"]))
    suggested = "HOLD" if quality == "low" else to_grade(
        RiskScore("photo", round(concern, 3), 1.0, {}, EngineMeta("rgb-assist", "0.3.0")))

    ann = img.copy()
    draw = ImageDraw.Draw(ann)
    for c in cands:
        r0, c0, bh, bw = c["bbox"]
        draw.rectangle([c0, r0, c0 + bw, r0 + bh], outline=(255, 40, 40), width=3)
    buf = io.BytesIO()
    ann.save(buf, format="JPEG", quality=80)
    return {
        "suggested_grade": suggested, "concern": round(concern, 3),
        "num_candidates": len(cands), "photo_quality": quality,
        "annotated_jpg": buf.getvalue(),
        "engine": {"type": "rgb-assist", "version": "0.3.0"},
    }


def _mon_gray(image_bytes):
    """감시용 축소·흐림 흑백 이미지 + 원본(색) 반환."""
    cfg = RGB["monitor"]
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    s = cfg["ref_dim"] / max(w, h)
    if s < 1:
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))))
    gray = img.convert("L").filter(ImageFilter.GaussianBlur(cfg["blur"]))
    return img, gray


def monitor_baseline(image_bytes: bytes) -> dict:
    """감시 기준(정상 상태) 등록: 기준 화면(흑백)을 PNG로 보관."""
    img, gray = _mon_gray(image_bytes)
    buf = io.BytesIO()
    gray.save(buf, format="PNG")     # 무손실(정확한 비교용)
    contrast = float(np.asarray(gray, dtype=float).std())
    return {"gray_png": buf.getvalue(),
            "quality": "low" if contrast < RGB["min_contrast"] else "ok"}


def monitor_check(image_bytes: bytes, baseline_png: bytes) -> dict:
    """감시 한 컷: 기준 대비 '새로 어두워진 곳'을 직접 비교.

    - 국부적 변화(낙서·균열 등) → 경보(빨간 박스)
    - 화면 전체가 바뀜(면적 비율↑) → '카메라 이동/조명 변화'로 따로 처리(오경보 방지)
    """
    cfg = RGB["monitor"]
    img, gray = _mon_gray(image_bytes)
    W, H = img.size
    cur = np.asarray(gray, dtype=float)
    base_img = Image.open(io.BytesIO(baseline_png)).convert("L")
    if base_img.size != (W, H):
        base_img = base_img.resize((W, H))
    base = np.asarray(base_img, dtype=float)

    darker = base - cur                       # 기준보다 어두워진 정도(새 표시=어두움)
    mask = darker >= cfg["diff_thresh"]
    changed_ratio = float(mask.mean())
    comps = label_components(mask, min_px=cfg["min_region_px"])

    ann = img.copy()
    draw = ImageDraw.Draw(ann)
    for c in comps:
        r0, c0, bh, bw = c["bbox"]
        draw.rectangle([c0, r0, c0 + bw, r0 + bh], outline=(255, 40, 40), width=3)
    buf = io.BytesIO()
    ann.save(buf, format="JPEG", quality=80)

    if changed_ratio > cfg["global_ratio"]:
        status, new_count = "moved", 0        # 화면 전체 변화(카메라 이동?)
    elif comps:
        status, new_count = "alert", len(comps)
    else:
        status, new_count = "normal", 0
    return {"status": status, "new_count": new_count,
            "changed_ratio": round(changed_ratio, 3),
            "annotated_jpg": buf.getvalue()}
