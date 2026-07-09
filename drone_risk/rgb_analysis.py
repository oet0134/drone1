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


def _centers(cands, W, H):
    return [(((c0 + bw / 2) / W), ((r0 + bh / 2) / H))
            for (r0, c0, bh, bw) in [c["bbox"] for c in cands]]


def baseline_from(image_bytes: bytes) -> dict:
    """감시 기준(정상 상태) 등록: 현재 이상 지점들의 위치를 기준으로 저장."""
    img, W, H, cands, quality = _detect(image_bytes)
    return {"centers": _centers(cands, W, H), "count": len(cands), "quality": quality}


def monitor_frame(image_bytes: bytes, baseline_centers, tol=0.05) -> dict:
    """감시 한 컷: 기준 대비 '새로 생긴' 이상만 골라 경보 여부 판단.

    tol = 같은 위치로 볼 허용 반경(이미지 대각 비율). 카메라 흔들림 흡수.
    """
    img, W, H, cands, quality = _detect(image_bytes)
    cur = _centers(cands, W, H)
    new_flags = []
    for (cx, cy) in cur:
        is_new = all((cx - bx) ** 2 + (cy - by) ** 2 > tol * tol
                     for (bx, by) in baseline_centers)
        new_flags.append(is_new)

    ann = img.copy()
    draw = ImageDraw.Draw(ann)
    for c, isnew in zip(cands, new_flags):
        r0, c0, bh, bw = c["bbox"]
        color = (255, 40, 40) if isnew else (150, 150, 150)   # 새 이상=빨강, 기존=회색
        draw.rectangle([c0, r0, c0 + bw, r0 + bh], outline=color, width=4 if isnew else 2)
    buf = io.BytesIO()
    ann.save(buf, format="JPEG", quality=80)

    new_count = sum(new_flags)
    return {
        "new_count": new_count, "total": len(cands),
        "status": "alert" if new_count >= 1 else "normal",
        "photo_quality": quality, "annotated_jpg": buf.getvalue(),
    }
