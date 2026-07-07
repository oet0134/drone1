"""일반(RGB) 사진 결함 '후보 표시' — 사람 판단 보조(Human-in-the-loop).

★ 자동 등급을 매기지 않는다. 단순 영상처리로는 진짜 균열과 창틀·벽돌 줄눈·
   구조선을 구분할 수 없기 때문(실측으로 확인됨). 대신 '균열처럼 보이는
   길고 진한 어두운 선'만 보수적으로 골라 빨간 박스로 표시하고, 최종 등급은
   사람이 지정한다(그 판단이 학습 데이터로 쌓인다).

재사용: 연결요소 라벨링(imaging), PCA 길쭉함.
"""
from __future__ import annotations
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .config import RGB
from .imaging import label_components


def _pca_elong(cells) -> float:
    """이상 영역의 길쭉함(장축/단축) — 회전 불변. 선형(균열형)이면 크게 나온다."""
    pts = np.array(list(cells), dtype=float)
    if len(pts) < 5:
        return 1.0
    pts -= pts.mean(axis=0)
    ev = np.linalg.eigvalsh(np.cov(pts.T))
    return (max(ev[1], 1e-9) / max(ev[0], 1e-9)) ** 0.5


def analyze_photo(image_bytes: bytes) -> dict:
    cfg = RGB
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 속도를 위해 축소
    w, h = img.size
    scale = cfg["max_dim"] / max(w, h)
    if scale < 1:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    W, H = img.size

    gray_img = img.convert("L")
    gray_raw = np.asarray(gray_img, dtype=float)                                  # 품질 판정용
    gray = np.asarray(gray_img.filter(ImageFilter.GaussianBlur(1)), dtype=float)  # 약한 노이즈 제거
    r = max(2, int(min(W, H) * cfg["blur_radius_frac"]))
    bg = np.asarray(gray_img.filter(ImageFilter.GaussianBlur(r)), dtype=float)
    dark = bg - gray                                                             # 주변보다 어두운 정도
    mask = dark >= cfg["dark_thresh"]

    comps = label_components(mask, min_px=cfg["min_comp_px"])
    diag = (W * W + H * H) ** 0.5
    min_len = cfg["min_len_frac"] * diag

    # 길고(min_len↑) 길쭉하고(elong↑) 진한(dark↑) 것만 후보로. 진하고 긴 순 상위 N개.
    cands = []
    for c in comps:
        r0, c0, bh, bw = c["bbox"]
        length = (bh * bh + bw * bw) ** 0.5
        if length < min_len:
            continue
        if _pca_elong(c["cells"]) < cfg["crack_elong"]:
            continue
        idx = np.array(list(c["cells"]))
        dmean = float(dark[idx[:, 0], idx[:, 1]].mean())
        cands.append({"bbox": (r0, c0, bh, bw), "prom": length * dmean})
    cands.sort(key=lambda x: -x["prom"])
    cands = cands[:cfg["max_candidates"]]

    contrast = float(gray_raw.std())
    quality = "low" if contrast < cfg["min_contrast"] else "ok"

    # 의심 지점에 빨간 박스
    ann = img.copy()
    draw = ImageDraw.Draw(ann)
    for c in cands:
        r0, c0, bh, bw = c["bbox"]
        draw.rectangle([c0, r0, c0 + bw, r0 + bh], outline=(255, 40, 40), width=3)
    buf = io.BytesIO()
    ann.save(buf, format="JPEG", quality=80)

    return {
        "num_candidates": len(cands),     # 사람이 검토할 의심 지점 수 (등급 아님)
        "photo_quality": quality,         # "ok" | "low"(대비 낮아 판단 곤란)
        "annotated_jpg": buf.getvalue(),
        "engine": {"type": "rgb-assist", "version": "0.2.0"},
    }
