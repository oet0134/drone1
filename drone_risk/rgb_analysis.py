"""일반(RGB) 사진 결함 분석 — AI 제안 + 사람 확정(Human-in-the-loop).

★ AI 등급은 '참고용 제안'이다. 단순 영상처리로는 진짜 균열과 창틀·줄눈을
   완벽히 구분할 수 없으므로(실측 확인됨), AI가 대략적인 제안 등급과 의심 지점만
   주고, 최종 등급은 사람이 확정한다(그 확정이 학습 데이터가 되고, AI 제안 대비
   사람 확정을 비교해 모델 정확도를 추적한다).

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

    # 길고·길쭉하고·진한 것만 후보로. 정규화 강도(promn)와 함께 보관.
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
        promn = (length / diag) * (dmean / 255.0)          # 정규화 강도 0~1
        cands.append({"bbox": (r0, c0, bh, bw), "prom": length * dmean, "promn": promn})
    cands.sort(key=lambda x: -x["prom"])
    cands = cands[:cfg["max_candidates"]]

    contrast = float(gray_raw.std())
    quality = "low" if contrast < cfg["min_contrast"] else "ok"

    # AI 제안 등급(참고용): 가장 강한 후보의 강도 + 후보 수 → concern → A~E
    s = cfg["suggest"]
    top = max((c["promn"] for c in cands), default=0.0)
    concern = _clamp(s["w_top"] * _norm(top, s["top_ref"])
                     + s["w_cnt"] * _norm(len(cands), s["cnt_ref"]))
    if quality == "low":
        suggested = "HOLD"                                  # 흐려서 판단 곤란
    else:
        suggested = to_grade(RiskScore("photo", round(concern, 3), 1.0, {},
                                       EngineMeta("rgb-assist", "0.3.0")))

    # 의심 지점에 빨간 박스
    ann = img.copy()
    draw = ImageDraw.Draw(ann)
    for c in cands:
        r0, c0, bh, bw = c["bbox"]
        draw.rectangle([c0, r0, c0 + bw, r0 + bh], outline=(255, 40, 40), width=3)
    buf = io.BytesIO()
    ann.save(buf, format="JPEG", quality=80)

    return {
        "suggested_grade": suggested,     # AI 제안(참고용) — 사람이 확정/수정
        "concern": round(concern, 3),
        "num_candidates": len(cands),     # 사람이 검토할 의심 지점 수
        "photo_quality": quality,         # "ok" | "low"
        "annotated_jpg": buf.getvalue(),
        "engine": {"type": "rgb-assist", "version": "0.3.0"},
    }
