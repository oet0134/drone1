"""지표 추출 — 실데이터 버전.

입력: 고해상도 열화상 프레임(°C)·깊이 프레임(mm) + 카메라 메타(거리·FOV).
출력: FeatureVector (계약 불변 — 규칙/ML 엔진이 그대로 받는다).

처리:
  열화상: 평면 디트렌딩(일사 구배 제거) → 임계 → 연결요소(잡음 제거)
          → ΔT_max / 면적(m²) / 구배(°C/cm) / 패턴(형태 분류)
  깊이:   평면 디트렌딩(틸트 제거) → 돌출 → bulge_max(mm) / 면적 / 단차
  융합:   두 이상 영역을 정규화 격자로 정합 → IoU = agree
"""
from __future__ import annotations
import numpy as np
from .contracts import ThermalFeat, UltraFeat, ContextFeat, FeatureVector
from .config import EXTRACT
from .imaging import (gsd_m_per_px, detrend_plane, label_components,
                      to_norm_grid, iou)


def _classify_pattern(comp) -> str:
    """이상 영역의 형태로 패턴 분류 — PCA 길쭉함(회전 불변)을 사용.

    바운딩박스 종횡비는 대각선 균열에서 실패하므로(박스가 정사각형에 가까움),
    픽셀 좌표의 주성분 고유값 비로 진짜 길쭉함을 잰다.
    """
    pts = np.array(list(comp["cells"]), dtype=float)
    if len(pts) < 5:
        return "moisture"
    pts -= pts.mean(axis=0)
    evals = np.linalg.eigvalsh(np.cov(pts.T))     # 오름차순 [작은, 큰]
    e2 = max(evals[0], 1e-9)
    e1 = max(evals[1], 1e-9)
    elong = (e1 / e2) ** 0.5                       # 장축/단축 비
    if elong >= 3.0:
        return "crack-linear"                      # 가늘고 긴 선형 → 균열
    if comp["area_px"] >= 8 and elong <= 2.2:
        return "debond"                            # 조밀한 덩어리 → 들뜸/박리
    return "moisture"                              # 흩어진 약한 이상 → 습기성


def extract_thermal(frame: np.ndarray, valid: bool, cam: dict) -> ThermalFeat:
    h, w = frame.shape
    gsd = gsd_m_per_px(cam["distance_m"], cam["fov_deg"], w)
    residual, _ = detrend_plane(frame)        # 국부 건전면 대비 ΔT

    thr = EXTRACT["thermal_cell_dt"]
    mask = np.abs(residual) >= thr
    comps = label_components(mask, min_px=EXTRACT["min_comp_px"])
    if not comps:
        return ThermalFeat(0.0, 0.0, 0, 0.0, "none", valid, set())

    cells = set().union(*[c["cells"] for c in comps])
    rr = np.array([p[0] for p in cells])
    cc = np.array([p[1] for p in cells])
    dt_max = float(np.max(np.abs(residual[rr, cc])))
    area = len(cells) * gsd * gsd

    # 잔차의 공간 구배 크기 → °C/cm (이상 영역 내 최대)
    gy, gx = np.gradient(residual)
    grad_mag = np.sqrt(gx * gx + gy * gy) / (gsd * 100.0)   # per cm
    gradient = float(np.max(grad_mag[rr, cc]))

    biggest = max(comps, key=lambda c: c["area_px"])
    pattern = _classify_pattern(biggest)
    return ThermalFeat(round(dt_max, 2), round(area, 3), len(comps),
                       round(gradient, 3), pattern, valid, cells)


def extract_ultra(depth: np.ndarray, cam: dict) -> UltraFeat:
    h, w = depth.shape
    gsd = gsd_m_per_px(cam["distance_m"], cam["fov_deg"], w)
    residual, _ = detrend_plane(depth)        # 틸트 제거. 음수 = 센서 쪽 돌출
    protrusion = -residual                     # 양수 = 돌출(mm)

    bulge_thr = EXTRACT["ultra_cell_bulge_mm"]
    mask = protrusion >= bulge_thr
    comps = label_components(mask, min_px=EXTRACT["min_comp_px"])
    cells = set().union(*[c["cells"] for c in comps]) if comps else set()

    bulge_max = float(np.max(protrusion[mask])) if cells else 0.0
    area = len(cells) * gsd * gsd

    # 거리 프로파일 단차: 인접 차가 임계 이상인 경계 픽셀 수
    gy, gx = np.gradient(depth)
    disc = int(np.sum(np.sqrt(gx * gx + gy * gy) >= EXTRACT["ultra_disc_mm"]))
    return UltraFeat(round(bulge_max, 2), round(area, 3), disc, cells)


def co_location(th: ThermalFeat, ul: UltraFeat,
                th_shape, ul_shape, n: int = 32) -> float:
    """해상도가 다른 두 이상 영역을 공통 정규화 격자로 정합 후 IoU.

    가정: 두 센서가 동일 구역을 같은 방향·FOV로 본다(공동 정합).
    실제 시스템은 외부 캘리브레이션(extrinsic)으로 정합해야 한다.
    """
    if not th.cells or not ul.cells:
        return 0.0
    a = to_norm_grid(th.cells, th_shape, n)
    b = to_norm_grid(ul.cells, ul_shape, n)
    return iou(a, b)


def build_feature_vector(sample: dict) -> FeatureVector:
    th_frame = np.asarray(sample["thermal"], dtype=float)
    ul_frame = np.asarray(sample["ultra"], dtype=float)
    cam = sample["cam"]

    th = extract_thermal(th_frame, sample["thermal_valid"], cam["thermal"])
    ul = extract_ultra(ul_frame, cam["ultra"])
    ctx = ContextFeat(
        height_m=sample["context"]["height_m"],
        over_public=sample["context"]["over_public"],
        cluster_density=sample["context"]["cluster_density"],
        facade=sample["context"]["facade"],
    )
    agree = co_location(th, ul, th_frame.shape, ul_frame.shape)
    return FeatureVector(sample["zone_id"], th, ul, ctx, agree, sample.get("geo", {}))
