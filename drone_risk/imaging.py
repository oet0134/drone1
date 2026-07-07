"""영상 처리 기반 — 카메라 모델(GSD), 디트렌딩, 연결요소, 센서 정합.

실데이터 추출의 핵심 난제를 다룬다:
  · 픽셀 → 미터 변환 (GSD): 면적·구배를 실단위로
  · 디트렌딩: 일사 구배(열화상)·드론 틸트(깊이)를 제거해야 '국부 건전면 대비' 이상이 보임
  · 정합: 열화상과 초음파는 해상도·FOV가 달라 공통 좌표로 맞춰야 동시성 판정 가능
"""
from __future__ import annotations
import math
from collections import deque
import numpy as np


def gsd_m_per_px(distance_m: float, fov_deg: float, px: int) -> float:
    """지상표본거리: 픽셀 1칸이 덮는 외벽 폭 [m/px]."""
    width_m = 2.0 * distance_m * math.tan(math.radians(fov_deg) / 2.0)
    return width_m / px


def detrend_plane(frame: np.ndarray):
    """1차 평면(z = a·x + b·y + c)을 최소제곱 피팅해 빼고 잔차를 반환.

    열화상: 일사로 인한 완만한 온도 구배 제거 → 잔차 = 국부 건전면 대비 ΔT.
    깊이:   드론이 비스듬히 본 평면의 기울기 제거 → 잔차 = 평면 대비 돌출.
    """
    h, w = frame.shape
    ys, xs = np.mgrid[0:h, 0:w]
    A = np.column_stack([xs.ravel(), ys.ravel(), np.ones(h * w)])
    coef, *_ = np.linalg.lstsq(A, frame.ravel(), rcond=None)
    trend = (A @ coef).reshape(h, w)
    return frame - trend, trend


def label_components(mask: np.ndarray, min_px: int = 4):
    """불리언 마스크를 4-연결 영역들로 라벨링. 작은 잡음(min_px 미만)은 버림.

    반환: [{'cells': set[(r,c)], 'area_px', 'bbox': (r0,c0,h,w)}, ...]
    """
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            if len(cells) < min_px:
                continue
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            bbox = (min(rs), min(cs), max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)
            comps.append({"cells": set(cells), "area_px": len(cells), "bbox": bbox})
    return comps


def to_norm_grid(cells, shape, n: int = 32) -> np.ndarray:
    """이상 셀들을 정규화 격자(n×n, 외벽 0~1 좌표)에 투영 → 센서 간 정합용."""
    h, w = shape
    out = np.zeros((n, n), dtype=bool)
    for (r, c) in cells:
        v = min(n - 1, int(r / h * n))
        u = min(n - 1, int(c / w * n))
        out[v, u] = True
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0
