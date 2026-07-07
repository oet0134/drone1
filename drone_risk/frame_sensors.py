"""현실적 프레임 생성기 — 실데이터 추출 파이프라인 검증용.

열화상(120×160, °C): 일사 구배 + 센서 노이즈 + 결함 열신호
깊이맵(30×40, mm):  드론 틸트(기운 평면) + 노이즈 + 돌출

결함은 좌표가 아니라 '신호'로 심는다 → features.py가 디트렌딩·임계·정합으로 직접 추출.
시나리오: ①건전 ②초기박리(열) ③탈락임박(열+깊이) ④흐린날(열무효) ⑤균열(선형패턴)
"""
from __future__ import annotations
import math
import numpy as np

# 공통 카메라: 표준거리 8m, FOV 50°
CAM = {"thermal": {"distance_m": 8.0, "fov_deg": 50.0},
       "ultra":   {"distance_m": 8.0, "fov_deg": 50.0}}

TH_H, TH_W = 120, 160
UL_H, UL_W = 30, 40

_rng = np.random.default_rng(42)


def _gsd(px, fov=50.0, dist=8.0):
    return 2.0 * dist * math.tan(math.radians(fov) / 2.0) / px


GSD_TH = _gsd(TH_W)   # ≈ 0.0466 m/px
GSD_UL = _gsd(UL_W)   # ≈ 0.187 m/px


def _blob(frame, u, v, sigma_m, amp, gsd):
    """정규화 좌표 (u,v)∈[0,1]에 가우시안 열/돌출 신호를 더한다."""
    h, w = frame.shape
    cx, cy = u * w, v * h
    sig = sigma_m / gsd
    ys, xs = np.mgrid[0:h, 0:w]
    frame += amp * np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sig ** 2)))


def _line(frame, u0, v0, u1, v1, sigma_m, amp, gsd):
    """선분(균열) 신호: 선까지 거리에 따른 가우시안 단면."""
    h, w = frame.shape
    p0 = np.array([u0 * w, v0 * h]); p1 = np.array([u1 * w, v1 * h])
    sig = sigma_m / gsd
    ys, xs = np.mgrid[0:h, 0:w]
    seg = p1 - p0
    L2 = float(seg @ seg)
    for r in range(h):
        for c in range(w):
            p = np.array([c, r])
            t = max(0.0, min(1.0, ((p - p0) @ seg) / L2))
            d = np.linalg.norm(p - (p0 + t * seg))
            frame[r, c] += amp * math.exp(-(d * d) / (2 * sig * sig))


def _thermal_base(valid=True):
    """일사 구배(좌→우 +6°C) + 노이즈. valid=False면 과도상태 없어 평탄."""
    ys, xs = np.mgrid[0:TH_H, 0:TH_W]
    grad = 6.0 if valid else 0.5            # 흐린날: 일사 구배 거의 없음
    f = 20.0 + grad * (xs / TH_W) + _rng.normal(0, 0.3, (TH_H, TH_W))
    return f


def _depth_base():
    """드론 오블리크 → 좌우 +80mm 틸트 평면 + 노이즈."""
    ys, xs = np.mgrid[0:UL_H, 0:UL_W]
    return 4000.0 + 80.0 * (xs / UL_W) + _rng.normal(0, 2.0, (UL_H, UL_W))


def _zone(zid, th, ul, valid, h_m, public, cluster, facade, geo):
    return {"zone_id": zid, "thermal": th, "thermal_valid": valid, "ultra": ul,
            "cam": CAM,
            "context": {"height_m": h_m, "over_public": public,
                        "cluster_density": cluster, "facade": facade},
            "geo": geo}


def building_scan() -> dict:
    # ① 건전: 구배+노이즈만
    z1 = _zone("east-1F", _thermal_base(), _depth_base(), True,
               4.0, False, 0.0, "E",
               {"lat": 37.5012, "lon": 127.0396, "alt_m": 4.0, "facade": "E"})

    # ② 진행된 박리: 큰 열 핫스팟(+6.5°C, 넓음), 깊이는 아직 평평(부풀음 전)
    th2 = _thermal_base()
    _blob(th2, 0.45, 0.50, 0.45, 6.5, GSD_TH)
    z2 = _zone("south-5F", th2, _depth_base(), True,
               28.0, True, 0.45, "S",
               {"lat": 37.5013, "lon": 127.0397, "alt_m": 28.0, "facade": "S"})

    # ③ 탈락 임박: 같은 위치(0.62,0.45)에 열(+6.5°C) + 돌출(22mm)
    th3 = _thermal_base(); ul3 = _depth_base()
    _blob(th3, 0.62, 0.45, 0.42, 6.5, GSD_TH)
    _blob(ul3, 0.62, 0.45, 0.42, -22.0, GSD_UL)     # 음수 = 센서 쪽 돌출
    z3 = _zone("north-7F", th3, ul3, True,
               30.0, True, 0.5, "N",
               {"lat": 37.5014, "lon": 127.0398, "alt_m": 30.0, "facade": "N"})

    # ④ 흐린날: 과도상태 부족 → 대비 미미 & valid=False → 보류
    th4 = _thermal_base(valid=False)
    _blob(th4, 0.45, 0.5, 0.28, 0.6, GSD_TH)        # 배면 박리 있어도 안 보임
    z4 = _zone("west-3F", th4, _depth_base(), False,
               16.0, True, 0.1, "W",
               {"lat": 37.5015, "lon": 127.0399, "alt_m": 16.0, "facade": "W"})

    # ⑤ 균열: 가늘고 긴 선형 열신호 → 패턴 분류 'crack-linear'
    th5 = _thermal_base()
    _line(th5, 0.30, 0.25, 0.55, 0.72, 0.07, 5.0, GSD_TH)
    z5 = _zone("south-2F", th5, _depth_base(), True,
               12.0, True, 0.3, "S",
               {"lat": 37.5016, "lon": 127.0400, "alt_m": 12.0, "facade": "S"})

    return {"building_id": "B-2026-0421",
            "captured_at": "2026-06-21T09:14:00+09:00",
            "drone": {"id": "DRN-07", "fw": "1.4.2"},
            "zones": [z1, z2, z3, z4, z5]}
