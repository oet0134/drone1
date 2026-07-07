"""예시 건물 데이터 생성 — 대시보드를 다양한 등급으로 채운다.

frame_sensors의 프레임 헬퍼를 재사용해, 구역 종류(kind)만 조합하면
서로 다른 위험 분포를 가진 건물들이 만들어진다.
"""
from __future__ import annotations
from .frame_sensors import (CAM, GSD_TH, GSD_UL, _thermal_base, _depth_base,
                            _blob, _line, _zone)


def _kind_frames(kind: str):
    """구역 종류 → (열화상 프레임, 깊이 프레임, 유효성)."""
    valid = (kind != "bad_capture")
    th = _thermal_base(valid=valid)
    ul = _depth_base()
    if kind == "sound":
        pass                                              # 건전 → A
    elif kind == "minor":
        _blob(th, 0.50, 0.50, 0.22, 3.2, GSD_TH)          # 경미 → B
    elif kind == "debond":
        _blob(th, 0.45, 0.50, 0.45, 6.5, GSD_TH)          # 진행된 박리 → D
    elif kind == "imminent":
        _blob(th, 0.62, 0.45, 0.42, 6.5, GSD_TH)          # 탈락 임박 → E
        _blob(ul, 0.62, 0.45, 0.42, -22.0, GSD_UL)        # 열+기하 동시
    elif kind == "crack":
        _line(th, 0.30, 0.25, 0.55, 0.72, 0.07, 5.0, GSD_TH)   # 균열 → C
    elif kind == "bad_capture":
        _blob(th, 0.45, 0.50, 0.28, 0.6, GSD_TH)          # 흐린날 → 보류
    return th, ul, valid


def _make_zone(kind, zid, height, public, cluster, facade, geo):
    th, ul, valid = _kind_frames(kind)
    return _zone(zid, th, ul, valid, height, public, cluster, facade, geo)


def _building(name, captured_at, zone_defs):
    zones = []
    for i, (kind, zid, h, pub, cl, fac) in enumerate(zone_defs):
        geo = {"lat": round(37.50 + i * 0.0007, 5),
               "lon": round(127.03 + i * 0.0009, 5),
               "alt_m": float(h), "facade": fac}
        zones.append(_make_zone(kind, zid, h, pub, cl, fac, geo))
    return {"building_id": name, "captured_at": captured_at,
            "drone": {"id": "DRN-07", "fw": "1.4.2"}, "zones": zones}


# 등급이 고루 분포하도록 큐레이션한 예시 건물들 (A·B·C·D·보류)
_SPEC = [
    ("강남 A타워", "2026-06-20T09:10:00+09:00", [
        ("sound", "1F-동", 5, False, 0.0, "E"),
        ("debond", "6F-남", 26, True, 0.4, "S"),
        ("imminent", "9F-북", 34, True, 0.5, "N")]),            # E
    ("역삼 비즈센터", "2026-06-20T10:25:00+09:00", [
        ("sound", "2F-서", 8, False, 0.0, "W"),
        ("sound", "5F-남", 20, True, 0.1, "S"),
        ("crack", "7F-동", 28, True, 0.3, "E")]),               # C
    ("삼성 플라자", "2026-06-20T11:40:00+09:00", [
        ("sound", "3F-북", 12, False, 0.0, "N"),
        ("debond", "8F-남", 32, True, 0.45, "S")]),             # D
    ("테헤란 리버뷰", "2026-06-20T13:05:00+09:00", [
        ("sound", "1F-남", 4, False, 0.0, "S"),
        ("sound", "4F-동", 16, False, 0.0, "E"),
        ("sound", "6F-서", 24, False, 0.05, "W")]),             # A
    ("서초 그린타워", "2026-06-20T14:20:00+09:00", [
        ("sound", "2F-남", 8, True, 0.0, "S"),
        ("crack", "5F-북", 20, True, 0.3, "N"),
        ("bad_capture", "7F-서", 28, True, 0.1, "W")]),         # 보류
    ("논현 스카이", "2026-06-20T15:35:00+09:00", [
        ("debond", "9F-남", 36, True, 0.5, "S"),
        ("imminent", "11F-북", 44, True, 0.6, "N"),
        ("sound", "3F-동", 12, False, 0.0, "E"),
        ("crack", "6F-서", 24, True, 0.3, "W")]),               # E
    ("방배 코트", "2026-06-20T16:50:00+09:00", [
        ("sound", "1F-남", 5, False, 0.0, "S"),
        ("minor", "4F-동", 16, True, 0.2, "E"),
        ("minor", "6F-북", 22, True, 0.25, "N")]),              # B (경미)
    ("잠원 하이츠", "2026-06-20T18:05:00+09:00", [
        ("sound", "2F-서", 8, False, 0.0, "W"),
        ("debond", "7F-남", 28, True, 0.4, "S"),
        ("bad_capture", "5F-동", 20, True, 0.1, "E")]),         # D
]


def sample_buildings():
    return [_building(name, cap, zdefs) for name, cap, zdefs in _SPEC]
