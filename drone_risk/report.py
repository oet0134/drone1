"""보고서 패킷 빌더 — 등급 결과를 백엔드 제출용 구조로 패킹.

무거운 이미지는 오브젝트 스토리지에 두고 패킷엔 참조만 (전송 경량화).
"""
from __future__ import annotations
import hashlib
import json
from .contracts import FeatureVector, RiskScore, Grade

GRADE_ACTION = {
    "E": "즉시 하부 통제·낙하물 방지·긴급 점검 (탈락 임박/활성 진행)",
    "D": "단기 내 정밀안전진단 (명백한 박리 또는 유의 변형)",
    "C": "정기 모니터링·재촬영 (초기 들뜸 징후)",
    "B": "경과 관찰 (경미한 징후)",
    "A": "차기 정기점검 (유의 결함 없음)",
    "HOLD": "조건 개선 후 재촬영 (판정 신뢰도 부족)",
}


def build_zone(fv: FeatureVector, rs: RiskScore, grade: Grade) -> dict:
    return {
        "zone_id": fv.zone_id,
        "geo": fv.geo,
        "grade": grade,
        "score": rs.score,
        "confidence": rs.confidence,
        "indicators": {
            "thermal": {
                "dt_max_c": fv.thermal.dt_max,
                "anomaly_area_m2": round(fv.thermal.anomaly_area, 2),
                "pattern": fv.thermal.pattern,
                "valid": fv.thermal.valid,
            },
            "ultrasonic": {
                "bulge_max_mm": fv.ultra.bulge_max,
                "bulge_area_m2": round(fv.ultra.bulge_area, 2),
                "profile_disc": fv.ultra.profile_disc,
            },
        },
        "contributions": rs.contributions,
        "action": GRADE_ACTION[grade],
        "evidence": [f"obj://{fv.zone_id}/thermal_overlay.png"],
        "engine": {"type": rs.engine.type, "version": rs.engine.version},
    }


def build_report(building_id: str, captured_at: str, drone: dict,
                 overall: Grade, zones: list) -> dict:
    report = {
        "report_id": f"{building_id}@{captured_at}",   # 멱등키(재전송 중복 방지)
        "building_id": building_id,
        "captured_at": captured_at,
        "drone": drone,
        "overall_grade": overall,
        "overall_action": GRADE_ACTION[overall],
        "zones": zones,
    }
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True).encode("utf-8")
    report["integrity"] = {"hash": "sha256:" + hashlib.sha256(payload).hexdigest()}
    return report
