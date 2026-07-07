"""온보드 파이프라인 오케스트레이션: 센서 샘플 → 지표 → 등급 → 보고서."""
from __future__ import annotations
from .features import build_feature_vector
from .risk_engine import RuleBasedEngine, to_grade, worst_grade
from .report import build_zone, build_report


def assess_building(scan: dict, engine=None) -> dict:
    """scan = {building_id, captured_at, drone, zones:[sample,...]}"""
    engine = engine or RuleBasedEngine()
    zone_reports, grades = [], []
    for sample in scan["zones"]:
        fv = build_feature_vector(sample)
        rs = engine.score(fv)
        grade = to_grade(rs)
        grades.append(grade)
        zone_reports.append(build_zone(fv, rs, grade))

    overall = worst_grade(grades)
    return build_report(
        scan["building_id"], scan["captured_at"], scan["drone"],
        overall, zone_reports,
    )
