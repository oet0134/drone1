"""위험 평가 엔진 — 규칙 기반(MVP). RiskEngine 인터페이스 뒤에 숨겨
나중에 MLEngine으로 교체해도 나머지 파이프라인은 무수정.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from .contracts import FeatureVector, RiskScore, EngineMeta, Grade
from .config import GRADING


def _norm(x: float, ref: float) -> float:
    """선형 포화 정규화: ref에서 1로 포화, [0,1]."""
    if ref <= 0:
        return 0.0
    return max(0.0, min(1.0, x / ref))


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


class RiskEngine(ABC):
    @abstractmethod
    def score(self, fv: FeatureVector) -> RiskScore: ...


class RuleBasedEngine(RiskEngine):
    version = "0.3.0"

    def __init__(self, cfg=GRADING):
        self.cfg = cfg

    def score(self, fv: FeatureVector) -> RiskScore:
        cfg = self.cfg
        ref = cfg["ref"]

        # --- 센서별 부분 점수 ---
        ts = cfg["thermal_sub"]
        s_thermal = _clamp(
            ts["dt"]   * _norm(fv.thermal.dt_max, ref["dt_max_c"]) +
            ts["area"] * _norm(fv.thermal.anomaly_area, ref["anomaly_area_m2"]) +
            ts["grad"] * _norm(fv.thermal.gradient, ref["gradient_c"])
        ) * cfg["pattern_weight"].get(fv.thermal.pattern, 0.0)

        us = cfg["ultra_sub"]
        s_ultra = _clamp(
            us["bulge"] * _norm(fv.ultra.bulge_max, ref["bulge_max_mm"]) +
            us["area"]  * _norm(fv.ultra.bulge_area, ref["bulge_area_m2"]) +
            us["disc"]  * _norm(fv.ultra.profile_disc, ref["profile_disc"])
        )

        # --- 융합: 가중합 + 동시성 보너스 (둘 다 높을 때만) ---
        base = cfg["w_thermal"] * s_thermal + cfg["w_ultra"] * s_ultra
        agree_bonus = cfg["lambda_agree"] * fv.agree * min(s_thermal, s_ultra)
        s_risk = _clamp(base + agree_bonus)

        # --- 컨텍스트(결과 심각도) 가중 ---
        ctx = fv.context
        c_factor = (1.0
                    + cfg["gamma_height"] * _norm(ctx.height_m, ref["height_m"])
                    + cfg["gamma_public"] * (1.0 if ctx.over_public else 0.0)
                    + cfg["gamma_cluster"] * ctx.cluster_density)
        s_final = _clamp(s_risk * c_factor)

        # --- 신뢰도 ---
        cc = cfg["conf"]
        confidence = _clamp(
            cc["base"]
            + cc["thermal_valid"] * (1.0 if fv.thermal.valid else 0.0)
            + cc["ultra"] * s_ultra
            + cc["agree"] * fv.agree
        )

        return RiskScore(
            zone_id=fv.zone_id,
            score=round(s_final, 3),
            confidence=round(confidence, 3),
            contributions={
                "thermal": round(s_thermal, 3),
                "ultra": round(s_ultra, 3),
                "agree": round(fv.agree, 3),
                "context_factor": round(c_factor, 3),
            },
            engine=EngineMeta("rule", self.version),
        )


def to_grade(rs: RiskScore, cfg=GRADING) -> Grade:
    """점수+신뢰도 → 등급. 신뢰도 부족하면 단정하지 않고 보류(HOLD)."""
    if rs.confidence < cfg["tau_conf"]:
        return "HOLD"
    g = cfg["grade"]
    if rs.score >= g["E"]:
        return "E"
    if rs.score >= g["D"]:
        return "D"
    if rs.score >= g["C"]:
        return "C"
    if rs.score >= g["B"]:
        return "B"
    return "A"


# 전체 등급 산정용 심각도 순위 (보수적: HOLD를 C보다 위로). E=최위험
_RANK = {"E": 5, "D": 4, "HOLD": 3.5, "C": 3, "B": 2, "A": 1}


def worst_grade(grades) -> Grade:
    return max(grades, key=lambda g: _RANK[g])
