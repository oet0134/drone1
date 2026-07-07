"""데이터 계약 — features 출력(FeatureVector)과 risk_engine 출력(RiskScore).

이 두 구조는 '고정 계약'이다. 규칙 엔진을 ML 엔진으로 교체해도
이 입출력만 지키면 지표 추출·보고서·전송 코드는 손대지 않는다.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

Pattern = Literal["debond", "moisture", "crack-linear", "thermal-bridge", "none"]
Grade = Literal["A", "B", "C", "D", "E", "HOLD"]


@dataclass
class ThermalFeat:
    dt_max: float                 # 건전 기준면 대비 최대 온도 대비 [°C]
    anomaly_area: float           # 연속 이상 영역 면적 [m²]
    anomaly_count: int            # 독립 이상 영역 개수
    gradient: float               # 이상 경계 온도 구배 [°C/cell]
    pattern: Pattern
    valid: bool                   # 캡처 유효성(열 과도상태 충분 여부)
    cells: set = field(default_factory=set)   # 이상 셀 좌표 (융합용)


@dataclass
class UltraFeat:
    bulge_max: float              # 외벽 평면 대비 최대 돌출 변위 [mm]
    bulge_area: float             # 돌출 영역 면적 [m²]
    profile_disc: int             # 거리 프로파일 급격한 단차 개수
    cells: set = field(default_factory=set)


@dataclass
class ContextFeat:
    height_m: float
    over_public: bool             # 보행로/출입구 상부 여부
    cluster_density: float        # 인접 이상 군집도 [0~1]
    facade: str                   # 방위 (N/E/S/W)


@dataclass
class FeatureVector:
    zone_id: str
    thermal: ThermalFeat
    ultra: UltraFeat
    context: ContextFeat
    agree: float                  # 열·초음파 이상 위치 일치도 [0~1] (융합)
    geo: dict = field(default_factory=dict)


@dataclass
class EngineMeta:
    type: str
    version: str


@dataclass
class RiskScore:
    zone_id: str
    score: float                  # S_final [0~1]
    confidence: float             # [0~1]
    contributions: dict           # {thermal, ultra, agree, context} 기여도 — 설명가능성
    engine: EngineMeta
