"""등급 산정 설정 — 코드 수정 없이 현장 보정.

★ 모든 임계값은 초기 추정치다. 건전/박리가 확인된 외벽 샘플로
   캘리브레이션해야 신뢰할 수 있는 등급이 나온다.
"""

CELL_AREA_M2 = 0.25            # 그리드 셀 1칸이 덮는 외벽 면적 (0.5m x 0.5m)

EXTRACT = {
    # 열화상 픽셀 단위 이상 판정 (디트렌딩 잔차 기준)
    "thermal_cell_dt": 2.0,    # |잔차 ΔT| ≥ 2.0°C 이면 이상 픽셀
    # 초음파/깊이 픽셀 단위 돌출 판정 (디트렌딩 잔차 기준)
    "ultra_cell_bulge_mm": 5.0,  # 평면 대비 ≥ 5mm 돌출이면 이상 픽셀
    "ultra_disc_mm": 10.0,       # 인접 픽셀 거리 구배 ≥ 10mm 이면 단차 픽셀
    "min_comp_px": 5,            # 연결요소 최소 크기(미만은 잡음으로 버림)
}

GRADING = {
    # 센서별 부분 점수 내부 가중 (각 합=1)
    "thermal_sub": {"dt": 0.5, "area": 0.3, "grad": 0.2},
    "ultra_sub":   {"bulge": 0.6, "area": 0.3, "disc": 0.1},

    # 지표 정규화 기준값 (이 값에서 norm=1로 포화)
    "ref": {
        "dt_max_c": 6.0,
        "anomaly_area_m2": 2.0,
        "gradient_c": 0.5,             # °C/cm (실프레임 잔차 구배 스케일에 맞춤)
        "bulge_max_mm": 20.0,
        "bulge_area_m2": 2.0,
        "profile_disc": 6.0,
        "height_m": 30.0,
    },

    # 패턴별 위험 가중 (박리가 주 타깃)
    "pattern_weight": {
        "debond": 1.0, "crack-linear": 0.7, "moisture": 0.4,
        "thermal-bridge": 0.3, "none": 0.0,
    },

    # 센서 융합
    "w_thermal": 0.55,         # 박리 주력 → 열화상 우세
    "w_ultra": 0.45,
    "lambda_agree": 0.25,      # 동시성 보너스

    # 컨텍스트(결과 심각도) 가중
    "gamma_height": 0.3,
    "gamma_public": 0.4,
    "gamma_cluster": 0.2,

    # 신뢰도 산출 가중
    "conf": {"base": 0.32, "thermal_valid": 0.30, "ultra": 0.45, "agree": 0.10},

    # 등급 임계 — 국가 시설물 안전등급과 동일 방향 (A=최안전 … E=최위험)
    "grade": {"E": 0.80, "D": 0.60, "C": 0.40, "B": 0.20},   # A(최안전)는 이 아래 전부
    "tau_conf": 0.55,          # 이 미만이면 '보류'(HOLD)
}

# 일반(RGB) 사진 결함 '후보 표시' 설정 — 코드 수정 없이 현장 보정
# ★ 자동 등급이 아니다. 단순 영상처리로는 균열과 창틀·줄눈을 못 가리므로,
#    '의심 지점(균열형 어두운 선)'만 보수적으로 표시하고 등급은 사람이 정한다.
RGB = {
    "max_dim": 1000,            # 분석 전 이미지를 이 크기로 축소(속도)
    "blur_radius_frac": 0.05,   # 배경 밝기 추정용 흐림 반경(짧은 변 대비 비율)
    "dark_thresh": 26,          # 주변보다 이만큼(0~255) 어두운 것만(약한 무늬 배제)
    "min_comp_px": 40,          # 이보다 작은 얼룩은 잡음으로 버림
    "crack_elong": 4.0,         # PCA 길쭉함이 이 이상인 '길쭉한' 것만 후보
    "min_len_frac": 0.10,       # 이미지 대각선의 10% 이상 길이만 후보
    "max_candidates": 25,       # 표시할 의심 지점 최대 수(진하고 긴 순)
    "min_contrast": 10,         # 전체 대비 이 미만이면 사진 품질 '낮음' 경고
    # AI 제안 등급(참고용) 산출 — 가장 강한 후보 강도 + 후보 수 → concern → A~E
    "suggest": {"top_ref": 0.22, "cnt_ref": 18.0, "w_top": 0.7, "w_cnt": 0.3},
}
