# drone_risk — 외장재 탈락·구조 안전 위험 평가 (온보드 MVP)

드론이 건물 외벽을 **열화상 + 초음파**로 촬영 → 위험도 **A~E**(+보류) 평가 → 보고서 패킷 생성.
규칙 기반 엔진(MVP). `FeatureVector → RiskScore` 계약을 지키면 나중에 ML 엔진으로 무수정 교체.

실데이터 버전: 고해상도 열화상(120×160)·깊이맵(30×40) 프레임을 **디트렌딩 → 임계 → 연결요소
→ GSD 실단위 변환 → PCA 패턴 분류 → 센서 정합**으로 직접 추출한다.

## 설치 / 실행

```bash
# 의존성 (1회)
python -m pip install numpy

# 이 환경의 Python 경로
"$LOCALAPPDATA/Programs/Python/Python312/python.exe" demo.py
# (또는 PATH 등록 후)  python demo.py
```

## 모듈

| 파일 | 역할 |
|---|---|
| `contracts.py` | FeatureVector / RiskScore — 고정 데이터 계약 |
| `config.py` | 임계값·가중치 (현장 보정용, 코드 수정 없이) |
| `imaging.py` | 카메라 모델(GSD)·평면 디트렌딩·연결요소·센서 정합 |
| `frame_sensors.py` | 현실적 프레임 생성 — 일사 구배·틸트·노이즈·결함 신호 |
| `features.py` | 프레임 → 지표 추출 (ΔT_max, bulge_max, 패턴, 동시성) |
| `risk_engine.py` | 규칙 엔진: 융합·점수화·등급(+보류) |
| `report.py` | 보고서 패킷 빌더 (+무결성 해시) |
| `pipeline.py` | 센서 → 지표 → 등급 → 보고서 오케스트레이션 |

## 검증 시나리오 (demo.py)

| 구역 | 상황 | 기대 등급 | 검증 포인트 |
|---|---|---|---|
| east-1F | 건전 | A | 구배·노이즈만 → 이상 없음 (최안전) |
| south-5F | 진행된 박리(열만) | D | 큰 열 핫스팟, 부풀음 전 |
| north-7F | 탈락 임박(열+기하) | E | 열·깊이 정합(agree) → 동시성 보너스 (최위험) |
| west-3F | 흐린날 촬영(열 무효) | 보류(HOLD) | valid=False → 안전 단정 금지 |
| south-2F | 균열(선형) | C | PCA 길쭉함 → crack-linear 분류 |

## 설계 문서

- `../drone-risk-assessment-architecture.md` — 전체 SW 아키텍처
- `../risk-grading-logic.md` — 등급 산정 로직 상세

## 테스트 (회귀 안전망)

```bash
# 의존성 없음(stdlib unittest). 프로젝트 루트(Downloads)에서:
python -m unittest discover -s tests -t . -v
```

36개 테스트 / 8개 파일 — imaging·features·risk_engine·report·transport 단위 +
`test_pipeline.py`가 현재 5개 시나리오 등급(D/B/A/HOLD/C)을 **고정**(엔진·추출·설정
변경의 부작용을 즉시 잡음) + `test_backend.py`가 HTTP 제출→조회→경보 왕복 검증.

## 백엔드 + 웹 대시보드

```bash
# 1) 백엔드 서버 (보고서 수신 API + 대시보드 서빙)
python backend/server.py 8765        # http://127.0.0.1:8765/ 접속

# 2) 온보드 → 백엔드 제출 시연 (오프라인 큐 + 멱등 재전송)
python submit_demo.py
```

- `drone_risk/transport/` — 오프라인 큐(`queue`)·디스크 버퍼(`store`)·API 클라이언트(`client`)
- `backend/server.py` — stdlib HTTP 서버: POST /api/reports(멱등), GET /api/reports·/api/alerts, 대시보드 서빙
- `backend/dashboard.html` — 구역별 등급 히트맵 + A/B 경보 + 클릭 상세(자동 갱신)

## 다음 작업 후보

- 실제 라디오메트릭 열화상/거리 파일(.tiff/.csv) 로더 — 현재는 합성 프레임
- 센서 외부 캘리브레이션(extrinsic)으로 열·초음파 정밀 정합 (현재는 공동 FOV 가정)
- 백엔드 인증·DB·오브젝트 스토리지(현재는 JSON 파일), 푸시 알림
- 캘리브레이션: known 박리 샘플로 임계값 보정 → ML 학습셋 축적
