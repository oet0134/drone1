"""transport — 백엔드 제출. 통신 두절 시에도 등급 결과를 잃지 않는다.

store  : 디스크 영구 보관(미전송 버퍼)
queue  : 오프라인 큐 — 끊기면 버퍼링, 복귀 시 멱등 재전송
client : 백엔드 API 클라이언트(HTTP POST, 멱등키, 실패 시 False)
"""
