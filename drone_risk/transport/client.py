"""백엔드 API 클라이언트 — HTTP POST + 멱등키. 실패(통신 두절)는 False로."""
from __future__ import annotations
import json
import urllib.error
import urllib.request


class BackendClient:
    def __init__(self, base_url: str, timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def submit(self, report: dict) -> bool:
        data = json.dumps(report, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/api/reports",
            data=data,
            headers={
                "Content-Type": "application/json",
                # 같은 보고서를 여러 번 보내도 서버가 한 건으로 처리
                "Idempotency-Key": report["report_id"],
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status in (200, 201)
        except (urllib.error.URLError, OSError):
            return False        # 두절 → 큐에 남겨 재전송
