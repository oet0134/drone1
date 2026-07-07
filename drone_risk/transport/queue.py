"""오프라인 큐 — 통신 두절 시 버퍼링, 복귀 시 멱등 재전송."""
from __future__ import annotations
from .store import PendingStore
from .client import BackendClient


class OfflineQueue:
    def __init__(self, store: PendingStore):
        self.store = store

    def enqueue(self, report: dict):
        self.store.put(report)

    def pending(self) -> int:
        return len(self.store.list())

    def flush(self, client: BackendClient):
        """전송 성공분만 제거. 실패분은 큐에 남아 다음 기회에 재시도."""
        sent, failed = [], []
        for fp in self.store.list():
            report = self.store.load(fp)
            if client.submit(report):
                self.store.remove(fp)
                sent.append(report["report_id"])
            else:
                failed.append(report["report_id"])
        return sent, failed
