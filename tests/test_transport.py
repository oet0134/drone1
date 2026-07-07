"""transport — 디스크 버퍼, 오프라인 큐, 멱등 재전송."""
import tempfile
import unittest
from drone_risk.transport.store import PendingStore
from drone_risk.transport.queue import OfflineQueue

REPORT = {"report_id": "B-1@2026-06-21T09:14:00+09:00", "overall_grade": "A"}


class StubClient:
    """submit 성공/실패를 제어하고 전송된 보고서를 기록하는 가짜 클라이언트."""
    def __init__(self, ok):
        self.ok = ok
        self.submitted = []

    def submit(self, report):
        if self.ok:
            self.submitted.append(report["report_id"])
        return self.ok


class TestStore(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            s = PendingStore(d)
            fp = s.put(REPORT)
            self.assertEqual(len(s.list()), 1)
            self.assertEqual(s.load(fp)["report_id"], REPORT["report_id"])
            s.remove(fp)
            self.assertEqual(len(s.list()), 0)


class TestQueue(unittest.TestCase):
    def test_offline_keeps_pending(self):
        with tempfile.TemporaryDirectory() as d:
            q = OfflineQueue(PendingStore(d))
            q.enqueue(REPORT)
            sent, failed = q.flush(StubClient(ok=False))   # 통신 두절
            self.assertEqual(sent, [])
            self.assertEqual(q.pending(), 1)               # 안 잃음

    def test_online_flushes(self):
        with tempfile.TemporaryDirectory() as d:
            q = OfflineQueue(PendingStore(d))
            q.enqueue(REPORT)
            client = StubClient(ok=True)
            sent, failed = q.flush(client)                 # 복귀
            self.assertEqual(sent, [REPORT["report_id"]])
            self.assertEqual(q.pending(), 0)
            self.assertEqual(client.submitted, [REPORT["report_id"]])


if __name__ == "__main__":
    unittest.main()
