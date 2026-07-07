"""backend/server.py — HTTP 왕복: 제출(멱등)·조회·경보·시딩·라벨링."""
import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer

from backend import server as srv
from drone_risk.frame_sensors import building_scan
from drone_risk.pipeline import assess_building
from drone_risk.transport.client import BackendClient


class TestBackend(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        srv.DATA = os.path.join(self._tmp.name, "reports")
        srv.LABELS = os.path.join(self._tmp.name, "labels")
        os.makedirs(srv.DATA); os.makedirs(srv.LABELS)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.Handler)
        self.port = self.httpd.server_address[1]
        self.t = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.t.start()
        self.client = BackendClient(f"http://127.0.0.1:{self.port}")
        self.report = assess_building(building_scan())

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self._tmp.cleanup()

    def _get(self, path):
        url = f"http://127.0.0.1:{self.port}{path}"
        return json.load(urllib.request.urlopen(url, timeout=5))

    def _post(self, path, body):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        return json.load(urllib.request.urlopen(req, timeout=5))

    def test_submit_idempotent_then_query(self):
        self.assertTrue(self.client.submit(self.report))
        self.assertTrue(self.client.submit(self.report))   # 재전송
        self.assertEqual(len(self._get("/api/reports")), 1)
        self.assertEqual(len(os.listdir(srv.DATA)), 1)

    def test_alerts_surface_high_risk(self):
        self.client.submit(self.report)
        grades = {a["grade"] for a in self._get("/api/alerts")}
        self.assertIn("E", grades)      # north-7F 탈락 임박
        self.assertIn("D", grades)      # south-5F 진행된 박리

    def test_seed_creates_varied_buildings(self):
        self._post("/api/seed", {})
        reps = self._get("/api/reports")
        self.assertEqual(len(reps), 8)
        grades = {r["overall_grade"] for r in reps}
        self.assertIn("A", grades)
        self.assertIn("E", grades)

    def test_label_accumulates_training_data(self):
        self.client.submit(self.report)
        full = self._get("/api/reports/"
                         + urllib.parse.quote(self.report["report_id"]))
        zid = full["zones"][0]["zone_id"]
        res = self._post("/api/labels", {"report_id": self.report["report_id"],
                                         "zone_id": zid, "grade": "A", "note": "t"})
        self.assertEqual(res["status"], "labeled")
        summ = self._get("/api/training-summary")
        self.assertEqual(summ["labeled"], 1)
        # 같은 구역 재라벨 → 덮어씀(멱등), 누적 1건 유지
        self._post("/api/labels", {"report_id": self.report["report_id"],
                                   "zone_id": zid, "grade": "B", "note": "t2"})
        self.assertEqual(self._get("/api/training-summary")["labeled"], 1)

    def test_bad_grade_rejected(self):
        self.client.submit(self.report)
        full = self._get("/api/reports/"
                         + urllib.parse.quote(self.report["report_id"]))
        zid = full["zones"][0]["zone_id"]
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self._post("/api/labels", {"report_id": self.report["report_id"],
                                       "zone_id": zid, "grade": "Z"})
        self.assertEqual(cm.exception.code, 400)

    def test_missing_report_id_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self._post("/api/reports", {"no": "id"})
        self.assertEqual(cm.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
