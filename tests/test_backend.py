"""backend/server.py — HTTP 왕복: 제출·조회·시딩·라벨·사진경보·현장요약."""
import base64
import io
import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer

from PIL import Image, ImageDraw

from backend import server as srv
from drone_risk.frame_sensors import building_scan
from drone_risk.pipeline import assess_building
from drone_risk.transport.client import BackendClient


def _jpg_b64(color=(140, 150, 160)):
    buf = io.BytesIO()
    Image.new("RGB", (400, 300), color).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _mark_b64():
    """밝은 배경에 진한 사각형(=새로 생긴 표시) 하나."""
    img = Image.new("RGB", (400, 300), (205, 205, 205))
    ImageDraw.Draw(img).rectangle([120, 100, 230, 190], fill=(20, 20, 20))
    buf = io.BytesIO(); img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


class TestBackend(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        for name in ("DATA", "LABELS", "PHOTOS", "UPLOADS", "MONITORS"):
            p = os.path.join(self._tmp.name, name.lower())
            setattr(srv, name, p)
            os.makedirs(p)
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
        return json.load(urllib.request.urlopen(req, timeout=30))

    def test_submit_idempotent(self):
        self.assertTrue(self.client.submit(self.report))
        self.assertTrue(self.client.submit(self.report))
        self.assertEqual(len(self._get("/api/reports")), 1)

    def test_photo_alert_and_site_summary(self):
        rec = self._post("/api/analyze-photo",
                         {"building": "테스트현장", "filename": "t.jpg",
                          "image_b64": _jpg_b64()})
        self.assertIsNone(rec["human_grade"])                 # 처음엔 미확정
        self._post("/api/photo-grade", {"id": rec["id"], "grade": "E"})
        self.assertIn("E", {a["grade"] for a in self._get("/api/alerts")})
        sites = self._get("/api/sites")
        self.assertTrue(any(s["site"] == "테스트현장" and s["overall"] == "E"
                            for s in sites))

    def test_photo_confirmation_counts_in_training(self):
        rec = self._post("/api/analyze-photo",
                         {"building": "A", "filename": "a.jpg", "image_b64": _jpg_b64()})
        self._post("/api/photo-grade", {"id": rec["id"], "grade": "C"})
        summ = self._get("/api/training-summary")
        self.assertGreaterEqual(summ["labeled"], 1)           # 사진 확정도 학습에 집계

    def test_photo_delete(self):
        rec = self._post("/api/analyze-photo",
                         {"building": "A", "filename": "a.jpg", "image_b64": _jpg_b64()})
        self.assertEqual(len(self._get("/api/photos")), 1)
        self._post("/api/photo-delete", {"id": rec["id"]})
        self.assertEqual(len(self._get("/api/photos")), 0)   # 삭제되어 목록에서 사라짐

    def test_monitor_unchanged_is_normal(self):
        self._post("/api/monitor/baseline",
                   {"monitor_id": "m1", "image_b64": _jpg_b64((205, 205, 205))})
        res = self._post("/api/monitor/frame",
                         {"monitor_id": "m1", "image_b64": _jpg_b64((205, 205, 205))})
        self.assertEqual(res["status"], "normal")
        self.assertIn("image_url", res)

    def test_monitor_debounce_then_alert(self):
        self._post("/api/monitor/baseline",
                   {"monitor_id": "m2", "image_b64": _jpg_b64((205, 205, 205))})
        r1 = self._post("/api/monitor/frame", {"monitor_id": "m2", "image_b64": _mark_b64()})
        self.assertEqual(r1["status"], "watching")     # 첫 감지 → 확인 중
        r2 = self._post("/api/monitor/frame", {"monitor_id": "m2", "image_b64": _mark_b64()})
        self.assertEqual(r2["status"], "alert")        # 연속 확인 → 경보

    def test_bad_grade_rejected(self):
        rec = self._post("/api/analyze-photo",
                         {"building": "A", "filename": "a.jpg", "image_b64": _jpg_b64()})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self._post("/api/photo-grade", {"id": rec["id"], "grade": "Z"})
        self.assertEqual(cm.exception.code, 400)

    def test_missing_report_id_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self._post("/api/reports", {"no": "id"})
        self.assertEqual(cm.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
