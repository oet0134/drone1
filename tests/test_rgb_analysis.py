"""rgb_analysis.py — 일반 사진 '의심 지점 표시'(사람 판단 보조, 자동등급 없음)."""
import io
import unittest
import numpy as np
from PIL import Image, ImageDraw
from drone_risk.rgb_analysis import analyze_photo, monitor_baseline, monitor_check

_rng = np.random.default_rng(0)
_BASE = np.clip(150 + _rng.normal(0, 25, (700, 900)), 0, 255).astype("uint8")


def _jpg(img):
    b = io.BytesIO(); img.save(b, format="JPEG", quality=90); return b.getvalue()


def _wall():
    return Image.fromarray(_BASE).convert("RGB")


class TestRgbAssist(unittest.TestCase):
    def test_suggests_grade_for_human(self):
        r = analyze_photo(_jpg(_wall()))
        self.assertIn(r["suggested_grade"], {"A", "B", "C", "D", "E", "HOLD"})  # AI 제안(참고용)
        self.assertIn("num_candidates", r)
        self.assertNotIn("grade", r)              # 최종'확정'이 아니라 제안일 뿐

    def test_clean_wall_few_candidates(self):
        r = analyze_photo(_jpg(_wall()))
        self.assertLessEqual(r["num_candidates"], 1)   # 깨끗한 벽 → 거의 없음
        self.assertEqual(r["photo_quality"], "ok")

    def test_crack_flagged(self):
        img = _wall()
        ImageDraw.Draw(img).line([(120, 150), (760, 540)], fill=(28, 28, 28), width=5)
        self.assertGreaterEqual(analyze_photo(_jpg(img))["num_candidates"], 1)

    def test_candidate_cap(self):
        # 선이 아주 많아도 표시 개수는 상한(max_candidates=25) 이하
        img = _wall(); d = ImageDraw.Draw(img)
        for y in range(40, 680, 12):
            d.line([(30, y), (870, y + 40)], fill=(25, 25, 25), width=3)
        self.assertLessEqual(analyze_photo(_jpg(img))["num_candidates"], 25)

    def test_returns_annotated_jpeg(self):
        self.assertEqual(analyze_photo(_jpg(_wall()))["annotated_jpg"][:2], b"\xff\xd8")

    def test_low_contrast_flagged(self):
        flat = Image.new("RGB", (400, 300), (128, 128, 128))
        self.assertEqual(analyze_photo(_jpg(flat))["photo_quality"], "low")


class TestMonitor(unittest.TestCase):
    """상시 감시(변화 감지): 기준 대비 새로 어두워진 국부 변화만 경보, 전체 변화는 '이동'."""
    def test_unchanged_is_normal(self):
        base = monitor_baseline(_jpg(_wall()))
        res = monitor_check(_jpg(_wall()), base["gray_png"])
        self.assertEqual(res["status"], "normal")
        self.assertEqual(res["new_count"], 0)

    def test_new_mark_alerts(self):
        base = monitor_baseline(_jpg(_wall()))
        img = _wall()
        ImageDraw.Draw(img).line([(200, 120), (300, 260)], fill=(20, 20, 20), width=10)
        res = monitor_check(_jpg(img), base["gray_png"])    # 짧은 낙서도 잡힘
        self.assertEqual(res["status"], "alert")
        self.assertGreaterEqual(res["new_count"], 1)

    def test_global_change_is_moved(self):
        base = monitor_baseline(_jpg(_wall()))
        dark = np.clip(_BASE.astype(int) - 60, 0, 255).astype("uint8")   # 화면 전체 어두워짐
        res = monitor_check(_jpg(Image.fromarray(dark).convert("RGB")), base["gray_png"])
        self.assertEqual(res["status"], "moved")


if __name__ == "__main__":
    unittest.main()
