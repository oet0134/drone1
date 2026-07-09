"""rgb_analysis.py — 일반 사진 '의심 지점 표시'(사람 판단 보조, 자동등급 없음)."""
import io
import unittest
import numpy as np
from PIL import Image, ImageDraw
from drone_risk.rgb_analysis import analyze_photo, baseline_from, monitor_frame

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
    """상시 감시: 기준의 기존 이상은 무시, 새로 생긴 것만 경보."""
    def _with_line(self, extra=None):
        img = _wall()
        d = ImageDraw.Draw(img)
        d.line([(100, 150), (400, 450)], fill=(30, 30, 30), width=6)   # 기존 특징
        if extra:
            d.line(extra, fill=(28, 28, 28), width=6)                  # 새 균열
        return img

    def test_unchanged_frame_is_normal(self):
        base = baseline_from(_jpg(self._with_line()))
        self.assertGreaterEqual(base["count"], 1)
        res = monitor_frame(_jpg(self._with_line()), base["centers"])
        self.assertEqual(res["status"], "normal")
        self.assertEqual(res["new_count"], 0)      # 기존 선은 기준에 있어 무시

    def test_new_defect_triggers_alert(self):
        base = baseline_from(_jpg(self._with_line()))
        res = monitor_frame(_jpg(self._with_line([(600, 150), (840, 470)])), base["centers"])
        self.assertEqual(res["status"], "alert")
        self.assertGreaterEqual(res["new_count"], 1)   # 새 균열만 경보


if __name__ == "__main__":
    unittest.main()
