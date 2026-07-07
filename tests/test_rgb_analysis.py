"""rgb_analysis.py — 일반 사진 '의심 지점 표시'(사람 판단 보조, 자동등급 없음)."""
import io
import unittest
import numpy as np
from PIL import Image, ImageDraw
from drone_risk.rgb_analysis import analyze_photo

_rng = np.random.default_rng(0)
_BASE = np.clip(150 + _rng.normal(0, 25, (700, 900)), 0, 255).astype("uint8")


def _jpg(img):
    b = io.BytesIO(); img.save(b, format="JPEG", quality=90); return b.getvalue()


def _wall():
    return Image.fromarray(_BASE).convert("RGB")


class TestRgbAssist(unittest.TestCase):
    def test_no_auto_grade(self):
        r = analyze_photo(_jpg(_wall()))
        self.assertNotIn("grade", r)              # 자동 등급을 매기지 않는다
        self.assertIn("num_candidates", r)

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


if __name__ == "__main__":
    unittest.main()
