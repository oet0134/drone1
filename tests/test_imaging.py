"""imaging.py — 카메라 모델·디트렌딩·연결요소·정합."""
import math
import unittest
import numpy as np
from drone_risk.imaging import (gsd_m_per_px, detrend_plane,
                                label_components, to_norm_grid, iou)


class TestImaging(unittest.TestCase):
    def test_gsd(self):
        # 거리8m, FOV50°, 160px → 폭 = 2*8*tan25° / 160
        expect = 2 * 8 * math.tan(math.radians(25)) / 160
        self.assertAlmostEqual(gsd_m_per_px(8, 50, 160), expect, places=6)

    def test_detrend_removes_linear_ramp(self):
        ys, xs = np.mgrid[0:8, 0:8]
        frame = 2.0 + 0.5 * xs + 0.3 * ys          # 순수 평면
        residual, _ = detrend_plane(frame)
        self.assertLess(np.abs(residual).max(), 1e-6)   # 잔차 ~0

    def test_detrend_preserves_local_bump(self):
        ys, xs = np.mgrid[0:8, 0:8]
        frame = 2.0 + 0.5 * xs + 0.3 * ys
        frame[4, 4] += 10.0                         # 국부 돌출
        residual, _ = detrend_plane(frame)
        self.assertGreater(residual[4, 4], 5.0)     # 돌출은 살아남음

    def test_label_components_filters_noise(self):
        mask = np.zeros((6, 6), dtype=bool)
        mask[1:4, 1:4] = True       # 3x3 = 9px 덩어리
        mask[5, 5] = True           # 고립 1px (잡음)
        comps = label_components(mask, min_px=4)
        self.assertEqual(len(comps), 1)
        self.assertEqual(comps[0]["area_px"], 9)

    def test_iou(self):
        a = np.zeros((4, 4), bool); a[0:2, 0:2] = True
        self.assertEqual(iou(a, a), 1.0)
        b = np.zeros((4, 4), bool); b[2:4, 2:4] = True
        self.assertEqual(iou(a, b), 0.0)

    def test_norm_grid_cross_resolution(self):
        # 해상도가 달라도 같은 정규화 위치면 겹친다
        a = to_norm_grid({(10, 10)}, (20, 20), n=32)
        b = to_norm_grid({(5, 5)}, (10, 10), n=32)
        self.assertEqual(iou(a, b), 1.0)


if __name__ == "__main__":
    unittest.main()
