"""features.py — 프레임에서 지표 추출, 패턴 분류, 센서 정합."""
import unittest
import numpy as np
from drone_risk.contracts import ThermalFeat, UltraFeat
from drone_risk.features import extract_thermal, extract_ultra, co_location

CAM = {"distance_m": 8.0, "fov_deg": 50.0}


def _disk(frame, cy, cx, r, amp):
    ys, xs = np.mgrid[0:frame.shape[0], 0:frame.shape[1]]
    frame[(ys - cy) ** 2 + (xs - cx) ** 2 <= r * r] += amp


class TestThermal(unittest.TestCase):
    def test_clean_frame_no_anomaly(self):
        f = np.full((30, 40), 20.0)
        th = extract_thermal(f, True, CAM)
        self.assertEqual(th.dt_max, 0.0)
        self.assertEqual(th.pattern, "none")
        self.assertEqual(len(th.cells), 0)

    def test_compact_blob_is_debond(self):
        f = np.full((30, 40), 20.0)
        _disk(f, 15, 20, 5, 6.0)
        th = extract_thermal(f, True, CAM)
        self.assertEqual(th.pattern, "debond")
        self.assertGreater(th.dt_max, 3.0)
        self.assertGreater(th.anomaly_area, 0.0)

    def test_thin_line_is_crack(self):
        f = np.full((30, 40), 20.0)
        f[5:25, 15:17] += 6.0            # 세로 2px 선 → 길쭉
        th = extract_thermal(f, True, CAM)
        self.assertEqual(th.pattern, "crack-linear")

    def test_valid_flag_passthrough(self):
        f = np.full((30, 40), 20.0)
        self.assertFalse(extract_thermal(f, False, CAM).valid)


class TestUltra(unittest.TestCase):
    def test_flat_no_bulge(self):
        d = np.full((30, 40), 4000.0)
        ul = extract_ultra(d, CAM)
        self.assertEqual(ul.bulge_max, 0.0)
        self.assertEqual(len(ul.cells), 0)

    def test_protrusion_detected(self):
        d = np.full((30, 40), 4000.0)
        _disk(d, 15, 20, 4, -25.0)       # 센서 쪽 25mm 돌출
        ul = extract_ultra(d, CAM)
        self.assertGreater(ul.bulge_max, 10.0)
        self.assertGreater(len(ul.cells), 0)


class TestFusion(unittest.TestCase):
    def _feat(self, tcells, ucells):
        th = ThermalFeat(5, 1, 1, 1, "debond", True, tcells)
        ul = UltraFeat(20, 1, 1, ucells)
        return th, ul

    def test_colocated_high_agree(self):
        th, ul = self._feat({(10, 10)}, {(5, 5)})       # 다른 해상도, 같은 위치
        self.assertAlmostEqual(co_location(th, ul, (20, 20), (10, 10)), 1.0)

    def test_disjoint_zero_agree(self):
        th, ul = self._feat({(19, 19)}, {(0, 0)})
        self.assertEqual(co_location(th, ul, (20, 20), (10, 10)), 0.0)

    def test_empty_side_zero(self):
        th, ul = self._feat(set(), {(5, 5)})
        self.assertEqual(co_location(th, ul, (20, 20), (10, 10)), 0.0)


if __name__ == "__main__":
    unittest.main()
