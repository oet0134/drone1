"""회귀 안전망 — 현재 end-to-end 동작을 고정한다.

합성 5개 시나리오의 등급이 바뀌면(엔진/추출/설정 변경의 부작용) 즉시 실패.
"""
import unittest
from drone_risk.frame_sensors import building_scan
from drone_risk.pipeline import assess_building

EXPECT = {"east-1F": "A", "south-5F": "D", "north-7F": "E",
          "west-3F": "HOLD", "south-2F": "C"}


class TestRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = assess_building(building_scan())
        cls.by_zone = {z["zone_id"]: z for z in cls.report["zones"]}

    def test_grades_locked(self):
        got = {zid: z["grade"] for zid, z in self.by_zone.items()}
        self.assertEqual(got, EXPECT)

    def test_overall_is_worst(self):
        self.assertEqual(self.report["overall_grade"], "E")   # 최악=최위험 구역

    def test_imminent_zone_fused(self):
        z = self.by_zone["north-7F"]
        self.assertGreaterEqual(z["score"], 0.8)
        self.assertGreater(z["contributions"]["agree"], 0.5)   # 동시성 포착
        self.assertEqual(z["indicators"]["thermal"]["pattern"], "debond")

    def test_crack_zone_classified(self):
        z = self.by_zone["south-2F"]
        self.assertEqual(z["indicators"]["thermal"]["pattern"], "crack-linear")

    def test_bad_capture_holds(self):
        z = self.by_zone["west-3F"]
        self.assertEqual(z["grade"], "HOLD")
        self.assertFalse(z["indicators"]["thermal"]["valid"])


if __name__ == "__main__":
    unittest.main()
