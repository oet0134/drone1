"""report.py — 보고서 패킷 구조, 멱등키, 무결성, 권고 조치."""
import unittest
from drone_risk.contracts import (ThermalFeat, UltraFeat, ContextFeat,
                                  FeatureVector, RiskScore, EngineMeta)
from drone_risk.report import build_zone, build_report, GRADE_ACTION


def _fv():
    th = ThermalFeat(6.5, 1.2, 1, 0.3, "debond", True, set())
    ul = UltraFeat(22.0, 1.5, 0, set())
    ctx = ContextFeat(30, True, 0.5, "N")
    return FeatureVector("north-7F", th, ul, ctx, 0.75,
                         {"facade": "N", "alt_m": 30})


def _rs():
    return RiskScore("north-7F", 1.0, 1.0,
                     {"thermal": 0.76, "ultra": 0.83, "agree": 0.75,
                      "context_factor": 1.8},
                     EngineMeta("rule", "0.3.0"))


class TestReport(unittest.TestCase):
    def test_zone_has_action_matching_grade(self):
        z = build_zone(_fv(), _rs(), "A")
        self.assertEqual(z["action"], GRADE_ACTION["A"])
        self.assertEqual(z["grade"], "A")
        self.assertIn("thermal", z["indicators"])
        self.assertIn("contributions", z)
        self.assertEqual(z["engine"]["version"], "0.3.0")

    def test_report_idempotency_key_and_integrity(self):
        zone = build_zone(_fv(), _rs(), "A")
        rep = build_report("B-1", "2026-06-21T09:14:00+09:00",
                           {"id": "DRN-07"}, "A", [zone])
        self.assertEqual(rep["report_id"], "B-1@2026-06-21T09:14:00+09:00")
        self.assertTrue(rep["integrity"]["hash"].startswith("sha256:"))
        self.assertEqual(rep["overall_grade"], "A")
        self.assertEqual(rep["overall_action"], GRADE_ACTION["A"])


if __name__ == "__main__":
    unittest.main()
