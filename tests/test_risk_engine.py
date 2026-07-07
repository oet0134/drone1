"""risk_engine.py — 정규화, 융합·동시성, 등급 매핑, 보류, 최악등급."""
import unittest
from drone_risk.contracts import (ThermalFeat, UltraFeat, ContextFeat,
                                  FeatureVector, RiskScore, EngineMeta)
from drone_risk.risk_engine import (RuleBasedEngine, to_grade, worst_grade,
                                    _norm, _clamp)

ENG = RuleBasedEngine()


def make_fv(dt=0, area=0, grad=0, pattern="none", valid=True,
            bulge=0, ul_area=0, disc=0, agree=0.0,
            height=0, public=False, cluster=0.0):
    th = ThermalFeat(dt, area, 1, grad, pattern, valid, set())
    ul = UltraFeat(bulge, ul_area, disc, set())
    ctx = ContextFeat(height, public, cluster, "N")
    return FeatureVector("z", th, ul, ctx, agree, {})


class TestNorm(unittest.TestCase):
    def test_norm(self):
        self.assertEqual(_norm(3, 6), 0.5)
        self.assertEqual(_norm(10, 5), 1.0)     # 포화
        self.assertEqual(_norm(0, 5), 0.0)
        self.assertEqual(_clamp(1.5), 1.0)


class TestScoring(unittest.TestCase):
    def test_benign_is_low(self):
        rs = ENG.score(make_fv(valid=True))
        self.assertLess(rs.score, 0.20)
        self.assertEqual(to_grade(rs), "A")     # 최안전, conf 충분(valid)

    def test_thermal_monotonic(self):
        lo = ENG.score(make_fv(dt=2, pattern="debond")).score
        hi = ENG.score(make_fv(dt=5, pattern="debond")).score
        self.assertGreater(hi, lo)

    def test_agree_bonus(self):
        base = dict(dt=3, area=1, grad=0.25, pattern="debond",
                    bulge=10, ul_area=1, disc=3)
        no = ENG.score(make_fv(agree=0.0, **base)).score
        yes = ENG.score(make_fv(agree=1.0, **base)).score
        self.assertGreater(yes, no)             # 동시성 보너스

    def test_context_amplifies(self):
        flat = ENG.score(make_fv(dt=4, pattern="debond")).score
        tall = ENG.score(make_fv(dt=4, pattern="debond",
                                 height=30, public=True, cluster=0.5)).score
        self.assertGreater(tall, flat)

    def test_invalid_thermal_low_confidence_holds(self):
        rs = ENG.score(make_fv(valid=False))
        self.assertLess(rs.confidence, 0.55)
        self.assertEqual(to_grade(rs), "HOLD")


class TestGradeMapping(unittest.TestCase):
    def _rs(self, score, conf):
        return RiskScore("z", score, conf, {}, EngineMeta("rule", "0"))

    def test_thresholds(self):
        self.assertEqual(to_grade(self._rs(0.9, 0.9)), "E")
        self.assertEqual(to_grade(self._rs(0.7, 0.9)), "D")
        self.assertEqual(to_grade(self._rs(0.45, 0.9)), "C")
        self.assertEqual(to_grade(self._rs(0.25, 0.9)), "B")
        self.assertEqual(to_grade(self._rs(0.1, 0.9)), "A")
        self.assertEqual(to_grade(self._rs(0.9, 0.4)), "HOLD")   # 신뢰도 부족

    def test_worst_grade(self):
        self.assertEqual(worst_grade(["A", "C", "HOLD"]), "HOLD")
        self.assertEqual(worst_grade(["E", "HOLD"]), "E")
        self.assertEqual(worst_grade(["A", "D"]), "D")
        self.assertEqual(worst_grade(["A", "B"]), "B")
        self.assertEqual(worst_grade(["A", "A"]), "A")


if __name__ == "__main__":
    unittest.main()
