"""end-to-end 데모 (실데이터 버전): 열화상/깊이 프레임 → 지표 추출 → 등급 → 보고서.

실행:  python demo.py
"""
import json
import sys

# Windows 콘솔(cp949)에서도 한글·기호가 깨지지 않도록 UTF-8 강제
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from drone_risk.frame_sensors import building_scan
from drone_risk.pipeline import assess_building

SCENARIO = {
    "east-1F":  "① 건전",
    "south-5F": "② 진행된 박리(열만)",
    "north-7F": "③ 탈락 임박(열+기하)",
    "west-3F":  "④ 흐린날(열 무효)",
    "south-2F": "⑤ 균열(선형)",
}
EXPECT = {"east-1F": "A", "south-5F": "D", "north-7F": "E",
          "west-3F": "HOLD", "south-2F": "C"}


def main():
    scan = building_scan()
    report = assess_building(scan)

    print("=" * 92)
    print(f"건물 {report['building_id']}  촬영 {report['captured_at']}"
          f"   전체 등급 ▶ {report['overall_grade']}")
    print("=" * 92)
    print(f"{'구역':<10}{'시나리오':<22}{'등급':<6}{'점수':<8}{'신뢰도':<8}{'기대':<6}{'판정'}")
    print("-" * 92)

    ok = True
    for z in report["zones"]:
        zid, exp, got = z["zone_id"], EXPECT[z["zone_id"]], z["grade"]
        passed = (got == exp)
        ok &= passed
        print(f"{zid:<10}{SCENARIO[zid]:<20}{got:<6}{z['score']:<8}"
              f"{z['confidence']:<8}{exp:<6}{'PASS' if passed else 'FAIL'}")

    print("-" * 92)
    print("추출 지표 + 기여도 (왜 그 등급인가):")
    for z in report["zones"]:
        c = z["contributions"]
        t = z["indicators"]["thermal"]
        u = z["indicators"]["ultrasonic"]
        print(f"  {z['zone_id']:<10} ΔT={t['dt_max_c']:>4}°C area={t['anomaly_area_m2']:>4}㎡ "
              f"pat={t['pattern']:<12} bulge={u['bulge_max_mm']:>5}mm | "
              f"S_t={c['thermal']:<5} S_u={c['ultra']:<5} agree={c['agree']:<5} ×{c['context_factor']}")

    print("=" * 92)
    print(f"검증 결과: {'ALL PASS ✅' if ok else 'FAIL ❌'}")
    print("=" * 92)

    print("\n[보고서 패킷 예시 — north-7F]")
    north = next(z for z in report["zones"] if z["zone_id"] == "north-7F")
    print(json.dumps(north, ensure_ascii=False, indent=2))
    print("\n[무결성]", report["integrity"]["hash"][:46], "...")


if __name__ == "__main__":
    main()
