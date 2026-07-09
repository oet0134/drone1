"""위험도 보고서 백엔드 — stdlib http.server (의존성 0).

보고서:
  POST /api/reports          보고서 패킷 수신(멱등: report_id로 1건 처리)
  GET  /api/reports          보고서 요약 목록
  GET  /api/reports/{id}     보고서 전체
  GET  /api/alerts           D/E 등급(고위험) 구역 경보
  POST /api/seed             예시 건물 데이터 채우기

학습(규칙→ML 데이터 축적):
  POST /api/labels           사람이 매긴 정답 등급 저장(구역별, 멱등) → 학습 레코드
  GET  /api/labels           라벨 목록
  GET  /api/training-summary 라벨 수·모델 일치율·등급 분포

일반 사진 분석(RGB, Mavic 3 등 일반 카메라) — 사람 판단 보조:
  POST /api/analyze-photo    사진 업로드(base64) → 의심 지점(균열형) 표시 (등급 X)
  POST /api/photo-grade      사람이 사진에 등급 지정(= 최종 판단 + 학습 데이터)
  GET  /api/photos           사진 분석 결과 목록
  GET  /uploads/{file}       의심 지점에 빨간 박스 그린 주석 이미지(JPG)

상시 감시(CCTV형 — 기준 대비 변화 감지):
  POST /api/monitor/baseline 현재 화면을 '정상 기준'으로 등록
  POST /api/monitor/frame    감시 한 컷 → 기준 대비 새 이상 → 정상/경보
  GET  /api/monitors         감시 지점별 상태·경보 목록
  GET  /monitor              감시 화면(카메라 실시간)

  GET  /  (또는 /dashboard)  웹 대시보드(HTML)

실행:  python backend/server.py [port]
"""
from __future__ import annotations
import base64
import hashlib
import json
import os
import socket
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, unquote

# backend/server.py를 직접 실행하면 sys.path가 backend/라서 drone_risk를 못 찾는다.
# 프로젝트 루트(backend의 상위)를 경로에 추가한다.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from drone_risk.sample_data import sample_buildings
from drone_risk.pipeline import assess_building
from drone_risk.rgb_analysis import analyze_photo, baseline_from, monitor_frame
from drone_risk.report import GRADE_ACTION
from drone_risk.risk_engine import worst_grade

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "reports")
LABELS = os.path.join(HERE, "data", "labels")
PHOTOS = os.path.join(HERE, "data", "photos")      # 사진 분석 결과(JSON)
UPLOADS = os.path.join(HERE, "data", "uploads")    # 주석 이미지(JPG)
MONITORS = os.path.join(HERE, "data", "monitors")  # 상시 감시 기준·상태(JSON)
DASHBOARD = os.path.join(HERE, "dashboard.html")
MONITOR_PAGE = os.path.join(HERE, "monitor.html")
for _d in (DATA, LABELS, PHOTOS, UPLOADS, MONITORS):
    os.makedirs(_d, exist_ok=True)

ALERT_GRADES = {"E", "D"}      # 최위험 두 등급 (국가 표준 방향)
VALID_GRADES = {"A", "B", "C", "D", "E", "HOLD"}


def _safe(s: str) -> str:
    return (s.replace(":", "_").replace("@", "_at_")
            .replace("/", "_").replace("\\", "_").replace(" ", "_"))


def _store(report: dict):
    fp = os.path.join(DATA, _safe(report["report_id"]) + ".json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False)


def _all_reports():
    out = []
    for x in sorted(os.listdir(DATA)):
        if x.endswith(".json"):
            with open(os.path.join(DATA, x), encoding="utf-8") as f:
                out.append(json.load(f))
    return out


def _summary(r: dict) -> dict:
    return {"report_id": r["report_id"], "building_id": r["building_id"],
            "captured_at": r["captured_at"], "overall_grade": r["overall_grade"],
            "zone_count": len(r["zones"])}


def _all_labels():
    out = []
    for x in sorted(os.listdir(LABELS)):
        if x.endswith(".json"):
            with open(os.path.join(LABELS, x), encoding="utf-8") as f:
                out.append(json.load(f))
    return out


def _all_photos():
    out = []
    for x in sorted(os.listdir(PHOTOS)):
        if x.endswith(".json"):
            with open(os.path.join(PHOTOS, x), encoding="utf-8") as f:
                out.append(json.load(f))
    return out


def _training_record(report, zone, human_grade, note):
    """구역의 지표 스냅샷 + 모델 등급 + 사람 정답 = 학습 1건."""
    t = zone["indicators"]["thermal"]
    u = zone["indicators"]["ultrasonic"]
    c = zone["contributions"]
    return {
        "report_id": report["report_id"], "building_id": report["building_id"],
        "zone_id": zone["zone_id"],
        "features": {
            "dt_max_c": t["dt_max_c"], "anomaly_area_m2": t["anomaly_area_m2"],
            "pattern": t["pattern"], "thermal_valid": t["valid"],
            "bulge_max_mm": u["bulge_max_mm"], "bulge_area_m2": u["bulge_area_m2"],
            "profile_disc": u["profile_disc"],
            "S_thermal": c["thermal"], "S_ultra": c["ultra"],
            "agree": c["agree"], "context_factor": c["context_factor"],
            "score": zone["score"], "confidence": zone["confidence"],
            "alt_m": zone.get("geo", {}).get("alt_m"),
        },
        "model_grade": zone["grade"], "human_grade": human_grade, "note": note,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False)
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/dashboard"):
            try:
                with open(DASHBOARD, encoding="utf-8") as f:
                    return self._send(200, f.read(), "text/html")
            except FileNotFoundError:
                return self._send(404, {"error": "dashboard not found"})
        if path == "/api/reports":
            return self._send(200, [_summary(r) for r in _all_reports()])
        if path.startswith("/api/reports/"):
            rid = unquote(path[len("/api/reports/"):])
            for r in _all_reports():
                if r["report_id"] == rid:
                    return self._send(200, r)
            return self._send(404, {"error": "not found"})
        if path == "/api/alerts":
            alerts = []
            for p in _all_photos():             # 사진: 확정(없으면 AI제안) D/E
                g = p.get("human_grade") or p.get("ai_grade")
                if g in ALERT_GRADES:
                    alerts.append({"building_id": p.get("building") or "현장",
                                   "zone_id": p.get("filename") or "사진", "grade": g,
                                   "action": GRADE_ACTION.get(g, "")})
            for x in sorted(os.listdir(MONITORS)):   # 상시감시: 경보 상태
                if x.endswith(".json"):
                    with open(os.path.join(MONITORS, x), encoding="utf-8") as f:
                        m = json.load(f)
                    if m.get("status") == "alert":
                        alerts.append({"building_id": m["id"], "zone_id": "상시감시",
                                       "grade": "E",
                                       "action": f"기준 대비 새 이상 {m.get('last_new', 0)}개 감지"})
            return self._send(200, alerts)
        if path == "/api/sites":                # 현장별(사진 building 기준) 위험 요약
            groups = {}
            for p in _all_photos():
                name = p.get("building") or "(이름없음)"
                groups.setdefault(name, []).append(p)
            out = []
            for name, items in groups.items():
                grades = [(p.get("human_grade") or p.get("ai_grade")) for p in items]
                grades = [g for g in grades if g]
                overall = worst_grade(grades) if grades else None
                out.append({"site": name, "overall": overall,
                            "photo_count": len(items),
                            "confirmed": sum(1 for p in items if p.get("human_grade"))})
            rank = {"E": 5, "D": 4, "HOLD": 3.5, "C": 3, "B": 2, "A": 1, None: 0}
            out.sort(key=lambda s: -rank.get(s["overall"], 0))
            return self._send(200, out)
        if path == "/api/labels":
            return self._send(200, _all_labels())
        if path == "/api/training-summary":
            # 사람이 확정한 것: (구역 라벨 - 레거시) + (사진 AI제안 vs 사람확정)
            pairs = [(r["model_grade"], r["human_grade"]) for r in _all_labels()]
            for p in _all_photos():
                if p.get("human_grade"):
                    pairs.append((p.get("ai_grade"), p["human_grade"]))
            agree = sum(1 for a, h in pairs if a == h)
            dist = {}
            for _, h in pairs:
                dist[h] = dist.get(h, 0) + 1
            rate = round(agree / len(pairs) * 100) if pairs else 0
            return self._send(200, {"labeled": len(pairs), "agree": agree,
                                    "agreement_rate": rate, "distribution": dist})
        if path in ("/monitor", "/monitor.html"):
            try:
                with open(MONITOR_PAGE, encoding="utf-8") as f:
                    return self._send(200, f.read(), "text/html")
            except FileNotFoundError:
                return self._send(404, {"error": "monitor page not found"})
        if path == "/api/monitors":
            out = []
            for x in sorted(os.listdir(MONITORS)):
                if x.endswith(".json"):
                    with open(os.path.join(MONITORS, x), encoding="utf-8") as f:
                        m = json.load(f)
                    out.append({k: m.get(k) for k in
                                ("id", "status", "last_new", "image_url",
                                 "updated", "alerts")})
            return self._send(200, out)
        if path == "/api/photos":
            return self._send(200, _all_photos())
        if path.startswith("/uploads/"):
            fn = _safe(unquote(path[len("/uploads/"):]))
            fp = os.path.join(UPLOADS, fn)
            if os.path.isfile(fp):
                with open(fp, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            return self._send(404, {"error": "not found"})
        return self._send(404, {"error": "unknown route"})

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self._body()
        except (ValueError, json.JSONDecodeError):
            return self._send(400, {"error": "bad json"})

        if path == "/api/reports":
            if "report_id" not in body:
                return self._send(400, {"error": "missing report_id"})
            _store(body)
            return self._send(201, {"status": "stored",
                                    "report_id": body["report_id"]})

        if path == "/api/seed":
            n = 0
            for scan in sample_buildings():
                _store(assess_building(scan))
                n += 1
            return self._send(201, {"status": "seeded", "buildings": n})

        if path == "/api/analyze-photo":
            b64 = body.get("image_b64", "")
            if "," in b64:                       # data URI 접두사 제거
                b64 = b64.split(",", 1)[1]
            try:
                img_bytes = base64.b64decode(b64)
                result = analyze_photo(img_bytes)
            except Exception as e:
                return self._send(400, {"error": f"analyze failed: {e}"})
            pid = hashlib.sha1(img_bytes).hexdigest()[:12]
            with open(os.path.join(UPLOADS, pid + ".jpg"), "wb") as f:
                f.write(result["annotated_jpg"])
            record = {
                "id": pid,
                "building": body.get("building", "현장 사진"),
                "filename": body.get("filename", ""),
                "ai_grade": result["suggested_grade"],       # AI 제안(참고용)
                "concern": result["concern"],
                "num_candidates": result["num_candidates"],  # 의심 지점 수
                "photo_quality": result["photo_quality"],
                "human_grade": None, "note": "",             # 사람이 확정(= 최종 + 학습)
                "image_url": f"/uploads/{pid}.jpg", "engine": result["engine"],
            }
            with open(os.path.join(PHOTOS, pid + ".json"), "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False)
            return self._send(201, record)

        if path == "/api/photo-grade":     # 사람이 사진에 등급 지정(= 판단 + 학습 데이터)
            pid = body.get("id")
            grade = body.get("grade")
            if grade not in VALID_GRADES:
                return self._send(400, {"error": "bad grade"})
            fp = os.path.join(PHOTOS, _safe(pid or "") + ".json")
            if not os.path.isfile(fp):
                return self._send(404, {"error": "photo not found"})
            with open(fp, encoding="utf-8") as f:
                rec = json.load(f)
            rec["human_grade"] = grade
            rec["note"] = body.get("note", "")
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False)
            return self._send(201, {"status": "graded", "id": pid,
                                    "human_grade": grade})

        if path == "/api/labels":
            rid = body.get("report_id"); zid = body.get("zone_id")
            grade = body.get("grade")
            if grade not in VALID_GRADES:
                return self._send(400, {"error": "bad grade"})
            report = next((r for r in _all_reports()
                           if r["report_id"] == rid), None)
            if not report:
                return self._send(404, {"error": "report not found"})
            zone = next((z for z in report["zones"]
                         if z["zone_id"] == zid), None)
            if not zone:
                return self._send(404, {"error": "zone not found"})
            rec = _training_record(report, zone, grade, body.get("note", ""))
            key = _safe(rid) + "__" + _safe(zid) + ".json"
            with open(os.path.join(LABELS, key), "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False)
            return self._send(201, {"status": "labeled",
                                    "model_grade": zone["grade"],
                                    "human_grade": grade})

        if path == "/api/monitor/baseline":     # '정상 상태' 기준 등록
            mid = _safe(body.get("monitor_id", "cam1") or "cam1")
            b64 = body.get("image_b64", "")
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            try:
                base = baseline_from(base64.b64decode(b64))
            except Exception as e:
                return self._send(400, {"error": f"baseline failed: {e}"})
            rec = {"id": mid, "centers": base["centers"],
                   "baseline_count": base["count"], "quality": base["quality"],
                   "status": "normal", "last_new": 0, "alerts": [],
                   "image_url": None, "updated": time.strftime("%H:%M:%S")}
            with open(os.path.join(MONITORS, mid + ".json"), "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False)
            return self._send(201, {"status": "baseline_set", "id": mid,
                                    "baseline_count": base["count"]})

        if path == "/api/monitor/frame":        # 감시 한 컷 → 기준 대비 새 이상
            mid = _safe(body.get("monitor_id", "cam1") or "cam1")
            fp = os.path.join(MONITORS, mid + ".json")
            if not os.path.isfile(fp):
                return self._send(409, {"error": "no baseline; set baseline first"})
            b64 = body.get("image_b64", "")
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            with open(fp, encoding="utf-8") as f:
                rec = json.load(f)
            try:
                res = monitor_frame(base64.b64decode(b64), rec["centers"])
            except Exception as e:
                return self._send(400, {"error": f"frame failed: {e}"})
            with open(os.path.join(UPLOADS, "mon_" + mid + ".jpg"), "wb") as f:
                f.write(res["annotated_jpg"])
            rec["status"] = res["status"]
            rec["last_new"] = res["new_count"]
            rec["updated"] = time.strftime("%H:%M:%S")
            rec["image_url"] = f"/uploads/mon_{mid}.jpg"
            if res["status"] == "alert":
                rec.setdefault("alerts", []).insert(
                    0, {"time": rec["updated"], "new": res["new_count"]})
                rec["alerts"] = rec["alerts"][:20]
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False)
            return self._send(200, {"status": res["status"],
                                    "new_count": res["new_count"],
                                    "total": res["total"],
                                    "photo_quality": res["photo_quality"],
                                    "image_url": rec["image_url"]})

        return self._send(404, {"error": "unknown route"})


def _lan_ip():
    """이 컴퓨터의 와이파이/랜 IP 주소를 알아낸다(다른 기기가 접속할 주소)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main():
    # 사용법: python server.py [port] [host]
    #   - 로컬: 기본 127.0.0.1(이 컴퓨터만). 0.0.0.0 이면 같은 네트워크 다른 기기도.
    #   - 클라우드(Render 등): PORT 환경변수가 주어짐 → 그 포트로 0.0.0.0 바인드.
    on_cloud = "PORT" in os.environ
    port = int(os.environ.get("PORT", sys.argv[1] if len(sys.argv) > 1 else 8765))
    host = sys.argv[2] if len(sys.argv) > 2 else ("0.0.0.0" if on_cloud else "127.0.0.1")

    srv = ThreadingHTTPServer((host, port), Handler)   # 이 시점에 포트가 열려 접속 가능
    print(f"backend listening on {host}:{port}  (dashboard at /)")
    if not on_cloud:
        print(f"  - 이 컴퓨터:  http://127.0.0.1:{port}/")
        if host == "0.0.0.0":
            print(f"  - 다른 기기:  http://{_lan_ip()}:{port}/   (같은 와이파이에서 접속)")
        # 서버가 준비된 뒤 브라우저를 연다(라이브러리 로딩 후라 '연결 안 됨' 방지)
        try:
            webbrowser.open(f"http://127.0.0.1:{port}/")
        except Exception:
            pass
    srv.serve_forever()


if __name__ == "__main__":
    main()
