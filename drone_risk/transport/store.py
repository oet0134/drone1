"""미전송 보고서의 디스크 영구 보관 — 전원이 꺼져도 등급 결과 보존."""
from __future__ import annotations
import json
import os


def _key(report: dict) -> str:
    return (report["report_id"]
            .replace(":", "_").replace("@", "_at_").replace("/", "_"))


class PendingStore:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(path, exist_ok=True)

    def put(self, report: dict) -> str:
        fp = os.path.join(self.path, _key(report) + ".json")
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False)
        return fp

    def list(self):
        return [os.path.join(self.path, x)
                for x in sorted(os.listdir(self.path)) if x.endswith(".json")]

    def load(self, fp: str) -> dict:
        with open(fp, encoding="utf-8") as f:
            return json.load(f)

    def remove(self, fp: str):
        if os.path.exists(fp):
            os.remove(fp)
