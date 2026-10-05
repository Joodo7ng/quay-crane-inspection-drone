"""Append-only observations/reviews/actions, unique evidence files, legacy reads."""

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import time
import uuid
import cv2
import numpy as np
from inspection_rules import RANK


def serializable(result):
    return {
        k: v
        for k, v in result.items()
        if k not in ("masks", "raw_masks", "evaluation_masks", "component_mask")
    }


def dump(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


class InspectionStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "inspections.sqlite3"
        with self.connect() as conn:
            conn.executescript(
                """
              CREATE TABLE IF NOT EXISTS observations(id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, payload TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT, observation_id TEXT NOT NULL REFERENCES observations(id), kind TEXT NOT NULL, timestamp TEXT NOT NULL, payload TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS timings(observation_id TEXT PRIMARY KEY REFERENCES observations(id), storage_ms REAL NOT NULL);
              CREATE INDEX IF NOT EXISTS events_observation ON events(observation_id, seq);
            """
            )

    def connect(self):
        conn = sqlite3.connect(self.db, timeout=15)
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def save(self, result, image=None, source_bytes=None):
        t0 = time.perf_counter()
        from model_infer import draw, image_fingerprint

        ident = result["observation_id"]
        uuid.UUID(ident)
        if image is not None and image_fingerprint(image) != result.get("frame_sha256"):
            raise ValueError("관측과 저장할 영상이 일치하지 않음")
        folder = self.root / "evidence" / ident
        folder.mkdir(parents=True, exist_ok=False)
        evidence = {}
        try:
            if image is not None:
                for key, frame in (
                    ("original", image),
                    ("overlay", draw(image, result)),
                ):
                    ok, data = cv2.imencode(".png", frame)
                    if not ok:
                        raise OSError("영상 저장 실패")
                    raw = data.tobytes()
                    p = folder / (key + ".png")
                    p.write_bytes(raw)
                    evidence[key] = {
                        "path": str(p.relative_to(self.root)),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
            if source_bytes is not None:
                p = folder / "source.bin"
                p.write_bytes(source_bytes)
                evidence["source_file"] = {
                    "path": str(p.relative_to(self.root)),
                    "sha256": hashlib.sha256(source_bytes).hexdigest(),
                }
            arrays = {}
            for category in ("masks", "raw_masks", "evaluation_masks"):
                for kind, mask in result.get(category, {}).items():
                    arrays[category + "_" + kind] = mask
            if result.get("component_mask") is not None:
                arrays["component_mask"] = result["component_mask"]
            if arrays:
                buff = io.BytesIO()
                np.savez_compressed(buff, **arrays)
                raw = buff.getvalue()
                p = folder / "masks.npz"
                p.write_bytes(raw)
                evidence["masks"] = {
                    "path": str(p.relative_to(self.root)),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            payload = serializable(result)
            payload["evidence"] = evidence
            with self.connect() as conn:
                conn.execute(
                    "INSERT INTO observations VALUES(?,?,?)",
                    (ident, payload["timestamp"], dump(payload)),
                )
            # Measured through evidence encoding and observation commit; metrics write excluded.
            elapsed = (time.perf_counter() - t0) * 1000
            with self.connect() as conn:
                conn.execute("INSERT INTO timings VALUES(?,?)", (ident, elapsed))
            return ident
        except Exception:
            # Only this newly created incomplete observation is cleaned up.
            with self.connect() as conn:
                persisted = conn.execute(
                    "SELECT 1 FROM observations WHERE id=?", (ident,)
                ).fetchone()
            if not persisted:
                for p in folder.iterdir():
                    p.unlink()
                folder.rmdir()
            raise

    def get(self, ident):
        with self.connect() as conn:
            row = conn.execute(
                "SELECT payload FROM observations WHERE id=?", (ident,)
            ).fetchone()
            if not row:
                raise KeyError(ident)
            result = json.loads(row[0])
            events = conn.execute(
                "SELECT seq,kind,timestamp,payload FROM events WHERE observation_id=? ORDER BY seq",
                (ident,),
            ).fetchall()
            timing = conn.execute(
                "SELECT storage_ms FROM timings WHERE observation_id=?", (ident,)
            ).fetchone()
        result["history"] = [
            {"seq": s, "kind": k, "timestamp": t, **json.loads(p)}
            for s, k, t, p in events
        ]
        if timing:
            result["timings"]["storage_ms"] = timing[0]
        return result

    def list(self, asset_id=None, component_instance_id=None):
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT id,payload FROM observations ORDER BY timestamp DESC"
            ).fetchall()
        result = []
        for ident, payload in rows:
            d = json.loads(payload)
            c = d["context"]
            if asset_id and c["asset_id"] != asset_id:
                continue
            if (
                component_instance_id
                and c["component_instance_id"] != component_instance_id
            ):
                continue
            result.append(
                {
                    "id": ident,
                    "timestamp": d["timestamp"],
                    "source": d["source"],
                    "model_label": c.get("comparison_label")
                    or " + ".join(
                        (v.get("model") or {}).get("model_id", "기존 기록")
                        for v in d["model_status"].values()
                    ),
                    "asset_id": c["asset_id"],
                    "inspection_id": c["inspection_id"],
                    "component": c["component_name"],
                    "part": c["part"],
                    "priority": d["overall_priority"],
                    "analysis_state": d["analysis_state"],
                    "candidates": len(d["defects"]),
                }
            )
        return result

    def evidence_path(self, item):
        p = (self.root / item["path"]).resolve()
        if not p.is_relative_to(self.root):
            raise ValueError("증거 경로가 저장소를 벗어남")
        if hashlib.sha256(p.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("증거 파일이 원래 기록과 일치하지 않음")
        return p

    def raw_for(self, observation):
        result = {
            k: observation[k]
            for k in (
                "defects",
                "model_status",
                "image_shape",
                "frame_sha256",
                "settings",
                "observation_id",
            )
        }
        result.update(masks={}, raw_masks={})
        entry = observation["evidence"].get("masks")
        component_mask = None
        if entry:
            with np.load(self.evidence_path(entry), allow_pickle=False) as arrays:
                for category in ("masks", "raw_masks", "evaluation_masks"):
                    for kind in ("균열", "부식"):
                        key = category + "_" + kind
                        if key in arrays:
                            result[category][kind] = arrays[key]
                if "component_mask" in arrays:
                    component_mask = arrays["component_mask"]
        return result, component_mask

    def append_review(
        self,
        ident,
        defect_id,
        reviewer,
        decision,
        final_priority,
        reason,
        final_judgment,
        context=None,
    ):
        if not reviewer.strip() or not reason.strip() or not final_judgment.strip():
            raise ValueError("확인자, 사유 및 최종 판단을 모두 입력해야 함")
        if decision not in (
            "confirmed",
            "false_positive",
            "deferred",
        ) or final_priority not in (*RANK, None):
            raise ValueError("확인 결과 또는 우선순위가 올바르지 않음")
        original = self.get(ident)
        if defect_id is not None and defect_id not in [
            d["defect_id"] for d in original["defects"]
        ]:
            raise ValueError("관측에 없는 결함임")
        payload = {
            "target_defect_id": defect_id,
            "reviewer": reviewer.strip(),
            "decision": decision,
            "final_priority": final_priority,
            "reason": reason.strip(),
            "final_judgment": final_judgment.strip(),
        }
        if context is not None:
            from inspection_pipeline import assess

            raw, cm = self.raw_for(original)
            # A corrected context requires mask reconfirmation; never silently reuses
            # a region previously assigned to a different member.
            reassessed = assess(raw, context, original["criteria_snapshot"])
            payload["context"] = reassessed["context"]
            payload["reassessment"] = serializable(reassessed)
        self._append(ident, "review", payload)

    def append_action(self, ident, defect_id, reviewer, content, status):
        if (
            not reviewer.strip()
            or not content.strip()
            or status not in ("open", "in_progress", "completed")
        ):
            raise ValueError("담당자, 조치 내용 및 올바른 상태가 필요함")
        observation = self.get(ident)
        if defect_id is not None and defect_id not in [
            d["defect_id"] for d in observation["defects"]
        ]:
            raise ValueError("관측에 없는 결함임")
        self._append(
            ident,
            "action",
            {
                "target_defect_id": defect_id,
                "reviewer": reviewer.strip(),
                "content": content.strip(),
                "status": status,
                "completed_at": (
                    datetime.now(timezone.utc).isoformat()
                    if status == "completed"
                    else None
                ),
            },
        )

    def _append(self, ident, kind, payload):
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            last = conn.execute(
                "SELECT seq FROM events WHERE observation_id=? ORDER BY seq DESC LIMIT 1",
                (ident,),
            ).fetchone()
            payload["previous_event_seq"] = last[0] if last else None
            conn.execute(
                "INSERT INTO events(observation_id,kind,timestamp,payload) VALUES(?,?,?,?)",
                (ident, kind, datetime.now(timezone.utc).isoformat(), dump(payload)),
            )

    def export_json(self, ident):
        return dump(self.get(ident))


def read_legacy_csv(path):
    """Read-only: old priorities, fixed confidences and overwritten images are not re-certified."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [
            {
                **r,
                "legacy_notice": "구버전 기록: 당시 등급과 신뢰도 및 사진 불변성 미검증",
            }
            for r in csv.DictReader(f)
        ]
