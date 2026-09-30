"""Shared upload/live/batch assessment; no Streamlit or persistence dependency."""

from copy import deepcopy
from datetime import datetime, timezone
import time
import uuid
import numpy as np
import cv2
from inspection_rules import RANK, load_criteria, normalize_context, judge_defect

TYPES = ("균열", "부식")


def assess(
    raw, context=None, criteria=None, component_mask=None, mask_frame_sha256=None
):
    started = time.perf_counter()
    criteria = criteria or load_criteria()
    context = normalize_context(context, criteria)
    statuses = deepcopy(raw.get("model_status", {}))
    for kind in TYPES:
        statuses.setdefault(kind, {"state": "not_run", "error": "분석 상태 미제공"})
    shape = raw.get("image_shape")
    h, w = shape if shape and len(shape) == 2 else (0, 0)
    valid_shape = isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0
    measurements = {}
    errors = []
    cm = None
    mask_state = "not_provided"
    if component_mask is not None:
        candidate = np.asarray(component_mask)
        if not context["component_confirmed"]:
            mask_state = "component_unconfirmed"
        elif not valid_shape or candidate.shape != (h, w):
            mask_state = "shape_mismatch"
        elif not mask_frame_sha256 or mask_frame_sha256 != raw.get("frame_sha256"):
            mask_state = "frame_mismatch"
        elif not np.isin(candidate, [0, 1]).all():
            mask_state = "invalid_values"
        elif not candidate.any():
            mask_state = "empty"
        else:
            cm = candidate.astype(bool)
            mask_state = "valid"
    for kind in TYPES:
        m = raw.get("masks", {}).get(kind)
        if m is not None:
            m = np.asarray(m)
            if not valid_shape or m.shape != (h, w) or not np.isin(m, [0, 1]).all():
                errors.append(f"{kind} 마스크 형식 불일치")
                statuses[kind].update(state="failed", error="마스크 형식 불일치")
                m = None
        ok = m is not None and statuses[kind]["state"] == "success"
        measurements[kind] = {
            "frame_area_px": int(h * w) if valid_shape else None,
            "defect_union_px": int(np.count_nonzero(m)) if ok else None,
            "frame_area_pct": (
                float(np.count_nonzero(m) / (h * w) * 100) if ok else None
            ),
            "component_observed_area_px": int(cm.sum()) if cm is not None else None,
            "component_intersection_px": (
                int(np.count_nonzero(m.astype(bool) & cm))
                if ok and cm is not None
                else None
            ),
            "component_area_pct": (
                float(np.count_nonzero(m.astype(bool) & cm) / cm.sum() * 100)
                if ok and cm is not None
                else None
            ),
            "outside_component_px": (
                int(np.count_nonzero(m.astype(bool) & ~cm))
                if ok and cm is not None
                else None
            ),
            "area_meaning": "영상 내 후보 합집합 / 영상 전체; 부재값은 보이는 부재 마스크 영역만 해당함",
        }
    labels_by_kind = {}
    if cm is not None:
        for kind in TYPES:
            original_mask = raw.get("raw_masks", {}).get(kind)
            if original_mask is not None:
                original_mask = np.asarray(original_mask)
                if (
                    original_mask.shape == (h, w)
                    and np.isin(original_mask, [0, 1]).all()
                ):
                    labels_by_kind[kind] = cv2.connectedComponentsWithStats(
                        original_mask.astype(np.uint8), 8
                    )[1]
    defects = []
    for i, d in enumerate(raw.get("defects", [])):
        defect = {k: deepcopy(v) for k, v in d.items() if k != "assessment"}
        defect.setdefault("defect_id", str(uuid.uuid4()))
        defect["candidate_index"] = i
        defect["confidence"] = None  # No calibrated candidate probability exists.
        kind = defect.get("defect")
        if kind not in TYPES:
            errors.append("지원하지 않는 결함 종류임")
            continue
        area = defect.get("area_px")
        if (
            not isinstance(area, (int, float))
            or not np.isfinite(area)
            or not valid_shape
            or not 0 < area <= h * w
        ):
            defect["area_px"] = None
            defect["area_pct"] = None
        else:
            defect["area_pct"] = float(area / (h * w) * 100)
        defect_context = context
        defect["component_overlap_px"] = None
        defect["attribution"] = (
            "점검자 지정 관측 위치; 개별 후보 귀속은 확인 필요함"
            if context["component_confirmed"]
            else "미귀속"
        )
        if cm is not None:
            labels = labels_by_kind.get(kind)
            component_label = defect.get("component_label")
            if (
                labels is not None
                and isinstance(component_label, int)
                and component_label > 0
            ):
                candidate_region = labels == component_label
                overlap = int(np.count_nonzero(candidate_region & cm))
                total = int(np.count_nonzero(candidate_region))
                defect["component_overlap_px"] = overlap
                if total and overlap == total:
                    defect["attribution"] = "확인된 부재 마스크 안에 포함됨"
                else:
                    defect["attribution"] = (
                        "부재 영역 밖·미귀속"
                        if overlap == 0
                        else "경계 중첩·귀속 확인 필요함"
                    )
                    defect_context = {**context, "component_confirmed": False}
            else:
                defect["attribution"] = "후보 영역 대응 정보 없음·미귀속"
                defect_context = {**context, "component_confirmed": False}
        defect["assessment"] = judge_defect(defect, defect_context, criteria)
        if statuses[kind]["state"] != "success":
            defect["assessment"].update(priority=None, status="deferred")
            defect["assessment"]["checks"].append("해당 모델 분석 성공 여부를 확인함")
        defect["grade"] = defect["assessment"]["priority"]
        defects.append(defect)
    complete = (
        valid_shape
        and all(statuses[k]["state"] == "success" for k in TYPES)
        and not errors
    )
    all_missing = all(statuses[k]["state"] in ("not_loaded", "not_run") for k in TYPES)
    state = (
        "complete"
        if complete
        else (
            "not_run"
            if all_missing
            else (
                "partial"
                if any(statuses[k]["state"] == "success" for k in TYPES)
                else "failed"
            )
        )
    )
    priorities = [
        d["assessment"]["priority"] for d in defects if d["assessment"]["priority"]
    ]
    candidate_priority = max(priorities, key=RANK.get, default=None)
    normal_policy = criteria["normal_policy"]
    no_candidates = not defects and all(
        measurements[k]["defect_union_px"] == 0 for k in TYPES
    )
    normal_allowed = (
        complete
        and context["quality"] == "usable"
        and no_candidates
        and normal_policy["status"] == "active"
        and normal_policy["review_status"] == "implementation_reviewed"
    )
    overall = None
    if (
        complete
        and context["quality"] == "usable"
        and all(d["assessment"]["status"] == "applied" for d in defects)
    ):
        overall = (
            candidate_priority if defects else ("정상" if normal_allowed else None)
        )
    return {
        "schema_version": 1,
        "observation_id": raw.get("observation_id", str(uuid.uuid4())),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "context": context,
        "frame_sha256": raw.get("frame_sha256"),
        "image_shape": shape,
        "analysis_state": state,
        "model_status": statuses,
        "input_error": raw.get("input_error"),
        "overall_priority": overall,
        "candidate_priority": candidate_priority,
        "normal_basis": deepcopy(normal_policy) if normal_allowed else None,
        "summary": (
            "현재 영상에서 결함 후보 미검출임"
            if normal_allowed
            else "점검자 확인이 필요함"
        ),
        "defects": defects,
        "measurements": measurements,
        "component_mask_state": mask_state,
        "component_mask": cm.astype(np.uint8) if cm is not None else None,
        "masks": raw.get("masks", {}),
        "raw_masks": raw.get("raw_masks", {}),
        "evaluation_masks": raw.get("evaluation_masks", {}),
        "settings": deepcopy(raw.get("settings", {})),
        "criteria_version": criteria["version"],
        "criteria_sha256": criteria["sha256"],
        "criteria_snapshot": deepcopy(criteria),
        "errors": errors,
        "timings": {"assessment_ms": (time.perf_counter() - started) * 1000},
    }


def analyze_and_store(
    image,
    models,
    context,
    store,
    source,
    threshold=0.5,
    min_area=1,
    component_mask=None,
    mask_frame_sha256=None,
    source_bytes=None,
    criteria=None,
):
    """The sole application entry point, used by both UI and CLI."""
    t0 = time.perf_counter()
    try:
        raw = models.detect(image, thr=threshold, min_area=min_area)
    except Exception as exc:
        from model_infer import image_fingerprint

        raw = {
            "defects": [],
            "masks": {},
            "raw_masks": {},
            "image_shape": None,
            "input_error": f"{type(exc).__name__}: {exc}",
            "settings": {"threshold": threshold, "min_area_px": min_area},
            "model_status": {k: {"state": "failed", "error": str(exc)} for k in TYPES},
        }
        try:
            raw["frame_sha256"] = image_fingerprint(image)
            raw["image_shape"] = list(image.shape[:2])
        except ValueError:
            pass
    infer_ms = (time.perf_counter() - t0) * 1000
    result = assess(raw, context, criteria, component_mask, mask_frame_sha256)
    result["source"] = source
    result["timings"]["detect_call_ms"] = infer_ms
    store.save(result, image, source_bytes=source_bytes)
    return store.get(result["observation_id"])
