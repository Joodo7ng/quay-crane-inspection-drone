import copy
import hashlib
import json
from pathlib import Path
import uuid
import cv2
import numpy as np
import pytest
from model_infer import ModelSet, TYPES, image_fingerprint
from inspection_rules import load_criteria, matches
from inspection_pipeline import assess, analyze_and_store
from inspection_store import InspectionStore, read_legacy_csv
from detect_run import run_one
from dashboard import process_upload


class FixedModel:
    def __init__(self, mask=None, error=None):
        self.value, self.error = mask, error

    def mask(self, rgb, thr):
        if self.error:
            raise RuntimeError(self.error)
        return self.value


def fixed_models(crack=None, corrosion=None, fail=None, omitted=None, shape=(80, 100)):
    models = ModelSet()
    for kind, value in zip(TYPES, (crack, corrosion)):
        if kind == omitted:
            continue
        value = np.zeros(shape, np.uint8) if value is None else value
        models.models[kind] = FixedModel(
            value, "injected failure" if kind == fail else None
        )
        models.statuses[kind] = {
            "state": "ready",
            "error": None,
            "model": {"fixture": True, "sha256": None},
        }
    return models


@pytest.fixture
def image():
    return np.full((80, 100, 3), 160, np.uint8)


@pytest.fixture
def masks():
    crack = np.zeros((80, 100), np.uint8)
    crack[10, 10:15] = 1
    rust = np.zeros_like(crack)
    rust[30:50, 30:60] = 1
    rust[60:65, 70:75] = 1
    return crack, rust


@pytest.fixture
def context():
    return {
        "asset_id": "QC-TEST",
        "inspection_id": "TEST-01",
        "component_type": "leg",
        "component_instance_id": "LEG-01",
        "part": "연결부",
        "component_confirmed": True,
        "quality": "usable",
    }


@pytest.fixture
def store(tmp_path):
    return InspectionStore(tmp_path / "records")


def test_all_candidates_saved(image, masks, context, store):
    out = analyze_and_store(image, fixed_models(*masks), context, store, "fixture")
    assert len(out["defects"]) == 3
    assert {d["defect"] for d in out["defects"]} == set(TYPES)
    assert len({d["defect_id"] for d in out["defects"]}) == 3
    assert len(store.get(out["observation_id"])["defects"]) == 3


def test_crack_not_downgraded_by_small_area(image, masks, context):
    out = assess(fixed_models(*masks).detect(image), context)
    assert out["defects"][0]["area_px"] == 5
    assert out["defects"][0]["assessment"]["priority"] is None


def test_missing_model_not_normal(image, context):
    out = assess(fixed_models(omitted="부식").detect(image), context)
    assert out["analysis_state"] == "partial" and out["overall_priority"] is None
    assert out["model_status"]["부식"]["state"] == "not_loaded"


def test_exception_preserves_other_model(image, masks, context):
    out = assess(fixed_models(*masks, fail="부식").detect(image), context)
    assert out["analysis_state"] == "partial" and out["overall_priority"] is None
    assert len(out["defects"]) == 1
    assert out["model_status"]["부식"]["state"] == "failed"


def test_both_fail(image, context):
    models = fixed_models()
    for m in models.models.values():
        m.error = "broken"
    out = assess(models.detect(image), context)
    assert out["analysis_state"] == "failed" and out["overall_priority"] is None


def test_not_run(image, context):
    out = assess(ModelSet().detect(image), context)
    assert out["analysis_state"] == "not_run" and out["overall_priority"] is None


def test_load_failure_is_explicit(tmp_path, image, context):
    models = ModelSet(str(tmp_path / "missing.pt"))
    out = assess(models.detect(image), context)
    assert out["model_status"]["균열"]["state"] == "load_failed"
    assert out["overall_priority"] is None


def test_invalid_input_saved(context, store):
    out = analyze_and_store(None, fixed_models(), context, store, "invalid")
    assert out["analysis_state"] == "failed" and out["overall_priority"] is None
    assert not out["evidence"] and out["input_error"]


@pytest.mark.parametrize("quality", ["poor", "unconfirmed"])
def test_quality_not_normal(image, context, quality):
    context["quality"] = quality
    assert assess(fixed_models().detect(image), context)["overall_priority"] is None


def test_normal_requires_both_models(image, context):
    out = assess(fixed_models().detect(image), context)
    assert (
        out["overall_priority"] == "정상"
        and out["normal_basis"]["id"] == "TEAM-NO-CANDIDATE"
    )


def test_empty_untyped_result_not_normal(context):
    assert assess({"defects": []}, context)["overall_priority"] is None


def test_unknown_retains_candidates(image, masks, context):
    context.update(component_type="unknown", part="미상", component_confirmed=False)
    out = assess(fixed_models(*masks).detect(image), context)
    assert len(out["defects"]) == 3
    assert not out["defects"][0]["assessment"]["references"]
    assert any("부재" in x for x in out["defects"][0]["assessment"]["checks"])


def test_leg_body_vs_connection(image, masks, context):
    raw = fixed_models(*masks).detect(image)
    a = assess(raw, {**context, "part": "몸통"})
    b = assess(raw, context)
    assert a["defects"][1]["grade"] is None and b["defects"][1]["grade"] is None
    assert (
        "사용 중지 검토"
        not in a["defects"][0]["assessment"]["document_guidance"]["action"]
    )
    assert (
        "사용 중지 검토" in b["defects"][0]["assessment"]["document_guidance"]["action"]
    )
    assert "G9-8" not in [r["id"] for r in a["defects"][0]["assessment"]["references"]]
    assert "G9-8" in [r["id"] for r in b["defects"][0]["assessment"]["references"]]


@pytest.mark.parametrize(
    "component,part,reference",
    [
        ("rail", "레일 표면", "G47"),
        ("spreader", "로크 핀", "G50"),
        ("boom_hinge", "힌지핀", "G51"),
        ("bogie", "차륜", "G49"),
        ("forestay", "끝 연결부", "G48"),
    ],
)
def test_component_reference_lookup(image, masks, context, component, part, reference):
    out = assess(
        fixed_models(*masks).detect(image),
        {**context, "component_type": component, "part": part},
    )
    assert reference in [r["id"] for r in out["defects"][0]["assessment"]["references"]]


def test_unconfirmed_component_no_specific_rules(image, masks, context):
    out = assess(
        fixed_models(*masks).detect(image), {**context, "component_confirmed": False}
    )
    assert not out["defects"][0]["assessment"]["references"]
    assert "TEAM-CRITICAL-CRACK" not in out["defects"][0]["assessment"]["rule_ids"]


def test_name_only_no_component_area(image, masks, context):
    out = assess(fixed_models(*masks).detect(image), context)
    assert out["measurements"]["부식"]["component_area_pct"] is None


@pytest.mark.parametrize(
    "case,state",
    [
        ("zero", "empty"),
        ("shape", "shape_mismatch"),
        ("frame", "frame_mismatch"),
        ("values", "invalid_values"),
        ("unconfirmed", "component_unconfirmed"),
    ],
)
def test_invalid_component_masks(image, masks, context, case, state):
    mask = np.ones((80, 100), np.uint8)
    sha = image_fingerprint(image)
    if case == "zero":
        mask[:] = 0
    if case == "shape":
        mask = mask[:10]
    if case == "frame":
        sha = "different"
    if case == "values":
        mask[:] = 2
    if case == "unconfirmed":
        context["component_confirmed"] = False
    out = assess(
        fixed_models(*masks).detect(image),
        context,
        component_mask=mask,
        mask_frame_sha256=sha,
    )
    assert out["component_mask_state"] == state
    assert out["measurements"]["부식"]["component_area_pct"] is None
    assert len(out["defects"]) == 3


def test_union_intersection_not_bbox_sum(image, masks, context):
    region = np.zeros((80, 100), np.uint8)
    region[30:50, 30:70] = 1
    out = assess(
        fixed_models(*masks).detect(image),
        context,
        component_mask=region,
        mask_frame_sha256=image_fingerprint(image),
    )
    m = out["measurements"]["부식"]
    assert m["defect_union_px"] == 625 and m["component_intersection_px"] == 600
    assert m["component_area_pct"] == 75 and m["outside_component_px"] == 25
    assert len(out["defects"]) == 3


def test_bad_model_mask_fails(image, context):
    out = assess(fixed_models(crack=np.ones((2, 2), np.uint8)).detect(image), context)
    assert (
        out["model_status"]["균열"]["state"] == "failed"
        and out["overall_priority"] is None
    )


@pytest.mark.parametrize("area,expected", [(99, 0), (100, 1), (101, 1)])
def test_postprocess_inclusive_boundary(image, area, expected):
    mask = np.zeros((80, 100), np.uint8)
    mask.flat[:area] = 1
    out = fixed_models(crack=mask).detect(image, min_area=100)
    assert len(out["defects"]) == expected
    assert out["raw_masks"]["균열"].sum() == area
    assert out["settings"]["min_area_px"] == 100


def test_fixed_confidence_removed(image, masks):
    assert all(
        d["confidence"] is None for d in fixed_models(*masks).detect(image)["defects"]
    )


def test_ui_batch_equivalence(image, masks, context, store, tmp_path):
    path = tmp_path / "input.png"
    cv2.imwrite(str(path), image)
    a = process_upload(
        path.read_bytes(), path.name, fixed_models(*masks), context, store
    )
    b = run_one(path, fixed_models(*masks), context, store)
    assert a["criteria_sha256"] == b["criteria_sha256"]
    assert a["overall_priority"] == b["overall_priority"]
    assert [d["assessment"] for d in a["defects"]] == [
        d["assessment"] for d in b["defects"]
    ]
    assert a["measurements"] == b["measurements"]


def test_old_evidence_immutable(image, masks, context, store):
    a = analyze_and_store(image, fixed_models(*masks), context, store, "same.png")
    original = store.evidence_path(a["evidence"]["original"])
    raw = original.read_bytes()
    b = analyze_and_store(image + 1, fixed_models(*masks), context, store, "same.png")
    assert a["evidence"]["original"]["path"] != b["evidence"]["original"]["path"]
    assert original.read_bytes() == raw
    assert np.array_equal(cv2.imread(str(original)), image)


def test_review_and_action_reload(image, masks, context, store):
    original = analyze_and_store(image, fixed_models(*masks), context, store, "fixture")
    oid = original["observation_id"]
    did = original["defects"][0]["defect_id"]
    store.append_review(
        oid, did, "시험자", "false_positive", "정상", "이음선으로 확인함", "오탐 확인함"
    )
    store.append_review(
        oid,
        did,
        "재검토자",
        "confirmed",
        "위험",
        "근접 확인에서 균열 확인함",
        "정밀 점검 요청함",
    )
    store.append_action(oid, did, "시험자", "검사 요청을 전달함", "completed")
    fresh = InspectionStore(store.root).get(oid)
    assert fresh["defects"] == original["defects"] and len(fresh["history"]) == 3
    assert fresh["history"][0]["decision"] == "false_positive"
    assert fresh["history"][2]["completed_at"]


def test_context_correction_preserves_original(image, masks, context, store):
    original = analyze_and_store(image, fixed_models(*masks), context, store, "fixture")
    oid = original["observation_id"]
    did = original["defects"][1]["defect_id"]
    store.append_review(
        oid,
        did,
        "검토자",
        "deferred",
        None,
        "몸통 사진임을 확인함",
        "추가 확인함",
        {**context, "part": "몸통"},
    )
    new = store.get(oid)
    assert new["context"]["part"] == "연결부"
    assert new["history"][0]["reassessment"]["context"]["part"] == "몸통"
    assert new["history"][0]["reassessment"]["defects"][1]["grade"] is None


@pytest.mark.parametrize("reviewer,reason", [("", "사유"), ("확인자", "")])
def test_review_requires_audit_fields(image, context, store, reviewer, reason):
    out = analyze_and_store(image, fixed_models(), context, store, "fixture")
    with pytest.raises(ValueError):
        store.append_review(
            out["observation_id"], None, reviewer, "deferred", None, reason, "보류함"
        )


def test_rule_snapshot_preserved(image, masks, context, store):
    c = load_criteria()
    out = analyze_and_store(
        image, fixed_models(*masks), context, store, "fixture", criteria=c
    )
    c["version"] = "next"
    c["rules"][0]["reason"] = "changed"
    old = store.get(out["observation_id"])
    assert old["criteria_version"] == "2026-09-30.2"
    assert old["criteria_snapshot"]["rules"][0]["reason"] != "changed"


@pytest.mark.parametrize(
    "change",
    [
        {"status": "draft"},
        {"review_status": "unreviewed"},
        {"category": "external_guidance"},
    ],
)
def test_unreviewed_rules_disabled(image, masks, context, change):
    c = load_criteria()
    for r in c["rules"]:
        r.update(change)
    out = assess(fixed_models(*masks).detect(image), context, c)
    assert all(d["assessment"]["priority"] is None for d in out["defects"])


def test_legacy_read_only(tmp_path):
    path = tmp_path / "old.csv"
    path.write_text("id,grade,confidence\nold,정상,0.9\n")
    before = path.read_bytes()
    rows = read_legacy_csv(path)
    assert len(rows) == 1 and "legacy_notice" in rows[0] and path.read_bytes() == before


def test_evidence_tampering_detected(image, context, store):
    out = analyze_and_store(image, fixed_models(), context, store, "fixture")
    entry = out["evidence"]["original"]
    store.evidence_path(entry).write_bytes(b"changed")
    with pytest.raises(ValueError):
        store.evidence_path(entry)


def test_invalid_member_part_rejected(image, context):
    with pytest.raises(ValueError):
        assess(fixed_models().detect(image), {**context, "part": "존재하지 않는 부위"})


def test_criteria_has_provenance():
    c = load_criteria()
    assert not any(r["category"] == "external_guidance" for r in c["rules"])
    assert all(
        r["verification_status"] == "attachment_only"
        and not r["automatic_legal_judgment"]
        for r in c["references"].values()
    )
    assert c["source_attachment"]["sha256"] and c["excluded_quantitative_thresholds"]


def test_adapter_exception_persisted(image, context, store):
    class Broken:
        def detect(self, *args, **kwargs):
            raise RuntimeError("adapter failure")

    out = analyze_and_store(image, Broken(), context, store, "fixture")
    assert out["analysis_state"] == "failed" and out["overall_priority"] is None
    assert out["evidence"]["original"]


def test_missing_measurement_defers(image, masks, context):
    raw = fixed_models(*masks).detect(image)
    raw["defects"][0]["area_px"] = None
    out = assess(raw, context)
    assert out["defects"][0]["assessment"]["status"] == "deferred"


@pytest.mark.parametrize("partial", [False, True])
def test_outside_or_boundary_not_assigned(image, masks, context, partial):
    region = np.zeros(image.shape[:2], np.uint8)
    region[0:20, 0 : (12 if partial else 5)] = 1
    out = assess(
        fixed_models(*masks).detect(image),
        context,
        component_mask=region,
        mask_frame_sha256=image_fingerprint(image),
    )
    crack = out["defects"][0]
    assert not crack["assessment"]["references"]
    assert crack["assessment"]["priority"] is None
    assert ("경계 중첩" if partial else "미귀속") in crack["attribution"]
    assert len(out["defects"]) == 3


@pytest.mark.parametrize(
    "component,part,expected",
    [
        ("rail", "레일 표면", ["G47"]),
        ("spreader", "로크 핀", ["G50"]),
        ("boom_hinge", "힌지핀", ["G48", "G51"]),
    ],
)
def test_document_judgment_and_method_separated(
    image, masks, context, component, part, expected
):
    out = assess(
        fixed_models(*masks).detect(image),
        {**context, "component_type": component, "part": part},
    )
    g = out["defects"][0]["assessment"]["document_guidance"]
    assert g["judgment_ids"] == expected
    assert not set(g["method_ids"]) & set(expected)
    assert out["overall_priority"] is None


def test_old_criteria_keeps_original_priority(image, masks, context):
    old = load_criteria(Path(__file__).parents[1] / "criteria/inspection_v1.json")
    assert (
        assess(fixed_models(*masks).detect(image), context, old)["overall_priority"]
        == "주의"
    )
