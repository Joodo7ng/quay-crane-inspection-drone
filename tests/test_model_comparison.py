import hashlib
import json
import numpy as np
import pytest
from PIL import Image
from pathlib import Path
from model_comparison import registered_models, compare_upload, REGISTRY
from model_infer import _Infer, ModelSet, decode_image
from test_inspection import fixed_models
from inspection_store import InspectionStore
from dashboard import process_upload
import cv2


def test_hash_mismatch_rejected_before_load(tmp_path):
    p = tmp_path / "wrong.pt"
    p.write_bytes(b"wrong checkpoint")
    model = registered_models(("C1", "R1"), [str(p), str(p)])
    assert not model.models
    assert all(
        s["state"] == "load_failed" and "SHA256" in s["error"]
        for s in model.statuses.values()
    )
    assert model.statuses["균열"]["model"]["model_id"] == "C1"


def test_missing_candidate_does_not_substitute_baseline(tmp_path):
    model = registered_models(
        ("C1", "R1"), [str(tmp_path / "C1.pt"), str(tmp_path / "R1.pt")]
    )
    assert not model.models
    out = model.detect(np.zeros((20, 30, 3), np.uint8))
    assert out["overall_grade"] is None
    assert all(s["state"] == "load_failed" for s in out["model_status"].values())


def test_paper_settings_locked():
    model = ModelSet(inference_profile="paper")
    for threshold, area in [(0.4, 1), (0.5, 100)]:
        with pytest.raises(ValueError):
            model.detect(np.zeros((20, 30, 3), np.uint8), threshold, area)


def test_pair_group_originals_and_reload(tmp_path):
    image = np.full((80, 100, 3), 90, np.uint8)
    _, data = cv2.imencode(".png", image)
    mask = np.zeros((80, 100), np.uint8)
    mask[1:4, 2:8] = 1
    store = InspectionStore(tmp_path)
    results = compare_upload(
        data.tobytes(),
        "fixture.png",
        [("before", fixed_models()), ("after", fixed_models(crack=mask))],
        {"quality": "usable"},
        store,
    )
    before, after = [
        InspectionStore(tmp_path).get(x["observation_id"]) for x in results
    ]
    assert before["context"]["comparison_id"] == after["context"]["comparison_id"]
    assert before["observation_id"] != after["observation_id"]
    assert before["frame_sha256"] == after["frame_sha256"]
    assert len(before["defects"]) == 0 and len(after["defects"]) == 1
    assert (
        before["evidence"]["original"]["path"] != after["evidence"]["original"]["path"]
    )


def test_evaluation_masks_saved_separately(tmp_path):
    image = np.zeros((80, 100, 3), np.uint8)
    _, data = cv2.imencode(".png", image)
    models = fixed_models()
    models.models["균열"].native_mask = np.ones((512, 512), np.uint8)
    store = InspectionStore(tmp_path)
    result = process_upload(data.tobytes(), "fixture.png", models, {}, store)
    stored = store.get(result["observation_id"])
    with np.load(store.evidence_path(stored["evidence"]["masks"])) as arrays:
        assert arrays["evaluation_masks_균열"].shape == (512, 512)
        assert arrays["raw_masks_균열"].shape == (80, 100)
    assert "evaluation_masks" not in stored
