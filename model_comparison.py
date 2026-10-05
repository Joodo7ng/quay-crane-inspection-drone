"""Explicit paper model registry; never substitute a different checkpoint."""

import json
import os
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REGISTRY = json.loads((ROOT / "criteria/model_registry.json").read_text())
PROFILES = {
    "기존 모델 · C0 + R0": ("C0", "R0"),
    "실사진 파인튜닝 · C1 + R1": ("C1", "R1"),
    "합성 비교 · C2 + R1": ("C2", "R1"),
    "부식 보완 비교 · C1 + 리플레이50%": ("C1", "R_REPLAY50"),
}


def model_path(model_id):
    if os.environ.get("MODEL_" + model_id):
        return os.environ["MODEL_" + model_id]
    if model_id == "C0":
        return os.environ.get("MON_CRACK", str(ROOT / "best_E5_tversky.pt"))
    if model_id == "R0" and os.environ.get("MON_CORROSION"):
        return os.environ["MON_CORROSION"]
    drive = os.environ.get("ICT_EXPERIMENT_ROOT")
    return str(
        Path(drive) / REGISTRY["models"][model_id]["drive_path"]
        if drive
        else ROOT / "weights/comparison" / (model_id + ".pt")
    )


def registered_models(ids, paths=None, profile="paper"):
    from model_infer import ModelSet

    paths = paths or [model_path(i) for i in ids]
    return ModelSet(
        *paths,
        model_ids=ids,
        expected_hashes=[REGISTRY["models"][i]["sha256"] for i in ids],
        inference_profile=profile
    )


def compare_upload(data, name, model_sets, context, store, criteria=None):
    from dashboard import process_upload

    group = str(uuid.uuid4())
    results = []
    for label, models in model_sets:
        results.append(
            process_upload(
                data,
                name,
                models,
                {**context, "comparison_id": group, "comparison_label": label},
                store,
                0.5,
                1,
                criteria,
            )
        )
    return results
