#!/usr/bin/env python3
"""Batch adapter using the exact pipeline and storage used by the dashboard."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from model_infer import decode_image, ModelSet
from inspection_pipeline import analyze_and_store
from inspection_rules import load_criteria
from inspection_store import InspectionStore


def collect(target):
    target = Path(target)
    if target.is_file():
        return [target]
    return sorted(
        p
        for p in target.rglob("*")
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    )


def run_one(path, models, context, store, threshold=0.5, min_area=1, criteria=None):
    path = Path(path)
    try:
        raw = path.read_bytes()
        image = decode_image(raw, getattr(models, "inference_profile", "legacy"))
    except OSError:
        raw, image = None, None
    return analyze_and_store(
        image,
        models,
        context,
        store,
        str(path),
        threshold,
        min_area,
        source_bytes=raw,
        criteria=criteria,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target")
    parser.add_argument(
        "--model-pair", nargs=2, default=None, metavar=("CRACK_ID", "RUST_ID")
    )
    parser.add_argument("--crack", default=None)
    parser.add_argument("--corrosion", default=None)
    parser.add_argument(
        "--store", default=str(Path(__file__).parent / "inspection_data")
    )
    parser.add_argument("--criteria", default=None)
    parser.add_argument("--asset", default="미지정")
    parser.add_argument("--inspection", default="미지정")
    parser.add_argument("--component", default="unknown")
    parser.add_argument("--member-id", default="")
    parser.add_argument("--part", default="미상")
    parser.add_argument("--confirm-component", action="store_true")
    parser.add_argument(
        "--quality", choices=["usable", "unconfirmed", "poor"], default="unconfirmed"
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=1)
    args = parser.parse_args()
    criteria = load_criteria(args.criteria) if args.criteria else load_criteria()
    context = {
        "asset_id": args.asset,
        "inspection_id": args.inspection,
        "component_type": args.component,
        "component_instance_id": args.member_id,
        "part": args.part,
        "component_confirmed": args.confirm_component,
        "component_source": "manual",
        "quality": args.quality,
    }
    from inspection_rules import normalize_context

    normalize_context(context, criteria)
    if not 0 < args.threshold < 1 or args.min_area < 1:
        parser.error("임계값 또는 최소 영역 설정이 잘못됨")
    paths = collect(args.target)
    if not paths:
        parser.error("입력 사진을 찾지 못함")
    if args.model_pair:
        from model_comparison import registered_models, model_path, PROFILES

        if tuple(args.model_pair) not in PROFILES.values():
            parser.error("등록되지 않은 모델 조합임")
        if args.threshold != 0.5 or args.min_area != 1:
            parser.error("논문 비교는 임계값 0.5와 영역 제거 없음으로 실행해야 함")
        models = registered_models(
            args.model_pair,
            [
                args.crack or model_path(args.model_pair[0]),
                args.corrosion or model_path(args.model_pair[1]),
            ],
        )
    else:
        models = ModelSet(args.crack, args.corrosion)
    store = InspectionStore(args.store)
    for path in paths:
        result = run_one(
            path, models, context, store, args.threshold, args.min_area, criteria
        )
        print(
            f"{result['observation_id']} | {path.name} | {result['analysis_state']} | 우선순위 {result['overall_priority'] or '판단 보류'} | 후보 {len(result['defects'])}개"
        )


if __name__ == "__main__":
    main()
