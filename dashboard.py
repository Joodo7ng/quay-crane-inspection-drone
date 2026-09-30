#!/usr/bin/env python3
"""Crane visual inspection: evidence → workflow rules → inspector decision."""
import os
from pathlib import Path
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import cv2
import numpy as np
import streamlit as st
from inspection_rules import (
    load_criteria,
    RANK,
    document_guidance,
    normalize_context,
    DEFAULT_CRITERIA,
)
from inspection_pipeline import analyze_and_store
from inspection_store import InspectionStore, read_legacy_csv
from model_infer import decode_image, ModelSet, image_fingerprint

from model_comparison import (
    PROFILES,
    REGISTRY,
    model_path,
    registered_models,
    compare_upload,
)

ROOT = Path(__file__).resolve().parent
STATUS = {
    "success": "분석 완료",
    "ready": "모델 준비",
    "not_loaded": "모델 미지정",
    "not_run": "미실행",
    "failed": "분석 실패",
    "load_failed": "모델 로딩 실패",
    "complete": "전체 분석 완료",
    "partial": "일부 분석 미완료",
}
MASK_STATUS = {
    "not_provided": "마스크 없음",
    "valid": "동일 영상의 유효한 마스크",
    "component_unconfirmed": "부재 확인 필요",
    "shape_mismatch": "영상 크기 불일치",
    "frame_mismatch": "원본 프레임 불일치",
    "invalid_values": "잘못된 마스크 값",
    "empty": "부재 면적 0",
}
DECISIONS = {
    "deferred": "판단 보류",
    "confirmed": "결함 확인",
    "false_positive": "오탐 확인",
}
QUALITY = {
    "unconfirmed": "품질 미확인",
    "usable": "분석 가능 · 점검자 확인",
    "poor": "재촬영 필요",
}


def process_upload(
    data,
    name,
    models,
    context,
    store,
    threshold=0.5,
    min_area=1,
    criteria=None,
    mask=None,
):
    image = decode_image(data, getattr(models, "inference_profile", "legacy"))
    fingerprint = image_fingerprint(image) if image is not None else None
    return analyze_and_store(
        image,
        models,
        context,
        store,
        name,
        threshold,
        min_area,
        component_mask=mask,
        mask_frame_sha256=fingerprint if mask is not None else None,
        source_bytes=data,
        criteria=criteria,
    )


def local_time(value):
    return (
        datetime.fromisoformat(value)
        .astimezone(ZoneInfo("Asia/Seoul"))
        .strftime("%Y-%m-%d %H:%M:%S KST")
    )


def pct(value):
    return "계산 불가" if value is None else f"{value:.4f}%"


def show_result(observation, store):
    ident = observation["observation_id"]
    context = observation["context"]
    defects = observation["defects"]
    identities = [
        state.get("model") or {} for state in observation["model_status"].values()
    ]
    st.caption(
        "모델: "
        + " + ".join(x.get("model_id", "기존 기록") for x in identities)
        + " · "
        + observation["settings"].get("inference_profile", "기존 방식")
    )
    comparison_id = context.get("comparison_id")
    if comparison_id:
        with st.expander("같은 사진 모델 비교", expanded=True):
            with store.connect() as conn:
                import json

                siblings = [
                    json.loads(row[0])
                    for row in conn.execute(
                        "SELECT payload FROM observations WHERE json_extract(payload, '$.context.comparison_id') = ?",
                        (comparison_id,),
                    )
                ]
            for col, item in zip(st.columns(len(siblings)), siblings):
                with col:
                    st.write(item["context"].get("comparison_label", "비교 결과"))
                    entry = item.get("evidence", {}).get("overlay")
                    if entry:
                        st.image(
                            str(store.evidence_path(entry)), use_container_width=True
                        )
                    st.caption(
                        " / ".join(
                            k + ": " + STATUS.get(v["state"], v["state"])
                            for k, v in item["model_status"].items()
                        )
                    )
                    st.caption(
                        f"후보 {len(item['defects'])}개 · {item['observation_id'][:8]}"
                    )
                    with st.expander("모델 및 추론 조건"):
                        st.json(
                            {"모델": item["model_status"], "설정": item["settings"]},
                            expanded=False,
                        )

    latest = {}
    for event in observation["history"]:
        if event["kind"] == "review":
            latest[event["target_defect_id"]] = event
    checked = sum(
        latest.get(d["defect_id"], {}).get("decision")
        in ("confirmed", "false_positive")
        for d in defects
    )
    cards = st.columns(4)
    cards[0].metric("점검 우선순위", observation["overall_priority"] or "판단 보류")
    if observation["criteria_version"] == "2026-09-30.1":
        st.caption(
            "과거 기록의 팀 임시 우선순위임 · 현재 기준은 등급 자동 적용을 보류함"
        )
    cards[1].metric("균열 후보", f"{sum(d['defect']=='균열' for d in defects)}개")
    cards[2].metric("부식 후보", f"{sum(d['defect']=='부식' for d in defects)}개")
    cards[3].metric(
        "점검자 확인",
        f"{checked}/{len(defects)}" if defects else ("기록됨" if latest else "미확인"),
    )
    st.caption(
        f"{context['asset_id']}  /  {context['component_name']} · {context['part']}  /  {local_time(observation['timestamp'])}"
    )
    problems = [
        f"{k} {STATUS.get(v['state'],v['state'])}"
        for k, v in observation["model_status"].items()
        if v["state"] != "success"
    ]
    if context["quality"] != "usable":
        problems.append(QUALITY[context["quality"]])
    if problems:
        st.warning(" · ".join(problems))
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        key = st.radio(
            "영상 표시",
            ["overlay", "original"],
            format_func=lambda x: "결함 표시" if x == "overlay" else "원본",
            horizontal=True,
            key="image_" + ident,
            label_visibility="collapsed",
        )
        if key in observation["evidence"]:
            try:
                st.image(
                    str(store.evidence_path(observation["evidence"][key])),
                    use_container_width=True,
                )
            except (OSError, ValueError) as exc:
                st.error(str(exc))
        else:
            st.info("표시할 원본 영상이 없음")
        st.caption("C · 균열(빨강)  |  R · 부식(주황)  |  번호는 후보 목록과 대응함")
    with right:
        with st.container(border=True):
            targets = [None] + [d["defect_id"] for d in defects]
            names = {
                None: "관측 전체",
                **{
                    d["defect_id"]: f"후보 {i+1} · {d['defect']}"
                    for i, d in enumerate(defects)
                },
            }
            target = st.selectbox(
                "확인할 후보",
                targets,
                format_func=names.get,
                index=1 if defects else 0,
                key="target_" + ident,
            )
            selected = next((d for d in defects if d["defect_id"] == target), None)
            if selected:
                assessment = selected["assessment"]
                guidance = assessment.get("document_guidance")
                if not guidance:
                    current = load_criteria()
                    guidance_context = normalize_context(context, current)
                    if not assessment["references"]:
                        guidance_context["component_confirmed"] = False
                    guidance = document_guidance(selected, guidance_context, current)
                    st.caption("현재 지침 참고 · 기존 저장 판단은 유지함")
                st.markdown("**적용 조항**")
                st.write(" · ".join(guidance["judgment_ids"]) or "부재 확인 필요")
                st.caption(guidance["reason"])
                st.markdown("**다음 조치**")
                st.write(guidance["action"])
                if not context["component_confirmed"]:
                    st.caption("부재 확인 필요 · 공통 지침 적용")
                else:
                    st.caption(context["component_name"] + " · " + context["part"])
                m = observation["measurements"][selected["defect"]]
                st.caption(selected["defect"] + " 전체 면적비")
                a, b = st.columns(2)
                a.metric("영상 전체", pct(m["frame_area_pct"]))
                b.metric("부재 관측 영역", pct(m["component_area_pct"]))
                if m["component_area_pct"] is None:
                    st.caption("부재 영역 마스크 없음 또는 사용 불가")
                with st.expander("판단 근거 자세히"):
                    st.markdown("**적용 기준과 판단 이유**")
                    st.caption(guidance["source"] + " / " + guidance["version"])
                    st.caption(guidance["verification"])
                    st.write(
                        "판정 관련 조항: "
                        + (", ".join(guidance["judgment_ids"]) or "적용 보류")
                    )
                    st.write(
                        "위치·방법 근거: "
                        + (", ".join(guidance["method_ids"]) or "없음")
                    )
                    for reference in guidance["references"]:
                        st.write(reference["clause"] + " — " + reference["summary"])
                    st.caption(
                        "녹 면적 1% 보수 기준은 표준 원문 미확인으로 자동 적용하지 않음"
                    )
                    st.caption("후보 위치 연결: " + selected["attribution"])
                    for reason in assessment["reasons"]:
                        st.write("• " + reason)
                    st.markdown("**권고 조치 및 추가 확인**")
                    for item in assessment["actions"] + assessment["checks"]:
                        st.write("• " + item)
                    for rule in assessment["rules"]:
                        st.write(f"**{rule['id']} / {rule['version']} — 팀 자체 기준**")
                        st.caption(rule["scope"] + " / " + rule["reviewer"])
                        st.json(
                            {
                                "필요입력": rule["required_inputs"],
                                "일치조건": rule["conditions"],
                            },
                            expanded=False,
                        )
                    for ref in assessment["references"]:
                        st.write(f"**{ref['document']} · {ref['clause']}**")
                        st.caption(ref["scope"] + " / " + ref["verification_label"])
                        st.link_button("근거 문서 열기", ref["url"])
                    st.caption(
                        "면적비는 영상에 보이는 영역만 해당함. 전체 부재 손실률이 아님. 후보 신뢰도는 미제공임."
                    )
                    st.caption("기준 버전 " + observation["criteria_version"])
            else:
                st.write(observation["summary"])
                if observation.get("normal_basis"):
                    st.caption("두 모델 분석 완료 · 현재 영상에서 후보 미검출")
        st.caption("분석 상태")
        for kind, state in observation["model_status"].items():
            st.caption(f"{kind} · {STATUS.get(state['state'],state['state'])}")
    with st.expander(f"전체 후보 목록 · {len(defects)}개"):
        if defects:
            st.dataframe(
                [
                    {
                        "번호": i + 1,
                        "종류": d["defect"],
                        "후보 면적(px)": d["area_px"],
                        "영상 대비": pct(d["area_pct"]),
                        "우선순위": d["assessment"]["priority"] or "보류",
                        "위치 연결": d["attribution"],
                    }
                    for i, d in enumerate(defects)
                ],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.caption("검출된 후보 없음")
    st.caption("시스템의 점검 우선순위임. 최종 판단은 점검자가 기록함.")
    with st.expander("점검자 확인 및 조치"):
        previous = [
            e
            for e in observation["history"]
            if e["kind"] == "review" and e["target_defect_id"] == target
        ]
        if previous:
            last = previous[-1]
            st.success(
                f"최근 점검자 판단: {DECISIONS[last['decision']]} / 우선순위 {last['final_priority'] or '보류'} / {last['final_judgment']}"
            )
            st.caption(f"확인자 {last['reviewer']} · 사유 {last['reason']}")
        else:
            st.info("점검자 최종 판단이 아직 기록되지 않음.")
        review_context = previous[-1].get("context", context) if previous else context
        with st.form("review_" + ident + "_" + str(target)):
            reviewer = st.text_input(
                "확인자", key="reviewer_" + ident + "_" + str(target)
            )
            decision = st.selectbox(
                "결함 확인 결과", list(DECISIONS), format_func=DECISIONS.get
            )
            final_priority = st.selectbox(
                "점검자가 정한 최종 우선순위",
                [None, *RANK],
                format_func=lambda x: x or "판단 보류",
            )
            judgment = st.text_input(
                "최종 판단", placeholder="예: 근접 확인 후 비파괴검사 필요함"
            )
            reason = st.text_area("확인 또는 변경 사유")
            st.caption(
                "입력하는 최종 판단은 점검자의 기록임. 시스템 원분석 결과는 보존함."
            )
            pairs = [
                (k, p)
                for k, c in observation["criteria_snapshot"]["components"].items()
                for p in c["parts"]
            ]
            current = (review_context["component_type"], review_context["part"])
            pair = st.selectbox(
                "부재 / 점검 부위 확인 또는 수정",
                pairs,
                index=pairs.index(current),
                format_func=lambda x: observation["criteria_snapshot"]["components"][
                    x[0]
                ]["name"]
                + " / "
                + x[1],
            )
            member = st.text_input(
                "개별 부재 ID 확인 또는 수정",
                value=review_context["component_instance_id"],
            )
            confirmed = st.checkbox(
                "부재와 후보 위치를 확인함", value=review_context["component_confirmed"]
            )
            quality = st.selectbox(
                "영상 품질 확인",
                list(QUALITY),
                index=list(QUALITY).index(review_context["quality"]),
                format_func=QUALITY.get,
            )
            if st.form_submit_button("확인 이력 저장", type="primary"):
                updated = {
                    **review_context,
                    "component_type": pair[0],
                    "part": pair[1],
                    "component_instance_id": member,
                    "component_confirmed": confirmed,
                    "quality": quality,
                }
                try:
                    store.append_review(
                        ident,
                        target,
                        reviewer,
                        decision,
                        final_priority,
                        reason,
                        judgment,
                        updated,
                    )
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
        with st.form("action_" + ident + "_" + str(target)):
            who = st.text_input("조치 담당자")
            action = st.text_area("조치 내용")
            state = st.selectbox(
                "처리 상태",
                ["open", "in_progress", "completed"],
                format_func=lambda x: {
                    "open": "예정",
                    "in_progress": "진행 중",
                    "completed": "완료",
                }[x],
            )
            if st.form_submit_button("조치 이력 추가"):
                try:
                    store.append_action(ident, target, who, action, state)
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
        with st.expander("확인 및 조치 이력", expanded=False):
            for event in observation["history"]:
                if event["target_defect_id"] not in (target, None):
                    continue
                st.write(
                    f"**{local_time(event['timestamp'])} · {event['reviewer']} · {'판단 확인' if event['kind']=='review' else '조치'}**"
                )
                if event["kind"] == "review":
                    st.write(
                        f"{DECISIONS[event['decision']]} / {event['final_priority'] or '보류'} — {event['final_judgment']}"
                    )
                    st.caption("사유: " + event["reason"])
                    if event.get("reassessment"):
                        st.caption(
                            "수정 정보로 재적용한 시스템 우선순위: "
                            + (event["reassessment"]["overall_priority"] or "판단 보류")
                        )
                        revised = next(
                            (
                                d
                                for d in event["reassessment"]["defects"]
                                if d["defect_id"] == event["target_defect_id"]
                            ),
                            None,
                        )
                        if revised:
                            st.write(
                                "재적용 근거: "
                                + " / ".join(revised["assessment"]["reasons"])
                            )
                        st.caption(
                            "수정된 부재: "
                            + event["context"]["component_name"]
                            + " / "
                            + event["context"]["part"]
                        )
                else:
                    st.write(event["content"] + " · " + event["status"])
                    if event["completed_at"]:
                        st.caption("완료 시각: " + event["completed_at"])
        st.download_button(
            "관측·근거·이력 내려받기",
            store.export_json(ident),
            file_name=ident + ".json",
            mime="application/json",
        )
        with st.expander("처리 시간 및 기술 상세"):
            st.json(observation["timings"])
            st.caption("화면 렌더링 및 네트워크 전송 시간은 위 수치에 포함하지 않음.")
            st.json(
                {
                    "모델": observation["model_status"],
                    "설정": observation["settings"],
                    "기준해시": observation["criteria_sha256"],
                    "입력오류": observation.get("input_error"),
                }
            )


def main():
    st.set_page_config(page_title="안벽크레인 점검 지원", layout="wide")
    st.markdown(
        """<style>
.block-container {padding-top:1.5rem; max-width:1480px;}
h1 {font-size:1.9rem !important; letter-spacing:-.04em;}
[data-testid="stMetric"] {background:#f5f7fa; border:1px solid #e7ebf0; border-radius:10px; padding:12px 16px;}
[data-testid="stMetricValue"] {font-size:1.7rem;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap:.65rem;}
[data-testid="stAppViewContainer"] {background:#fff;}
</style>""",
        unsafe_allow_html=True,
    )
    st.title("안벽크레인 점검")
    criteria = load_criteria()
    store = InspectionStore(
        os.environ.get("INSPECTION_STORE", str(ROOT / "inspection_data"))
    )
    with st.sidebar:
        st.subheader("새 점검")
        with st.expander("장비 및 회차"):
            asset = st.text_input("장비 ID", value="CRANE-01")
            inspection = st.text_input("점검 회차 ID", value=time.strftime("%Y%m%d"))
            member = st.text_input("개별 부재 ID", placeholder="예: LEG-01")
        component = st.selectbox(
            "부재 종류",
            list(criteria["components"]),
            format_func=lambda k: criteria["components"][k]["name"],
        )
        part = st.selectbox(
            "세부 점검 부위", criteria["components"][component]["parts"]
        )
        confirmed = st.checkbox(
            "부재 및 촬영 위치를 확인함", disabled=component == "unknown"
        )
        quality = st.selectbox("영상 품질", list(QUALITY), format_func=QUALITY.get)
        mode = st.selectbox(
            "입력 소스", ["사진 업로드", "영상 파일", "카메라", "화면 캡처"]
        )
        selection = st.selectbox(
            "학습 모델", [*PROFILES, "직접 지정 · 기존 대시보드 방식"]
        )
        selected_ids = PROFILES.get(selection)
        with st.expander("모델 및 분석 설정"):
            crack = st.text_input(
                "균열 모델 경로",
                value=model_path(selected_ids[0] if selected_ids else "C0"),
                key="crack_" + selection,
            )
            corrosion = st.text_input(
                "부식 모델 경로",
                value=model_path(selected_ids[1] if selected_ids else "R0"),
                key="rust_" + selection,
            )
            threshold = st.number_input(
                "픽셀 이진화 임계값", 0.01, 0.99, 0.5, 0.01, disabled=bool(selected_ids)
            )
            min_area = st.number_input(
                "최소 연결영역 픽셀 수", 1, 100000, 1, disabled=bool(selected_ids)
            )
            if selected_ids:
                threshold, min_area = 0.5, 1
                st.caption(
                    "논문 비교: PIL 512×512 · 임계값 >0.5 · 영역 제거 없음. 표시용 마스크만 원본 크기로 변환함."
                )
                absent = [
                    mid
                    for mid, path in zip(selected_ids, [crack, corrosion])
                    if not Path(path).is_file()
                ]
                if absent:
                    st.warning("가중치 파일 필요: " + ", ".join(absent))
                st.caption("최신 후보가 모든 조건에서 더 우수함을 의미하지 않음.")
    if st.session_state.get("input_mode") != mode:
        cap = st.session_state.pop("capture", None)
        if cap is not None:
            cap.release()
        st.session_state.live = False
        st.session_state.input_mode = mode
    context = {
        "asset_id": asset,
        "inspection_id": inspection,
        "component_type": component,
        "component_instance_id": member,
        "part": part,
        "component_confirmed": confirmed,
        "component_source": "manual",
        "quality": quality,
    }

    @st.cache_resource
    def cached_models(crack, corrosion, signatures, selected_ids):
        return (
            registered_models(selected_ids, [crack, corrosion])
            if selected_ids
            else ModelSet(crack or None, corrosion or None)
        )

    def models():
        signatures = tuple(
            (
                (p, Path(p).stat().st_mtime_ns, Path(p).stat().st_size)
                if p and Path(p).is_file()
                else (p, None, None)
            )
            for p in (crack, corrosion)
        )
        return cached_models(crack, corrosion, signatures, selected_ids)

    input_tab, record_tab, criteria_tab = st.tabs(
        ["영상 분석", "기록 및 점검자 확인", "기준 관리"]
    )
    with input_tab:
        if mode == "사진 업로드":
            uploads = st.file_uploader(
                "점검 사진",
                type=["png", "jpg", "jpeg", "bmp", "webp"],
                accept_multiple_files=True,
            )
            with st.expander("선택 사항: 부재 관측 영역 마스크"):
                st.caption(
                    "사진 한 장에만 사용함. 동일 크기의 흑백 PNG(0/255)가 필요하며 흰색이 부재 영역임. 업로드 시 현재 사진에 대응함을 직접 확인해야 함."
                )
                mask_file = st.file_uploader("부재 마스크 PNG", type=["png"])
                mask_confirm = st.checkbox(
                    "이 마스크가 현재 사진과 해당 부재의 관측 영역에 대응함을 확인함"
                )
            compare = st.checkbox(
                "같은 사진으로 기존 모델과 비교",
                disabled=not selected_ids or selected_ids == ("C0", "R0"),
            )
            if st.button("사진 분석 및 저장", type="primary"):
                if not uploads:
                    st.warning("사진을 선택해야 함")
                elif mask_file and (
                    len(uploads) != 1 or not mask_confirm or not confirmed
                ):
                    st.error(
                        "부재 마스크 사용은 사진 1장, 부재 확인 및 마스크 대응 확인이 필요함"
                    )
                else:
                    mask = None
                    if mask_file:
                        mask = cv2.imdecode(
                            np.frombuffer(mask_file.getvalue(), np.uint8),
                            cv2.IMREAD_GRAYSCALE,
                        )
                        if mask is None or not np.isin(mask, [0, 255]).all():
                            st.error("0/255 흑백 마스크를 읽을 수 없음")
                            st.stop()
                        mask = (mask > 0).astype(np.uint8)
                    with st.spinner("분석과 기록 저장 중"):
                        model_set = models()
                        for uploaded in uploads:
                            if (
                                compare
                                and selected_ids
                                and selected_ids != ("C0", "R0")
                            ):
                                if mask is not None:
                                    st.error(
                                        "모델 비교는 부재 마스크 없이 실행함. 부재 마스크를 해제해야 함"
                                    )
                                    st.stop()
                                baseline = cached_models(
                                    model_path("C0"),
                                    model_path("R0"),
                                    tuple(
                                        (
                                            p,
                                            (
                                                Path(p).stat().st_mtime_ns
                                                if Path(p).is_file()
                                                else None
                                            ),
                                        )
                                        for p in [model_path("C0"), model_path("R0")]
                                    ),
                                    ("C0", "R0"),
                                )
                                pair = compare_upload(
                                    uploaded.getvalue(),
                                    uploaded.name,
                                    [("C0 + R0", baseline), (selection, model_set)],
                                    context,
                                    store,
                                    criteria,
                                )
                                result = pair[-1]
                            else:
                                result = process_upload(
                                    uploaded.getvalue(),
                                    uploaded.name,
                                    model_set,
                                    context,
                                    store,
                                    threshold,
                                    min_area,
                                    criteria,
                                    mask,
                                )
                            st.session_state.selected_observation = result[
                                "observation_id"
                            ]
                    st.success(
                        f"{len(uploads) * (2 if compare and selected_ids and selected_ids != ('C0', 'R0') else 1)}개 관측을 저장함. 기록 탭에서 전체 후보 및 근거를 확인할 수 있음."
                    )
        else:
            source = (
                st.text_input("영상 파일 경로")
                if mode == "영상 파일"
                else (
                    st.number_input("카메라 번호", 0, 10, 0)
                    if mode == "카메라"
                    else None
                )
            )
            interval = st.slider("분석 간격(초)", 0.5, 5.0, 1.0, 0.5)
            a, b = st.columns(2)
            if a.button("분석 시작"):
                old = st.session_state.pop("capture", None)
                if old is not None:
                    old.release()
                st.session_state.live = False
                if mode != "화면 캡처":
                    cap = cv2.VideoCapture(source)
                    if not cap.isOpened():
                        cap.release()
                        st.error(
                            "영상 소스를 열지 못함. 분석하지 않았으며 정상 기록을 생성하지 않음."
                        )
                    else:
                        st.session_state.capture = cap
                        st.session_state.live = True
                else:
                    st.session_state.live = True
                if st.session_state.live:
                    st.session_state.live_mode = mode
                    st.session_state.live_source = (
                        str(source) if source is not None else "primary-monitor"
                    )
                    st.session_state.live_context = dict(context)
                    st.session_state.live_models = models()
                    st.session_state.live_settings = (threshold, min_area)
            if b.button("분석 중지"):
                st.session_state.live = False
                cap = st.session_state.pop("capture", None)
                if cap is not None:
                    cap.release()

            @st.fragment(run_every=interval if st.session_state.get("live") else None)
            def live_view():
                if not st.session_state.get("live"):
                    return
                current_mode = st.session_state.live_mode
                frame = None
                try:
                    if current_mode == "화면 캡처":
                        import mss

                        with mss.mss() as capture:
                            frame = cv2.cvtColor(
                                np.array(capture.grab(capture.monitors[1])),
                                cv2.COLOR_BGRA2BGR,
                            )
                    else:
                        ok, frame = st.session_state.capture.read()
                        if not ok:
                            st.session_state.live = False
                            st.session_state.capture.release()
                            st.warning(
                                "영상 종료 또는 프레임 취득 실패로 분석을 중지함. 정상으로 기록하지 않음."
                            )
                            return
                    th, minimum = st.session_state.live_settings
                    result = analyze_and_store(
                        frame,
                        st.session_state.live_models,
                        st.session_state.live_context,
                        store,
                        current_mode
                        + ":"
                        + st.session_state.live_source
                        + (
                            f"#frame={int(st.session_state.capture.get(cv2.CAP_PROP_POS_FRAMES))};ms={st.session_state.capture.get(cv2.CAP_PROP_POS_MSEC):.1f}"
                            if current_mode == "영상 파일"
                            else ""
                        ),
                        th,
                        minimum,
                        criteria=criteria,
                    )
                    st.session_state.selected_observation = result["observation_id"]
                    st.image(
                        str(store.evidence_path(result["evidence"]["overlay"])),
                        use_container_width=True,
                    )
                    st.write(
                        f"점검 우선순위: {result['overall_priority'] or '판단 보류'} · {len(result['defects'])}개 후보"
                    )
                    st.caption(
                        "각 분석 프레임을 독립 관측으로 저장함. 프레임 간 동일 결함 추적은 미구현임. 기록 탭의 목록 새로고침으로 조회함."
                    )
                except Exception as exc:
                    st.session_state.live = False
                    st.error("영상 분석 중지: " + str(exc))

            live_view()
    with record_tab:
        st.button("기록 목록 새로고침")
        rows = store.list()
        st.caption(f"관측 {len(rows)}건")
        if rows:
            ids = [r["id"] for r in rows]
            chosen = st.session_state.get("selected_observation")
            index = ids.index(chosen) if chosen in ids else 0
            labels = {
                r[
                    "id"
                ]: f"{local_time(r['timestamp'])} · {Path(r['source']).name[:28]} · {r['component']} / {r['part']} · {r['model_label']}"
                for r in rows
            }
            ident = st.selectbox(
                "저장 관측 선택", ids, index=index, format_func=labels.get
            )
            show_result(store.get(ident), store)
        else:
            st.info("아직 분석된 관측이 없음. 대기 상태를 정상으로 표시하지 않음.")
        with st.expander("구버전 CSV 읽기 전용 조회"):
            legacy = read_legacy_csv(ROOT / "detections.csv")
            st.caption(
                "과거 기록을 수정하거나 새 기록을 이어 쓰지 않음. 구버전 고정 신뢰도와 사진 보존 상태는 검증되지 않음."
            )
            if legacy:
                st.dataframe(legacy, hide_index=True, use_container_width=True)
    with criteria_tab:
        st.subheader("적용 기준 " + criteria["version"])
        st.info(
            "팀 운영 규칙은 후보 확인 순서만 정함. 외부 지침의 원문과 실제 적용 대상은 별도 검토가 필요함."
        )
        st.dataframe(
            [
                {
                    "ID": r["id"],
                    "분류": "팀 자체 운영 기준",
                    "판정": r["priority"],
                    "범위": r["scope"],
                    "검토": r["reviewer"],
                }
                for r in criteria["rules"]
            ],
            hide_index=True,
            use_container_width=True,
        )
        st.write("자동 적용하지 않는 수치")
        st.dataframe(
            criteria["excluded_quantitative_thresholds"],
            hide_index=True,
            use_container_width=True,
        )
        st.download_button(
            "기준 데이터 내려받기",
            DEFAULT_CRITERIA.read_bytes(),
            file_name=DEFAULT_CRITERIA.name,
            mime="application/json",
        )


if __name__ == "__main__":
    main()
