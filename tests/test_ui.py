from pathlib import Path
from streamlit.testing.v1 import AppTest
from test_inspection import fixed_models
from inspection_pipeline import analyze_and_store
from inspection_store import InspectionStore
import numpy as np

ROOT = Path(__file__).parents[1]


def seed(tmp_path):
    store = InspectionStore(tmp_path / "ui_records")
    image = np.full((80, 100, 3), 180, np.uint8)
    mask = np.zeros((80, 100), np.uint8)
    mask[20, 20:30] = 1
    context = {
        "asset_id": "시험장비",
        "inspection_id": "UI-TEST",
        "component_type": "leg",
        "part": "연결부",
        "component_confirmed": True,
        "quality": "usable",
    }
    r = analyze_and_store(
        image, fixed_models(crack=mask), context, store, "고정 마스크 기능시험"
    )
    return store, r


def test_ui_empty_not_normal(tmp_path, monkeypatch):
    monkeypatch.setenv("INSPECTION_STORE", str(tmp_path / "empty"))
    app = AppTest.from_file(str(ROOT / "dashboard.py")).run(timeout=20)
    assert not app.exception
    assert any("대기 상태를 정상으로 표시하지 않음" in x.value for x in app.info)


def test_ui_shows_guidance_and_all_candidates(tmp_path, monkeypatch):
    store, r = seed(tmp_path)
    monkeypatch.setenv("INSPECTION_STORE", str(store.root))
    app = AppTest.from_file(str(ROOT / "dashboard.py")).run(timeout=20)
    assert not app.exception
    assert any("적용 기준과 판단 이유" in x.value for x in app.markdown)
    assert any("G48" in x.value for x in app.markdown)
    assert any("사용 중지 검토" in x.value for x in app.markdown)
    assert app.metric[0].value == "판단 보류"


def test_ui_review_submission(tmp_path, monkeypatch):
    store, r = seed(tmp_path)
    monkeypatch.setenv("INSPECTION_STORE", str(store.root))
    app = AppTest.from_file(str(ROOT / "dashboard.py")).run(timeout=20)
    for field in app.text_input:
        if field.label == "확인자":
            field.set_value("UI 검토자")
        if field.label == "최종 판단":
            field.set_value("이음선으로 확인함")
    for field in app.text_area:
        if field.label == "확인 또는 변경 사유":
            field.set_value("원본을 대조함")
    next(s for s in app.selectbox if s.label == "결함 확인 결과").set_value(
        "false_positive"
    )
    next(b for b in app.button if b.label == "확인 이력 저장").click()
    app.run(timeout=20)
    assert not app.exception
    assert store.get(r["observation_id"])["history"][0]["decision"] == "false_positive"


def test_ui_action_submission(tmp_path, monkeypatch):
    store, r = seed(tmp_path)
    monkeypatch.setenv("INSPECTION_STORE", str(store.root))
    app = AppTest.from_file(str(ROOT / "dashboard.py")).run(timeout=20)
    next(f for f in app.text_input if f.label == "조치 담당자").set_value("UI 담당자")
    next(f for f in app.text_area if f.label == "조치 내용").set_value(
        "추가 영상을 확보함"
    )
    next(s for s in app.selectbox if s.label == "처리 상태").set_value("completed")
    next(b for b in app.button if b.label == "조치 이력 추가").click()
    app.run(timeout=20)
    assert not app.exception
    assert store.get(r["observation_id"])["history"][0]["completed_at"]


def test_ui_latest_context_used_for_next_review(tmp_path, monkeypatch):
    store, r = seed(tmp_path)
    did = r["defects"][0]["defect_id"]
    store.append_review(
        r["observation_id"],
        did,
        "검토자",
        "deferred",
        None,
        "몸통으로 정정함",
        "확인 대기함",
        {**r["context"], "part": "몸통"},
    )
    monkeypatch.setenv("INSPECTION_STORE", str(store.root))
    app = AppTest.from_file(str(ROOT / "dashboard.py")).run(timeout=20)
    assert not app.exception
    assert next(
        s for s in app.selectbox if s.label == "부재 / 점검 부위 확인 또는 수정"
    ).value == ("leg", "몸통")
