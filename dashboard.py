#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dashboard.py — 안벽 크레인 균열·부식 실시간 관제 대시보드
실시간 영상 + 종합 위험등급 + 알림 + Detection Info + 탐지 기록을 한 화면에.

실행:  streamlit run dashboard.py
사이드바에서 입력 소스(카메라 / 화면캡처 / 영상파일) 고르고 ▶ 시작.
"""
import csv
import json
import os
import time
import uuid
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import model_infer

CSV = "detections.csv"
STATUS = "status.json"
COLS = ["id", "timestamp", "source", "type", "grade", "area_pct",
        "confidence", "bbox", "image_path", "overall_grade", "status"]

CLR = {"위험": "#d32f2f", "주의": "#ef8f00", "경미": "#1565c0", "정상": "#2e7d32"}
BGC = {"위험": "#fdecea", "주의": "#fff5e0", "경미": "#e8f0fb", "정상": "#eaf4ec"}
MASK_BGR = {"균열": (0, 0, 220), "부식": (0, 140, 255)}
BOX_BGR = {"위험": (60, 60, 214), "주의": (0, 170, 235),
           "경미": (192, 101, 21), "정상": (60, 160, 40)}
EN = {"위험": "DANGER", "주의": "CAUTION", "경미": "MINOR", "정상": "NONE"}
RANK = {"정상": 0, "경미": 1, "주의": 2, "위험": 3}


def grade_of(d):
    """결함 하나의 등급. 모델이 '정상'이라 해도 '탐지된 것'이므로 최소 '경미'."""
    g = d.get("grade", "정상")
    return "경미" if g == "정상" else g


def overall_of(res):
    """이미지 종합 등급. 결함이 하나도 없을 때만 '정상'."""
    defs = res.get("defects", [])
    if not defs:
        return "정상"
    return max((grade_of(d) for d in defs), key=lambda g: RANK[g])

st.set_page_config(page_title="안벽 크레인 안전 진단 관제", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""
<style>
#MainMenu, footer {visibility:hidden;}
[data-testid="stToolbar"] {visibility:hidden;}
[data-testid="stExpandSidebarButton"], [data-testid="stExpandSidebarButton"] *,
[data-testid="stSidebarCollapsedControl"], [data-testid="stSidebarCollapsedControl"] *
  {visibility:visible !important; opacity:1 !important; z-index:9999 !important;}
[data-testid="stExpandSidebarButton"] button,
[data-testid="stSidebarCollapsedControl"] button {background:#fff !important;
  border:1px solid #bbb !important; border-radius:6px !important;
  box-shadow:0 1px 4px rgba(0,0,0,.2) !important;}
.block-container {padding-top:2.4rem; padding-bottom:1rem; max-width:1650px;}
.title {font-size:1.5rem; font-weight:700; margin-bottom:.6rem;}
.big {border-radius:8px; padding:16px 20px; text-align:center; margin-bottom:10px;}
.big .g {font-size:2.9rem; font-weight:800; line-height:1.1;}
.big .s {font-size:.88rem; letter-spacing:1px;}
.alert {border-radius:8px; padding:10px 13px; margin-bottom:7px;
  box-shadow:0 1px 3px rgba(0,0,0,.12); font-size:.98rem;}
.alert .t {float:right; font-size:.78rem; opacity:.7; font-weight:400;}
.info {border:1px solid #e0e0e0; border-radius:8px; padding:10px 14px; background:#fafafa;}
.info .row {display:flex; justify-content:space-between; padding:5px 0;
  border-bottom:1px solid #eee; font-size:.98rem;}
.info .row:last-child {border-bottom:none;}
.info .k {color:#666;}
.info .v {font-weight:700; font-family:ui-monospace,Menlo,monospace;}
.sec {font-size:1.05rem; font-weight:700; margin:10px 0 6px;}
table.rec {width:100%; border-collapse:collapse; font-size:.92rem;}
table.rec th {background:#f2f2ef; text-align:left; padding:6px 10px;
  border-bottom:1px solid #ddd; font-size:.82rem; color:#555;}
table.rec td {padding:6px 10px; border-bottom:1px solid #eee;}
table.rec td.m {font-family:ui-monospace,Menlo,monospace;}
</style>
""", unsafe_allow_html=True)


def find_model(kind="crack"):
    """폴더에서 모델 파일(.pt/.onnx)을 자동으로 찾는다."""
    import glob
    here = os.path.dirname(os.path.abspath(__file__))
    files = [f for e in ("pt", "onnx") for f in glob.glob(os.path.join(here, f"*.{e}"))]
    corr = [f for f in files if "corros" in os.path.basename(f).lower()
            or "rust" in os.path.basename(f).lower()]
    if kind == "corrosion":
        return os.path.basename(corr[0]) if corr else ""
    cand = [f for f in files if f not in corr]
    for key in ("crack_seg", "crack", "tversky", "best"):          # 우선순위
        for f in cand:
            if key in os.path.basename(f).lower():
                return os.path.basename(f)
    return os.path.basename(cand[0]) if cand else "crack_seg.pt"


@st.cache_resource
def load_model(crack, corrosion):
    model_infer.load_models(crack, corrosion or None)
    return True


def load_status():
    if os.path.exists(STATUS):
        try:
            return json.load(open(STATUS, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_status(s):
    json.dump(s, open(STATUS, "w", encoding="utf-8"), ensure_ascii=False)


def annotate(frame, res):
    img = frame.copy()
    for dt, m in res.get("masks", {}).items():
        if m is None:
            continue
        c = MASK_BGR.get(dt, (0, 0, 220))
        img[m == 1] = (0.45 * np.array(c) + 0.55 * img[m == 1]).astype(np.uint8)
    for d in res.get("defects", []):
        x, y, w, h = d["bbox"]
        g = grade_of(d)
        c = BOX_BGR.get(g, (200, 200, 200))
        cv2.rectangle(img, (x, y), (x + w, y + h), c, 3)
        cv2.putText(img, f"CRACK {d['area_pct']:.1f}% {EN.get(g,'')}",
                    (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    return img


def log_row(res, img_path, source="LIVE"):
    new = not os.path.exists(CSV)
    with open(CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        defs = res.get("defects", [])
        worst = max(defs, key=lambda d: d["area_pct"], default=None)
        base = {"id": uuid.uuid4().hex[:8], "timestamp": now, "source": source,
                "image_path": img_path, "status": "pending",
                "overall_grade": overall_of(res)}
        if worst:
            x, y, bw, bh = worst["bbox"]
            base.update({"type": worst["defect"], "grade": grade_of(worst),
                         "area_pct": worst["area_pct"], "confidence": worst["confidence"],
                         "bbox": f"{x},{y},{bw},{bh}"})
        else:
            base.update({"type": "점검", "grade": "정상", "area_pct": 0.0,
                         "confidence": "", "bbox": ""})
        w.writerow(base)


def read_df():
    if not os.path.exists(CSV):
        return pd.DataFrame(columns=COLS)
    d = pd.read_csv(CSV, dtype=str).fillna("")
    if d.empty:
        return d
    d["area_pct"] = pd.to_numeric(d["area_pct"], errors="coerce").fillna(0.0)
    stt = load_status()
    d["done"] = d["id"].map(lambda i: stt.get(i, False))
    return d


def dot(g):
    return (f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
            f'background:{CLR.get(g,"#888")};margin-right:6px"></span>'
            f'<span style="color:{CLR.get(g,"#555")};font-weight:700">{g}</span>')


# ---------------- 사이드바 ----------------
with st.sidebar:
    st.markdown("### 설정")
    modes = ["카메라", "화면 캡처(폰 미러링)", "영상 파일", "사진 업로드"]
    dm = os.environ.get("MON_MODE")
    mode = st.radio("입력 소스", modes, index=modes.index(dm) if dm in modes else 0)
    cam_idx = st.number_input("카메라 번호", 0, 5, 0, disabled=(mode != "카메라"))
    vid_path = st.text_input("영상 파일 경로", os.environ.get("MON_VIDEO", ""),
                             disabled=(mode != "영상 파일"))
    reg_str = ""
    if mode == "화면 캡처(폰 미러링)":
        reg_str = st.text_input("캡처 영역 x,y,w,h (비우면 전체 화면)",
                                os.environ.get("MON_REGION", ""),
                                help="폰 미러링 창만 잡으려면 그 창의 위치·크기를 입력. "
                                     "전체 화면을 잡으면 대시보드 자신이 찍혀 '무한 거울'이 됩니다.")
        preview = st.button("영역 미리보기", use_container_width=True)
        st.caption("시작 전에 눌러서 폰 화면만 잡히는지 확인하세요.")
    uploaded = None
    if mode == "사진 업로드":
        uploaded = st.file_uploader("사진 선택 (여러 장 가능)",
                                    type=["jpg", "jpeg", "png", "bmp", "webp"],
                                    accept_multiple_files=True)
    st.divider()
    crack_p = st.text_input("균열 모델", os.environ.get("MON_CRACK") or find_model("crack"))
    corr_p = st.text_input("부식 모델(선택)", find_model("corrosion"))
    if crack_p and not os.path.exists(crack_p):
        st.error(f"모델 파일 없음: {crack_p}")
    interval = st.slider("탐지 주기(초)", 0.1, 2.0, 0.3, 0.1)
    st.divider()
    c1, c2 = st.columns(2)
    start = c1.button("▶ 시작", use_container_width=True)
    stop = c2.button("■ 중지", use_container_width=True)
    if st.button("기록 초기화", use_container_width=True):
        for p in (CSV, STATUS):
            if os.path.exists(p):
                os.remove(p)
        st.session_state.alerts = deque(maxlen=6)
        st.rerun()

if "run" not in st.session_state:
    st.session_state.run = os.environ.get("MON_AUTOSTART") == "1"
if "alerts" not in st.session_state:
    st.session_state.alerts = deque(maxlen=6)
if start:
    st.session_state.run = True
if stop:
    st.session_state.run = False
running = st.session_state.run

st.markdown(
    f'<div class="title"><span style="color:{"#2e7d32" if running else "#aaa"}">●</span> '
    f'안벽 크레인 균열·부식 실시간 관제 '
    f'<span style="font-size:.88rem;color:#777">'
    f'{"감시 중" if running else "대기 중 — 사이드바에서 ▶ 시작"}</span></div>',
    unsafe_allow_html=True)

vid_col, side_col = st.columns([0.64, 0.36])
with vid_col:
    frame_slot = st.empty()
with side_col:
    grade_slot = st.empty()
    info_slot = st.empty()
    st.markdown('<div class="sec">알림</div>', unsafe_allow_html=True)
    alert_slot = st.empty()

st.markdown('<div class="sec">탐지 기록</div>', unsafe_allow_html=True)
rec_slot = st.empty()


def render_side(res, fps, latency):
    g = overall_of(res)
    defs = res.get("defects", [])
    n_c = sum(1 for d in defs if d["defect"] == "균열")
    n_r = sum(1 for d in defs if d["defect"] == "부식")
    area = max([d["area_pct"] for d in defs], default=0.0)
    conf = max([d["confidence"] for d in defs], default=0.0)
    grade_slot.markdown(
        f'<div class="big" style="background:{BGC[g]};border:2px solid {CLR[g]}">'
        f'<div class="s" style="color:{CLR[g]}">종합 안전 상태</div>'
        f'<div class="g" style="color:{CLR[g]}">{g}</div></div>', unsafe_allow_html=True)
    info_slot.markdown(
        f'<div class="info">'
        f'<div class="row"><span class="k">균열 탐지</span><span class="v">{n_c} 건</span></div>'
        f'<div class="row"><span class="k">부식 탐지</span><span class="v">{n_r} 건</span></div>'
        f'<div class="row"><span class="k">최대 면적</span><span class="v">{area:.2f} %</span></div>'
        f'<div class="row"><span class="k">신뢰도</span><span class="v">{conf:.2f}</span></div>'
        f'<div class="row"><span class="k">FPS</span><span class="v">{fps:.1f}</span></div>'
        f'<div class="row"><span class="k">지연시간</span><span class="v">{latency:.0f} ms</span></div>'
        f'</div>', unsafe_allow_html=True)
    html = "".join(
        f'<div class="alert" style="background:{BGC[g2]};border-left:5px solid {CLR[g2]}">'
        f'<b style="color:{CLR[g2]}">{msg}</b><span class="t">{t}</span></div>'
        for g2, msg, t in list(st.session_state.alerts))
    alert_slot.markdown(html or '<div style="color:#999;font-size:.9rem">알림 없음</div>',
                        unsafe_allow_html=True)


def render_records_static(limit=12):
    d = read_df()
    if d.empty:
        rec_slot.markdown('<div style="color:#999">기록 없음</div>', unsafe_allow_html=True)
        return
    tot = len(d)
    nd, nw = int((d["grade"] == "위험").sum()), int((d["grade"] == "주의").sum())
    nm = int((d["grade"] == "경미").sum())
    rows = ""
    for _, r in d.iloc[::-1].head(limit).iterrows():
        rows += (f'<tr><td>{dot(r["grade"])}</td><td>{r["type"]}</td>'
                 f'<td class="m">{r["area_pct"]:.2f}%</td><td class="m">{r["source"]}</td>'
                 f'<td class="m">{r["timestamp"][11:]}</td></tr>')
    rec_slot.markdown(
        f'<div style="color:#666;font-size:.9rem;margin-bottom:6px">'
        f'총 {tot}건 · <span style="color:{CLR["위험"]}">위험 {nd}</span> · '
        f'<span style="color:{CLR["주의"]}">주의 {nw}</span> · '
        f'<span style="color:{CLR["경미"]}">경미 {nm}</span> (최근 {limit}건)</div>'
        f'<table class="rec"><tr><th>등급</th><th>종류</th><th>면적</th>'
        f'<th>소스</th><th>시각</th></tr>{rows}</table>', unsafe_allow_html=True)


# ---------------- 캡처 영역 미리보기 ----------------
if mode == "화면 캡처(폰 미러링)" and 'preview' in dir() and preview:
    try:
        import mss
        with mss.mss() as _s:
            _reg = _s.monitors[1]
            if reg_str.strip():
                v = [int(t) for t in reg_str.replace(" ", ",").split(",") if t.strip()][:4]
                if len(v) == 4:
                    _reg = {"left": v[0], "top": v[1], "width": v[2], "height": v[3]}
            _img = cv2.cvtColor(np.array(_s.grab(_reg)), cv2.COLOR_BGRA2BGR)
        frame_slot.image(cv2.cvtColor(_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption(f"미리보기 — 캡처 크기 {_img.shape[1]}×{_img.shape[0]}px. "
                   f"폰 화면만 보이면 OK. 잘렸거나 다른 게 보이면 좌표를 조정하세요. "
                   f"(레티나 화면이면 값을 2배로 넣어야 할 수 있습니다)")
    except Exception as e:
        frame_slot.error(f"미리보기 실패: {e}")
    render_side({"overall_grade": "정상", "defects": []}, 0.0, 0.0)
    st.stop()

# ---------------- 사진 업로드 모드 ----------------
if mode == "사진 업로드":
    if running:
        if not uploaded:
            st.session_state.run = False
            frame_slot.warning("사이드바에서 사진을 먼저 선택하세요.")
        else:
            load_model(crack_p, corr_p)
            os.makedirs("caps", exist_ok=True)
            prog = st.progress(0.0, text="탐지 중...")
            results = []
            for i, uf in enumerate(uploaded, 1):
                buf = np.frombuffer(uf.getvalue(), np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                t0 = time.time()
                res = model_infer.detect(img)
                lat = (time.time() - t0) * 1000
                p = os.path.join("caps", f"up_{os.path.splitext(uf.name)[0][:30]}.png")
                cv2.imwrite(p, annotate(img, res))
                log_row(res, p, source=uf.name)
                results.append((p, res, lat, uf.name))
                g = overall_of(res)
                if RANK[g] >= 1:
                    lbl = {"위험": "위험 균열 감지", "주의": "균열 감지 (주의)",
                           "경미": "경미한 균열 감지"}.get(g, "균열 감지")
                    st.session_state.alerts.appendleft(
                        (g, f"{uf.name} — {lbl}", datetime.now().strftime("%H:%M:%S")))
                prog.progress(i / len(uploaded), text=f"탐지 중... {i}/{len(uploaded)}")
            prog.empty()
            if results:
                p, res, lat, nm = max(results, key=lambda r: RANK[overall_of(r[1])])
                st.session_state.last_upload = (p, res, lat, nm)
            st.session_state.run = False
            st.rerun()

# ---------------- 대기 상태: 기록 조작 가능 ----------------
if not running:
    lu = st.session_state.get("last_upload")
    if mode == "사진 업로드" and lu:
        p, res, lat, nm = lu
        if os.path.exists(p):
            frame_slot.image(p, use_container_width=True)
        st.caption(f"업로드 탐지 결과: {nm}")
        render_side(res, 0.0, lat)
    else:
        if mode == "사진 업로드":
            frame_slot.info("사이드바에서 사진을 선택한 뒤 **▶ 시작**을 누르세요.")
        else:
            frame_slot.info("사이드바에서 **▶ 시작**을 누르면 실시간 탐지가 시작됩니다.")
        render_side({"overall_grade": "정상", "defects": []}, 0.0, 0.0)
    d = read_df()
    with rec_slot.container():
        if d.empty:
            st.markdown('<div style="color:#999">기록 없음</div>', unsafe_allow_html=True)
        else:
            stt = load_status()
            tot = len(d)
            st.markdown(
                f'<div style="color:#666;font-size:.9rem">총 {tot}건 · '
                f'<span style="color:{CLR["위험"]}">위험 {int((d["grade"]=="위험").sum())}</span> · '
                f'<span style="color:{CLR["주의"]}">주의 {int((d["grade"]=="주의").sum())}</span> · '
                f'<span style="color:{CLR["경미"]}">경미 {int((d["grade"]=="경미").sum())}</span> · '
                f'미조치 {int((~d["done"]).sum())}</div>', unsafe_allow_html=True)
            d["_o"] = d["grade"].map(RANK).fillna(0)
            view = d.sort_values(["done", "_o"], ascending=[True, False]).head(20)
            for _, r in view.iterrows():
                c = st.columns([0.13, 0.10, 0.11, 0.24, 0.14, 0.13, 0.15])
                c[0].markdown(dot(r["grade"]), unsafe_allow_html=True)
                c[1].write(r["type"])
                c[2].write(f'{r["area_pct"]:.2f}%')
                c[3].caption(r["source"])
                c[4].caption(r["timestamp"][11:])
                if c[5].button("보기", key=f"v_{r['id']}") and r["image_path"] \
                        and os.path.exists(r["image_path"]):
                    frame_slot.image(r["image_path"], use_container_width=True)
                done = c[6].checkbox("조치", value=bool(r["done"]), key=f"c_{r['id']}")
                if done != bool(r["done"]):
                    stt[r["id"]] = done
                    save_status(stt)
                    st.rerun()
    st.stop()

# ---------------- 실행 ----------------
load_model(crack_p, corr_p)
os.makedirs("caps", exist_ok=True)

grab = cap = None
if mode == "화면 캡처(폰 미러링)":
    import mss
    sct = mss.mss()
    reg = sct.monitors[1]
    if reg_str.strip():
        try:
            x, y, w, h = [int(v.strip()) for v in reg_str.replace(" ", ",").split(",") if v.strip()][:4]
            reg = {"left": x, "top": y, "width": w, "height": h}
        except Exception:
            frame_slot.warning(f"캡처 영역 형식이 잘못됨: '{reg_str}' → 전체 화면으로 진행합니다.")

    def grab():
        return cv2.cvtColor(np.array(sct.grab(reg)), cv2.COLOR_BGRA2BGR)
else:
    src = int(cam_idx) if mode == "카메라" else vid_path
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        frame_slot.error(f"소스를 열 수 없습니다: {src}")
        st.session_state.run = False
        st.stop()

last_det = last_log = last_rec = 0.0
last_res = {"defects": [], "masks": {}, "overall_grade": "정상"}
fps = latency = 0.0
t_prev = time.time()
prev_grade = "정상"

while st.session_state.run:
    if grab:
        frame = grab()
    else:
        ok, frame = cap.read()
        if not ok:
            if mode == "영상 파일":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

    now = time.time()
    if now - last_det >= interval:
        t0 = time.time()
        last_res = model_infer.detect(frame)
        latency = (time.time() - t0) * 1000
        last_det = now
        g = overall_of(last_res)
        if RANK[g] > RANK[prev_grade] and RANK[g] >= 1:
            msg = {"위험": "위험 균열이 감지되었습니다",
                   "주의": "균열이 감지되었습니다 (주의)",
                   "경미": "경미한 균열이 감지되었습니다"}.get(g, "균열이 감지되었습니다")
            st.session_state.alerts.appendleft((g, msg, datetime.now().strftime("%H:%M:%S")))
        prev_grade = g
        if now - last_log >= 1.5:
            last_log = now
            p = "caps/live.png"
            cv2.imwrite(p, annotate(frame, last_res))
            log_row(last_res, p)

    vis = annotate(frame, last_res)
    dt = now - t_prev
    t_prev = now
    if dt > 0:
        fps = 0.8 * fps + 0.2 * (1 / dt) if fps else 1 / dt

    frame_slot.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)
    render_side(last_res, fps, latency)
    if now - last_rec >= 2.0:
        last_rec = now
        render_records_static()

if cap:
    cap.release()
