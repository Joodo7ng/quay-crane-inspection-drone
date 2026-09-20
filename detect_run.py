#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect_run.py — 팀원(정혜민)의 균열 세그멘테이션 모델을 대시보드에 붙이는 '어댑터'.

팀원 코드(model_infer.py)는 하나도 안 바꾼다. 그 detect() 결과를 받아
대시보드가 읽는 detections.csv 로 옮기고, 오버레이 이미지를 caps/ 에 저장한다.

    사진/프레임 폴더 ─▶ [model_infer.detect] ─▶ detections.csv + caps/ ─▶ dashboard.py

사용법:
    python3 detect_run.py demo_photos
    python3 detect_run.py 프레임.png --corrosion best_corrosion_efficientnetb0.pt
    python3 detect_run.py demo_photos --reset      # csv 초기화하고 새로

옵션:
    --crack       균열 모델 경로 (기본 crack_seg.pt)
    --corrosion   부식 모델 경로 (있을 때만)
    --reset       기존 detections.csv 삭제 후 시작
"""
import argparse
import csv
import os
import uuid
from datetime import datetime

import cv2
import model_infer   # 팀원 코드 (수정 안 함)

# 대시보드와 약속한 컬럼. 팀원 detect() 출력(종류·박스·면적·위험등급)을 그대로 담는다.
COLS = ["id", "timestamp", "source", "type", "grade", "area_pct",
        "confidence", "bbox", "image_path", "overall_grade", "status"]

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def collect(target):
    if os.path.isfile(target):
        return [target] if target.lower().endswith(IMG_EXTS) else []
    if os.path.isdir(target):
        out = []
        for root, _, files in os.walk(target):
            for f in sorted(files):
                if f.lower().endswith(IMG_EXTS):
                    out.append(os.path.join(root, f))
        return out
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", default="demo_photos", help="사진 파일 또는 폴더")
    ap.add_argument("--crack", default="crack_seg.pt", help="균열 모델 경로")
    ap.add_argument("--corrosion", default=None, help="부식 모델 경로(선택)")
    ap.add_argument("--csv", default="detections.csv")
    ap.add_argument("--caps", default="caps")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    imgs = collect(args.target)
    if not imgs:
        print(f"[에러] '{args.target}' 에서 이미지를 못 찾음")
        return

    if args.reset and os.path.exists(args.csv):
        os.remove(args.csv)
        print(f"[리셋] {args.csv} 삭제")
    os.makedirs(args.caps, exist_ok=True)

    # 팀원 모델 로드 (앱/스크립트당 한 번)
    print(f"[모델] 균열={args.crack}" + (f", 부식={args.corrosion}" if args.corrosion else ""))
    model_infer.load_models(args.crack, args.corrosion)

    new = not os.path.exists(args.csv)
    f = open(args.csv, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=COLS)
    if new:
        w.writeheader()

    n_img, n_row = 0, 0
    for path in imgs:
        img = cv2.imread(path)
        if img is None:
            continue
        n_img += 1
        res = model_infer.detect(img)                 # 팀원 함수 호출
        overlay = model_infer.draw(img, res)          # 팀원 오버레이
        src = os.path.basename(path)
        cap_name = f"{os.path.splitext(src)[0]}_det.png"
        cap_path = os.path.join(args.caps, cap_name)
        cv2.imwrite(cap_path, overlay)
        rel = cap_path.replace("\\", "/")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        overall = res.get("overall_grade", "정상")
        defects = res.get("defects", [])

        if defects:
            for d in defects:
                x, y, bw, bh = d["bbox"]
                w.writerow({
                    "id": uuid.uuid4().hex[:8],
                    "timestamp": now,
                    "source": src,
                    "type": d["defect"],                 # 균열 / 부식
                    "grade": d["grade"],                 # 정상 / 주의 / 위험
                    "area_pct": d["area_pct"],
                    "confidence": d["confidence"],
                    "bbox": f"{x},{y},{bw},{bh}",
                    "image_path": rel,
                    "overall_grade": overall,
                    "status": "pending",
                })
                n_row += 1
        else:
            # 결함 없음 = 점검 완료(정상). 그래도 한 줄 남겨 '점검됨'을 표시.
            w.writerow({
                "id": uuid.uuid4().hex[:8], "timestamp": now, "source": src,
                "type": "점검", "grade": "정상", "area_pct": 0.0, "confidence": "",
                "bbox": "", "image_path": rel, "overall_grade": "정상", "status": "pending",
            })
            n_row += 1
        f.flush()
        print(f"  {overall:4s}  {src}  (결함 {len(defects)}건) -> {rel}")

    f.close()
    print(f"\n[완료] 이미지 {n_img}장, {n_row}건 기록 -> {args.csv}")


if __name__ == "__main__":
    main()
