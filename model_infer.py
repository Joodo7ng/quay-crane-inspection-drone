"""
model_infer.py — 대시보드에 붙이는 '모델쪽 코드'
=================================================
이미지를 넣으면 균열·부식 탐지 결과를 dict(JSON)로 돌려준다.
대시보드(수빈님)는 이 detect() 결과를 받아 화면에 표시하면 됨.

사용법:
    import cv2
    from model_infer import load_models, detect
    load_models("crack_efb0.onnx", "corrosion_efb0.onnx")   # 모델 파일 경로 (.pt도 됨)
    result = detect(cv2.imread("frame.png"))
    # result 예:
    # {
    #   "overall_grade": "위험",
    #   "defects": [
    #       {"defect":"부식","bbox":[x,y,w,h],"area_pct":12.3,"grade":"위험","confidence":0.9},
    #       {"defect":"균열","bbox":[x,y,w,h],"area_pct":0.8,"grade":"주의","confidence":0.9}
    #   ],
    #   "masks": {"균열": <0/1 numpy배열>, "부식": <0/1 numpy배열>}   # 화면 오버레이용 (선택)
    # }

준비: pip install onnxruntime opencv-python numpy   (.pt면 torch, segmentation-models-pytorch)
"""
import numpy as np
import cv2

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
TH = {"균열": {"warn": 0.3, "danger": 1.5}, "부식": {"warn": 3.0, "danger": 12.0}}
def _grade(d, a): t = TH[d]; return "위험" if a >= t["danger"] else "주의" if a >= t["warn"] else "정상"


class _Infer:
    def __init__(self, path, imgsz=512, encoder="efficientnet-b0"):
        self.imgsz = imgsz; self.onnx = path.endswith(".onnx")
        if self.onnx:
            import onnxruntime as ort
            self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self.iname = self.sess.get_inputs()[0].name
        else:
            import torch, segmentation_models_pytorch as smp
            self.torch = torch
            self.m = smp.Unet(encoder, encoder_weights=None, in_channels=3, classes=1)
            self.m.load_state_dict(torch.load(path, map_location="cpu")); self.m.eval()

    def mask(self, rgb, thr=0.5):
        H, W = rgb.shape[:2]
        x = cv2.resize(rgb, (self.imgsz, self.imgsz)).astype(np.float32) / 255.
        x = ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
        if self.onnx:
            logit = self.sess.run(None, {self.iname: x})[0]; prob = 1 / (1 + np.exp(-logit))[0, 0]
        else:
            with self.torch.no_grad():
                prob = self.torch.sigmoid(self.m(self.torch.from_numpy(x)))[0, 0].numpy()
        return (cv2.resize(prob, (W, H)) > thr).astype(np.uint8)


_crack = None
_corr = None


def load_models(crack_path=None, corrosion_path=None, imgsz=512):
    """모델 파일 로드 (앱 시작할 때 한 번). 균열/부식 중 하나만 줘도 됨."""
    global _crack, _corr
    _crack = _Infer(crack_path, imgsz) if crack_path else None
    _corr = _Infer(corrosion_path, imgsz) if corrosion_path else None
    if not (_crack or _corr):
        raise RuntimeError("모델 경로를 최소 하나 지정하세요.")


def detect(image_bgr, thr=0.5, min_area=100, return_masks=True):
    """이미지(BGR, cv2.imread 결과) → 탐지 결과 dict."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    out = {"defects": [], "masks": {}, "overall_grade": "정상"}
    for inf, dtype in [(_corr, "부식"), (_crack, "균열")]:
        if inf is None:
            continue
        m = inf.mask(rgb, thr)
        if return_masks:
            out["masks"][dtype] = m
        n, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < min_area:
                continue
            a = area / (H * W) * 100
            out["defects"].append({
                "defect": dtype, "bbox": [int(x), int(y), int(w), int(h)],
                "area_pct": round(float(a), 2), "grade": _grade(dtype, a), "confidence": 0.9,
            })
    grades = [d["grade"] for d in out["defects"]]
    out["overall_grade"] = "위험" if "위험" in grades else "주의" if "주의" in grades else "정상"
    return out


# 오버레이 이미지가 필요하면 이 헬퍼도 사용 가능 (선택)
def draw(image_bgr, result):
    img = image_bgr.copy()
    color = {"균열": (0, 0, 255), "부식": (0, 140, 255)}
    for dt, m in result.get("masks", {}).items():
        img[m == 1] = (0.45 * np.array(color[dt]) + 0.55 * img[m == 1]).astype(np.uint8)
    for d in result["defects"]:
        x, y, w, h = d["bbox"]
        cv2.rectangle(img, (x, y), (x + w, y + h), color[d["defect"]], 2)
        cv2.putText(img, f"{d['defect']} {d['area_pct']}% {d['grade']}", (x, max(14, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color[d["defect"]], 2)
    return img
