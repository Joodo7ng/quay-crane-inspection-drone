"""Two independent U-Net models; raw predictions never imply structural safety."""

from pathlib import Path
import hashlib
import time
import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
TYPES = ("균열", "부식")


def image_fingerprint(image):
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or not image.size
        or image.dtype != np.uint8
    ):
        raise ValueError("유효한 uint8 BGR 영상이 필요함")
    return hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()


class _Infer:
    def __init__(
        self,
        path,
        imgsz=512,
        encoder="efficientnet-b0",
        model_id=None,
        expected_sha256=None,
        inference_profile="legacy",
    ):
        self.inference_profile = inference_profile
        self.native_mask = None
        self.imgsz = imgsz
        self.onnx = str(path).endswith(".onnx")
        raw = Path(path).read_bytes()
        actual_sha = hashlib.sha256(raw).hexdigest()
        if expected_sha256 and actual_sha != expected_sha256:
            raise ValueError(f"SHA256 불일치: {model_id}; 실제 {actual_sha}")
        self.identity = {
            "model_id": model_id or "CUSTOM",
            "expected_sha256": expected_sha256,
            "hash_verified": bool(expected_sha256),
            "inference_profile": inference_profile,
            "path": str(Path(path).resolve()),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "architecture": "Unet/efficientnet-b0",
            "input_size": imgsz,
            "device": "cpu",
        }
        if self.onnx:
            import onnxruntime as ort

            self.sess = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
            self.iname = self.sess.get_inputs()[0].name
        else:
            import torch
            import segmentation_models_pytorch as smp

            self.torch = torch
            self.m = smp.Unet(encoder, encoder_weights=None, in_channels=3, classes=1)
            state = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(state, dict):
                for key in ("state_dict", "model_state_dict"):
                    if key in state:
                        state = state[key]
                        break
            if (
                not isinstance(state, dict)
                or not state
                or not all(isinstance(v, torch.Tensor) for v in state.values())
            ):
                raise ValueError("유효한 tensor state_dict가 아님")
            if all(k.startswith("module.") for k in state):
                state = {k[7:]: v for k, v in state.items()}
            if not all(torch.isfinite(v).all() for v in state.values()):
                raise ValueError("가중치에 유효하지 않은 값이 있음")
            self.m.load_state_dict(state, strict=True)
            self.m.eval()

    def mask(self, rgb, thr=0.5):
        h, w = rgb.shape[:2]
        if self.inference_profile == "paper":
            from PIL import Image

            resized = np.asarray(
                Image.fromarray(rgb).resize(
                    (self.imgsz, self.imgsz), Image.Resampling.BILINEAR
                )
            )
        else:
            resized = cv2.resize(rgb, (self.imgsz, self.imgsz))
        x = resized.astype(np.float32) / 255.0
        x = ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
        if self.onnx:
            logit = self.sess.run(None, {self.iname: x})[0]
            prob = 1 / (1 + np.exp(-np.clip(logit, -80, 80)))[0, 0]
        else:
            with self.torch.inference_mode():
                prob = self.torch.sigmoid(self.m(self.torch.from_numpy(x)))[
                    0, 0
                ].numpy()
        if not np.isfinite(prob).all():
            raise ValueError("모델 출력에 유효하지 않은 수치가 있음")
        self.native_mask = (prob > thr).astype(np.uint8)
        if self.inference_profile == "paper":
            from PIL import Image

            return np.asarray(
                Image.fromarray(self.native_mask).resize(
                    (w, h), Image.Resampling.NEAREST
                )
            )
        return (cv2.resize(prob, (w, h)) > thr).astype(np.uint8)


class ModelSet:
    """Per-session model ownership; partial load failures retain the other model."""

    def __init__(
        self,
        crack_path=None,
        corrosion_path=None,
        imgsz=512,
        model_ids=None,
        expected_hashes=None,
        inference_profile="legacy",
    ):
        if inference_profile not in ("legacy", "paper"):
            raise ValueError("지원하지 않는 추론 모드임")
        self.inference_profile = inference_profile
        model_ids = model_ids or [None, None]
        expected_hashes = expected_hashes or [None, None]
        self.models = {}
        self.statuses = {}
        for kind, path, mid, sha in zip(
            TYPES, (crack_path, corrosion_path), model_ids, expected_hashes
        ):
            if not path:
                self.statuses[kind] = {
                    "state": "not_loaded",
                    "error": "모델 미지정",
                    "model": {"model_id": mid, "expected_sha256": sha},
                }
                continue
            try:
                model = _Infer(
                    path,
                    imgsz,
                    model_id=mid,
                    expected_sha256=sha,
                    inference_profile=inference_profile,
                )
                self.models[kind] = model
                self.statuses[kind] = {
                    "state": "ready",
                    "error": None,
                    "model": model.identity,
                }
            except Exception as exc:
                self.statuses[kind] = {
                    "state": "load_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "model": {
                        "path": str(path),
                        "model_id": mid,
                        "expected_sha256": sha,
                    },
                }

    def detect(self, image_bgr, thr=0.5, min_area=1, return_masks=True):
        if (
            not (0 < thr < 1)
            or isinstance(min_area, bool)
            or int(min_area) != min_area
            or min_area < 1
        ):
            raise ValueError("이진화 임계값은 0과 1 사이, 최소 영역은 양의 정수여야 함")
        if self.inference_profile == "paper" and (thr != 0.5 or min_area != 1):
            raise ValueError("논문 비교는 임계값 0.5 및 영역 제거 없음으로 고정함")
        from copy import deepcopy

        out = {
            "defects": [],
            "masks": {},
            "raw_masks": {},
            "evaluation_masks": {},
            "model_status": deepcopy(self.statuses),
            "overall_grade": None,
            "settings": {
                "inference_profile": self.inference_profile,
                "preprocessing": {
                    "color": "RGB",
                    "resize": (
                        "PIL bilinear"
                        if self.inference_profile == "paper"
                        else "OpenCV linear"
                    ),
                    "scale": "0..1",
                    "mean": MEAN.tolist(),
                    "std": STD.tolist(),
                },
                "binarization": "sigmoid > threshold",
                "evaluation_grid": (
                    "model input grid; evaluation_masks"
                    if self.inference_profile == "paper"
                    else "not paper-equivalent"
                ),
                "display_projection": (
                    "binary nearest to original"
                    if self.inference_profile == "paper"
                    else "probability linear to original then threshold"
                ),
                "postprocessing": (
                    "none" if min_area == 1 else "connected component area filter"
                ),
                "threshold": thr,
                "min_area_px": int(min_area),
                "confidence_definition": "미제공: 보정된 후보 신뢰도를 산출하지 않음",
            },
            "image_shape": None,
        }
        try:
            out["frame_sha256"] = image_fingerprint(image_bgr)
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            out["image_shape"] = [h, w]
        except Exception as exc:
            out["input_error"] = str(exc)
            for status in out["model_status"].values():
                if status["state"] == "ready":
                    status.update(state="failed", error="입력 영상 읽기 실패")
            return out
        for kind in TYPES:
            if kind not in self.models:
                continue
            t0 = time.perf_counter()
            try:
                mask = np.asarray(self.models[kind].mask(rgb, thr))
                if mask.shape != (h, w) or not np.isin(mask, [0, 1]).all():
                    raise ValueError("모델 마스크의 크기 또는 값이 잘못됨")
                mask = mask.astype(np.uint8)
                n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
                filtered = np.zeros_like(mask)
                candidates = []
                for i in range(1, n):
                    x, y, bw, bh, area = (int(v) for v in stats[i])
                    if area < min_area:
                        continue
                    filtered[labels == i] = 1
                    candidates.append(
                        {
                            "defect": kind,
                            "bbox": [x, y, bw, bh],
                            "area_px": area,
                            "area_pct": area / (h * w) * 100,
                            "grade": None,
                            "confidence": None,
                            "component_label": i,
                        }
                    )
                out["defects"].extend(candidates)
                if return_masks:
                    if getattr(self.models[kind], "native_mask", None) is not None:
                        out["evaluation_masks"][kind] = self.models[
                            kind
                        ].native_mask.copy()
                    out["raw_masks"][kind] = mask
                    out["masks"][kind] = filtered
                out["model_status"][kind].update(state="success", error=None)
            except Exception as exc:
                out["model_status"][kind].update(
                    state="failed", error=f"{type(exc).__name__}: {exc}"
                )
            out["model_status"][kind]["inference_ms"] = (
                time.perf_counter() - t0
            ) * 1000
        # Compatibility keys use exactly the shared judgment implementation.
        from inspection_pipeline import assess

        assessed = assess(out)
        for raw_defect, assessed_defect in zip(out["defects"], assessed["defects"]):
            raw_defect["grade"] = assessed_defect["assessment"]["priority"]
        out["overall_grade"] = assessed["overall_priority"]
        return out


_default = None


def load_models(crack_path=None, corrosion_path=None, imgsz=512):
    global _default
    _default = ModelSet(crack_path, corrosion_path, imgsz)
    return _default


def detect(image_bgr, thr=0.5, min_area=1, return_masks=True):
    return (_default or ModelSet()).detect(image_bgr, thr, min_area, return_masks)


def draw(image_bgr, result):
    img = image_bgr.copy()
    colors = {"균열": (40, 55, 220), "부식": (0, 150, 255)}
    for kind, mask in result.get("masks", {}).items():
        if mask.shape != img.shape[:2]:
            continue
        active = mask == 1
        img[active] = (0.45 * np.array(colors[kind]) + 0.55 * img[active]).astype(
            np.uint8
        )
    for index, d in enumerate(result.get("defects", []), 1):
        x, y, w, h = d["bbox"]
        color = colors[d["defect"]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        label = "C" if d["defect"] == "균열" else "R"
        cv2.putText(
            img,
            f"{label}{index}",
            (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
    return img


def decode_image(data, profile="legacy"):
    if profile == "paper":
        from PIL import Image
        import io

        try:
            with Image.open(io.BytesIO(data)) as im:
                return cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
