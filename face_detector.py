"""
face_detector.py
================
Production-grade ensemble face detection CLI tool.

Features
--------
- Ensemble of RetinaFace + YOLOv8-face with MTCNN fallback
- Multi-scale pyramid inference (0.5x, 1.0x, 2.0x)
- Ensemble voting (vote_count >= 2 to accept a face)
- Gaussian Soft-NMS post-processing
- Adaptive low-light handling with Laplacian validation
- Geometric filtering (min size, aspect ratio)
- Inputs: single image, image folder, video file, webcam
- Outputs: annotated image/video, face crops, JSON metadata
- CUDA auto-detection with CPU fallback
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Logging setup - reconfigured after --verbose is parsed
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("face_detector")

# Suppress noisy third-party warnings unless verbose
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCALES: List[float] = [0.5, 1.0, 2.0]
CONF_THRESHOLD: float = 0.6
LOW_LIGHT_THRESHOLD: float = 80.0
LOW_LIGHT_CONF: float = 0.4
LAPLACIAN_VAR_MIN: float = 30.0
IOU_CLUSTER_THRESHOLD: float = 0.5
MIN_VOTE_COUNT: int = 2
MIN_BOX_SIDE: int = 30
ASPECT_RATIO_MIN: float = 0.8
ASPECT_RATIO_MAX: float = 2.0
SOFT_NMS_SIGMA: float = 0.5
SOFT_NMS_SCORE_THRESH: float = 0.3

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ---------------------------------------------------------------------------
# Detection data structure
# ---------------------------------------------------------------------------

Detection = Dict[str, Any]
# Keys: box=[x1,y1,x2,y2], confidence=float, model=str, scale=float


# ===========================================================================
# Geometry helpers
# ===========================================================================

def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute Intersection over Union of two [x1,y1,x2,y2] boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    if inter == 0.0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def scale_box(box: List[float], scale: float) -> List[float]:
    """Scale a [x1,y1,x2,y2] box back to original coordinates."""
    return [v / scale for v in box]


def clip_box(box: List[float], img_w: int, img_h: int) -> List[float]:
    """Clip box to image boundaries."""
    x1 = max(0.0, min(box[0], img_w - 1))
    y1 = max(0.0, min(box[1], img_h - 1))
    x2 = max(0.0, min(box[2], img_w - 1))
    y2 = max(0.0, min(box[3], img_h - 1))
    return [x1, y1, x2, y2]


def box_area(box: List[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def average_boxes(boxes: List[List[float]]) -> List[float]:
    """Element-wise average of a list of [x1,y1,x2,y2] boxes."""
    arr = np.array(boxes, dtype=np.float32)
    return arr.mean(axis=0).tolist()


# ===========================================================================
# Soft-NMS
# ===========================================================================

def soft_nms(
    boxes: List[List[float]],
    scores: List[float],
    iou_threshold: float = IOU_CLUSTER_THRESHOLD,
    sigma: float = SOFT_NMS_SIGMA,
    score_threshold: float = SOFT_NMS_SCORE_THRESH,
) -> Tuple[List[List[float]], List[float]]:
    """
    Gaussian Soft-NMS.

    For each iteration, pick the highest-scoring box M, then decay all
    remaining boxes b_i using:
        s_i = s_i * exp( -(iou(M, b_i)^2) / sigma )

    Returns filtered (boxes, scores) where score > score_threshold.

    Parameters
    ----------
    boxes : list of [x1,y1,x2,y2]
    scores : list of float, same length as boxes
    iou_threshold : not used for decay but kept for API compat
    sigma : decay parameter (default 0.5)
    score_threshold : minimum score after decay to keep detection

    Returns
    -------
    kept_boxes, kept_scores
    """
    if not boxes:
        return [], []

    boxes_arr = np.array(boxes, dtype=np.float64)
    scores_arr = np.array(scores, dtype=np.float64)
    n = len(scores_arr)
    indices = list(range(n))
    result_boxes: List[List[float]] = []
    result_scores: List[float] = []

    while indices:
        # Pick highest score
        best_local = int(np.argmax(scores_arr[indices]))
        best_idx = indices[best_local]
        result_boxes.append(boxes_arr[best_idx].tolist())
        result_scores.append(float(scores_arr[best_idx]))
        indices.pop(best_local)

        # Decay remaining scores
        remaining: List[int] = []
        for idx in indices:
            iou_val = compute_iou(
                boxes_arr[best_idx].tolist(), boxes_arr[idx].tolist()
            )
            decay = math.exp(-(iou_val ** 2) / sigma)
            scores_arr[idx] *= decay
            if scores_arr[idx] > score_threshold:
                remaining.append(idx)
        indices = remaining

    return result_boxes, result_scores


# ===========================================================================
# Ensemble voting / clustering
# ===========================================================================

def cluster_detections(
    detections: List[Detection],
    iou_threshold: float = IOU_CLUSTER_THRESHOLD,
) -> List[List[Detection]]:
    """
    Greedy IoU-based clustering.

    Sort by confidence descending, then greedily group detections whose
    IoU with the cluster seed exceeds iou_threshold.

    Returns a list of clusters (each cluster is a list of Detection dicts).
    """
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    assigned = [False] * len(sorted_dets)
    clusters: List[List[Detection]] = []

    for i, det in enumerate(sorted_dets):
        if assigned[i]:
            continue
        cluster = [det]
        assigned[i] = True
        for j in range(i + 1, len(sorted_dets)):
            if assigned[j]:
                continue
            iou_val = compute_iou(det["box"], sorted_dets[j]["box"])
            if iou_val > iou_threshold:
                cluster.append(sorted_dets[j])
                assigned[j] = True
        clusters.append(cluster)

    return clusters


def vote_and_merge(
    clusters: List[List[Detection]],
    min_votes: int = MIN_VOTE_COUNT,
) -> List[Dict[str, Any]]:
    """
    Apply ensemble voting logic.

    For each cluster:
    - Count distinct (model, scale) pairs -> vote_count
    - Accept if vote_count >= min_votes
    - Final box = average of all boxes in cluster
    - Final confidence = max of confidences

    Returns list of accepted faces with keys:
        box, confidence, vote_count, voters (list of model names)
    """
    accepted: List[Dict[str, Any]] = []
    for cluster in clusters:
        voters_set = set()
        for d in cluster:
            voters_set.add((d["model"], d["scale"]))
        vote_count = len(voters_set)
        if vote_count < min_votes:
            continue
        avg_box = average_boxes([d["box"] for d in cluster])
        max_conf = max(d["confidence"] for d in cluster)
        voter_models = sorted({d["model"] for d in cluster})
        accepted.append(
            {
                "box": avg_box,
                "confidence": max_conf,
                "vote_count": vote_count,
                "voters": voter_models,
            }
        )
    return accepted


# ===========================================================================
# Geometric & Laplacian filtering
# ===========================================================================

def geometric_filter(
    faces: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
) -> List[Dict[str, Any]]:
    """
    Reject bounding boxes that violate size or aspect-ratio constraints.
    """
    kept = []
    for face in faces:
        x1, y1, x2, y2 = face["box"]
        w = x2 - x1
        h = y2 - y1
        if w < MIN_BOX_SIDE or h < MIN_BOX_SIDE:
            log.debug("Geometric filter: box too small (%.0fx%.0f)", w, h)
            continue
        ratio = w / h if h > 0 else 0
        if not (ASPECT_RATIO_MIN <= ratio <= ASPECT_RATIO_MAX):
            log.debug("Geometric filter: bad aspect ratio %.2f", ratio)
            continue
        kept.append(face)
    return kept


def laplacian_validate(
    image: np.ndarray,
    faces: List[Dict[str, Any]],
    var_threshold: float = LAPLACIAN_VAR_MIN,
) -> List[Dict[str, Any]]:
    """
    Reject detections in low-light frames where the ROI has almost no
    texture (Laplacian variance below var_threshold).
    """
    kept = []
    h, w = image.shape[:2]
    for face in faces:
        x1, y1, x2, y2 = [int(v) for v in face["box"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        lap_var = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
        if lap_var < var_threshold:
            log.debug("Laplacian filter: low texture %.2f", lap_var)
            continue
        kept.append(face)
    return kept


# ===========================================================================
# Model wrappers
# ===========================================================================

class BaseDetector:
    """Abstract base for all face detectors."""

    name: str = "Base"

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = CONF_THRESHOLD,
    ) -> List[Detection]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# RetinaFace wrapper (insightface)
# ---------------------------------------------------------------------------

class RetinaFaceDetector(BaseDetector):
    """
    Wraps insightface FaceAnalysis to use RetinaFace model.
    Accepts BGR images (cv2 native format).
    """

    name = "RetinaFace"

    def __init__(self, device: str = "cpu") -> None:
        import insightface  # type: ignore
        from insightface.app import FaceAnalysis  # type: ignore

        ctx_id = 0 if device == "cuda" else -1
        log.info("Loading RetinaFace (insightface) on %s ...", device)
        self._app = FaceAnalysis(
            name="buffalo_sc",
            allowed_modules=["detection"],
            providers=["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        log.info("RetinaFace loaded.")

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = CONF_THRESHOLD,
    ) -> List[Detection]:
        faces = self._app.get(image)
        results: List[Detection] = []
        for f in faces:
            score = float(f.det_score)
            if score < conf_threshold:
                continue
            x1, y1, x2, y2 = f.bbox.astype(float).tolist()
            results.append(
                {
                    "box": [x1, y1, x2, y2],
                    "confidence": score,
                    "model": self.name,
                    "scale": 1.0,
                }
            )
        return results


# ---------------------------------------------------------------------------
# YOLOv8-face wrapper (ultralytics)
# ---------------------------------------------------------------------------

class YOLOFaceDetector(BaseDetector):
    """
    Wraps ultralytics YOLO with a face-specific model weight file.
    Expects BGR input (converts to RGB internally via ultralytics).
    """

    name = "YOLOv8-face"

    def __init__(self, weights: str = "yolov8n-face.pt", device: str = "cpu") -> None:
        from ultralytics import YOLO  # type: ignore

        log.info("Loading YOLOv8-face from %s on %s ...", weights, device)
        self._model = YOLO(weights)
        self._device = device
        log.info("YOLOv8-face loaded.")

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = CONF_THRESHOLD,
    ) -> List[Detection]:
        import torch

        with torch.no_grad():
            results = self._model.predict(
                image,
                conf=conf_threshold,
                device=self._device,
                verbose=False,
            )
        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                score = float(box.conf[0])
                if score < conf_threshold:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    {
                        "box": [x1, y1, x2, y2],
                        "confidence": score,
                        "model": self.name,
                        "scale": 1.0,
                    }
                )
        return detections


# ---------------------------------------------------------------------------
# MTCNN wrapper (facenet-pytorch) - used as fallback
# ---------------------------------------------------------------------------

class MTCNNDetector(BaseDetector):
    """
    Wraps facenet-pytorch MTCNN detector. Used as fallback when primary
    models fail to load.
    """

    name = "MTCNN"

    def __init__(self, device: str = "cpu") -> None:
        from facenet_pytorch import MTCNN  # type: ignore
        import torch

        log.info("Loading MTCNN on %s ...", device)
        self._mtcnn = MTCNN(
            keep_all=True,
            device=torch.device(device),
            select_largest=False,
            min_face_size=20,
        )
        log.info("MTCNN loaded.")

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = CONF_THRESHOLD,
    ) -> List[Detection]:
        from PIL import Image as PILImage

        # MTCNN expects RGB PIL image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)
        boxes, probs = self._mtcnn.detect(pil_img)
        if boxes is None:
            return []
        results: List[Detection] = []
        for box, prob in zip(boxes, probs):
            if prob is None or prob < conf_threshold:
                continue
            x1, y1, x2, y2 = box.tolist()
            results.append(
                {
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(prob),
                    "model": self.name,
                    "scale": 1.0,
                }
            )
        return results


# ===========================================================================
# Ensemble Detector
# ===========================================================================

class EnsembleDetector:
    """
    Orchestrates multi-scale inference across all loaded models, performs
    ensemble voting, Soft-NMS, and geometric/low-light filtering.

    Parameters
    ----------
    device : "cuda" or "cpu"
    yolo_weights : path to YOLOv8-face weights file
    scales : list of scale factors for multi-scale pyramid
    """

    def __init__(
        self,
        device: str = "cpu",
        yolo_weights: str = "yolov8n-face.pt",
        scales: Optional[List[float]] = None,
    ) -> None:
        self._device = device
        self._scales = scales or SCALES
        self._models: List[BaseDetector] = []
        self._load_models(yolo_weights)

    # ------------------------------------------------------------------
    # Model loading with fallback logic
    # ------------------------------------------------------------------

    def _load_models(self, yolo_weights: str) -> None:
        primary_loaded = 0

        # --- RetinaFace ---
        try:
            self._models.append(RetinaFaceDetector(self._device))
            primary_loaded += 1
        except Exception as exc:
            log.warning("RetinaFace failed to load: %s", exc)

        # --- YOLOv8-face ---
        try:
            self._models.append(YOLOFaceDetector(yolo_weights, self._device))
            primary_loaded += 1
        except Exception as exc:
            log.warning("YOLOv8-face failed to load: %s", exc)

        # --- MTCNN fallback ---
        if primary_loaded < 2:
            log.warning(
                "Only %d primary model(s) loaded. Activating MTCNN fallback.", primary_loaded
            )
            try:
                self._models.append(MTCNNDetector(self._device))
            except Exception as exc:
                log.error("MTCNN fallback also failed: %s", exc)

        if not self._models:
            raise RuntimeError("No face detection models could be loaded. Aborting.")

        log.info(
            "Ensemble ready with %d model(s): %s",
            len(self._models),
            [m.name for m in self._models],
        )

    # ------------------------------------------------------------------
    # Luminance check
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_luminance(image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    # ------------------------------------------------------------------
    # Multi-scale inference for one model
    # ------------------------------------------------------------------

    def _run_model_multiscale(
        self,
        model: BaseDetector,
        image: np.ndarray,
        conf_threshold: float,
    ) -> List[Detection]:
        """
        Run a single model at each scale factor. Boxes are rescaled back
        to original image coordinates.
        """
        orig_h, orig_w = image.shape[:2]
        all_dets: List[Detection] = []

        for scale in self._scales:
            if scale == 1.0:
                scaled_img = image
            else:
                new_w = max(1, int(orig_w * scale))
                new_h = max(1, int(orig_h * scale))
                interp = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
                scaled_img = cv2.resize(image, (new_w, new_h), interpolation=interp)

            try:
                dets = model.detect(scaled_img, conf_threshold)
            except Exception as exc:
                log.warning("%s inference at scale %.1f failed: %s", model.name, scale, exc)
                continue

            for det in dets:
                # Scale bounding box back to original coordinates
                box = scale_box(det["box"], scale)
                box = clip_box(box, orig_w, orig_h)
                all_dets.append(
                    {
                        "box": box,
                        "confidence": det["confidence"],
                        "model": model.name,
                        "scale": scale,
                    }
                )

        return all_dets

    # ------------------------------------------------------------------
    # Main detection pipeline
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Full ensemble pipeline for a single BGR image.

        Returns list of accepted faces, each with:
          box, confidence, vote_count, voters
        """
        orig_h, orig_w = image.shape[:2]

        # Luminance-adaptive threshold
        lum = self._mean_luminance(image)
        is_dark = lum < LOW_LIGHT_THRESHOLD
        conf_threshold = LOW_LIGHT_CONF if is_dark else CONF_THRESHOLD
        if is_dark:
            log.debug("Low-light frame (luminance=%.1f); using conf_threshold=%.2f", lum, conf_threshold)

        # Collect detections from all models x all scales
        all_dets: List[Detection] = []
        for model in self._models:
            dets = self._run_model_multiscale(model, image, conf_threshold)
            all_dets.extend(dets)

        log.debug("Raw detections before voting: %d", len(all_dets))

        if not all_dets:
            return []

        # Cluster -> vote -> merge
        clusters = cluster_detections(all_dets, IOU_CLUSTER_THRESHOLD)
        voted = vote_and_merge(clusters, MIN_VOTE_COUNT)

        log.debug("After voting (%d votes required): %d face(s)", MIN_VOTE_COUNT, len(voted))

        if not voted:
            return []

        # Soft-NMS on voted faces
        boxes = [f["box"] for f in voted]
        scores = [f["confidence"] for f in voted]
        kept_boxes, kept_scores = soft_nms(
            boxes,
            scores,
            iou_threshold=IOU_CLUSTER_THRESHOLD,
            sigma=SOFT_NMS_SIGMA,
            score_threshold=SOFT_NMS_SCORE_THRESH,
        )

        # Rebuild result list preserving vote metadata
        # Match back by box index (soft-NMS preserves order of kept)
        box_to_face = {tuple(round(v, 2) for v in f["box"]): f for f in voted}
        final_faces: List[Dict[str, Any]] = []
        for kb, ks in zip(kept_boxes, kept_scores):
            key = tuple(round(v, 2) for v in kb)
            face_meta = box_to_face.get(key, {})
            final_faces.append(
                {
                    "box": kb,
                    "confidence": ks,
                    "vote_count": face_meta.get("vote_count", 1),
                    "voters": face_meta.get("voters", []),
                }
            )

        log.debug("After Soft-NMS: %d face(s)", len(final_faces))

        # Geometric filter
        final_faces = geometric_filter(final_faces, orig_w, orig_h)

        # Low-light: Laplacian validation to reject false positives
        if is_dark:
            final_faces = laplacian_validate(image, final_faces)

        log.debug("Final accepted faces: %d", len(final_faces))
        return final_faces


# ===========================================================================
# Visualization & I/O helpers
# ===========================================================================

def draw_detections(
    image: np.ndarray,
    faces: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Draw bounding boxes and labels on image.
    Returns a copy with annotations.
    """
    vis = image.copy()
    for face in faces:
        x1, y1, x2, y2 = [int(v) for v in face["box"]]
        conf = face.get("confidence", 0.0)
        votes = face.get("vote_count", 1)
        label = f"face: {conf:.2f} (votes={votes})"

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Background rectangle for text readability
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = max(y1 - 8, th + 4)
        cv2.rectangle(vis, (x1, ty - th - 4), (x1 + tw + 2, ty + 2), (0, 255, 0), -1)
        cv2.putText(
            vis,
            label,
            (x1 + 1, ty - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return vis


def save_crops(
    image: np.ndarray,
    faces: List[Dict[str, Any]],
    crops_dir: Path,
    prefix: str = "face",
) -> None:
    """Save each detected face ROI as a JPEG file."""
    crops_dir.mkdir(parents=True, exist_ok=True)
    for idx, face in enumerate(faces):
        x1, y1, x2, y2 = [int(v) for v in face["box"]]
        crop = image[max(0, y1): y2, max(0, x1): x2]
        if crop.size == 0:
            continue
        out_path = crops_dir / f"{prefix}_{idx}.jpg"
        cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])


def face_to_json_dict(face: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a face dict to a JSON-serialisable structure."""
    box = face["box"]
    return {
        "box": [round(v, 2) for v in box],
        "confidence": round(face.get("confidence", 0.0), 4),
        "ensemble_vote": face.get("voters", []),
    }


# ===========================================================================
# Input processing functions
# ===========================================================================

def process_image(
    detector: EnsembleDetector,
    input_path: Path,
    output_path: Optional[Path],
    crops_dir: Optional[Path],
) -> List[Dict[str, Any]]:
    """
    Process a single image file.

    Returns the list of detected faces (JSON-serialisable dicts).
    """
    log.info("Processing image: %s", input_path)
    image = cv2.imread(str(input_path))
    if image is None:
        log.error("Could not read image: %s", input_path)
        return []

    faces = detector.detect(image)
    log.info("  -> %d face(s) detected", len(faces))

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        annotated = draw_detections(image, faces)
        cv2.imwrite(str(output_path), annotated)
        log.info("  -> Saved annotated image: %s", output_path)

    if crops_dir is not None:
        prefix = input_path.stem
        save_crops(image, faces, crops_dir, prefix=prefix)

    return [face_to_json_dict(f) for f in faces]


def process_folder(
    detector: EnsembleDetector,
    input_dir: Path,
    output_dir: Optional[Path],
    crops_dir: Optional[Path],
) -> List[Dict[str, Any]]:
    """
    Process all images in a directory.

    Returns list of per-image result dicts.
    """
    image_files = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    if not image_files:
        log.warning("No image files found in %s", input_dir)
        return []

    all_results: List[Dict[str, Any]] = []
    for img_path in image_files:
        out_path = None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / img_path.name

        faces_json = process_image(detector, img_path, out_path, crops_dir)
        all_results.append(
            {
                "file": str(img_path),
                "num_faces": len(faces_json),
                "faces": faces_json,
            }
        )

    return all_results


def process_video(
    detector: EnsembleDetector,
    source: str,
    output_path: Optional[Path],
    crops_dir: Optional[Path],
    no_display: bool = False,
) -> List[Dict[str, Any]]:
    """
    Process a video file or webcam stream frame by frame.

    Returns list of per-frame detection dicts.
    """
    # Determine source
    try:
        cam_index = int(source)
        cap = cv2.VideoCapture(cam_index)
        log.info("Opening webcam index %d", cam_index)
    except ValueError:
        cap = cv2.VideoCapture(source)
        log.info("Opening video file: %s", source)

    if not cap.isOpened():
        log.error("Could not open video source: %s", source)
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer: Optional[cv2.VideoWriter] = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_w, frame_h))
        log.info("Video output: %s", output_path)

    frame_results: List[Dict[str, Any]] = []
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                faces = detector.detect(frame)
            except Exception as exc:
                log.warning("Frame %d detection error: %s", frame_idx, exc)
                faces = []

            faces_json = [face_to_json_dict(f) for f in faces]
            frame_results.append(
                {"frame": frame_idx, "num_faces": len(faces_json), "faces": faces_json}
            )
            log.debug("Frame %d: %d face(s)", frame_idx, len(faces))

            annotated = draw_detections(frame, faces)

            if writer is not None:
                writer.write(annotated)

            if not no_display:
                cv2.imshow("Face Detector - press Q to quit", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log.info("User quit.")
                    break

            if crops_dir is not None and faces:
                save_crops(
                    frame,
                    faces,
                    crops_dir,
                    prefix=f"frame{frame_idx:06d}",
                )

            frame_idx += 1

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not no_display:
            cv2.destroyAllWindows()

    log.info("Processed %d frame(s).", frame_idx)
    return frame_results


# ===========================================================================
# WIDER Face Evaluation
# ===========================================================================

def evaluate_wider(
    detector: EnsembleDetector,
    wider_root: Path,
    max_samples: int = 500,
) -> None:
    """
    Evaluate the detector on the WIDER Face validation set and report
    mean Average Precision (mAP) at IoU=0.5.

    Expected directory structure:
        wider_root/
            WIDER_val/images/...
            wider_face_split/wider_face_val_bbx_gt.txt

    Parameters
    ----------
    detector : EnsembleDetector
    wider_root : Path to WIDER Face root directory
    max_samples : maximum number of images to evaluate (for speed)
    """
    gt_file = wider_root / "wider_face_split" / "wider_face_val_bbx_gt.txt"
    images_root = wider_root / "WIDER_val" / "images"

    if not gt_file.exists():
        log.error("WIDER Face GT file not found: %s", gt_file)
        return
    if not images_root.exists():
        log.error("WIDER Face images not found: %s", images_root)
        return

    # Parse ground truth file
    gt_data: List[Tuple[str, List[List[float]]]] = []
    with open(gt_file, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    i = 0
    while i < len(lines):
        rel_path = lines[i]
        i += 1
        if i >= len(lines):
            break
        try:
            n_faces = int(lines[i])
        except ValueError:
            i += 1
            continue
        i += 1
        boxes = []
        if n_faces == 0:
            i += 1  # skip dummy "0 0 0 0 0 0 0 0 0 0" line
        else:
            for _ in range(n_faces):
                parts = list(map(float, lines[i].split()[:4]))
                x, y, bw, bh = parts
                boxes.append([x, y, x + bw, y + bh])
                i += 1
        gt_data.append((rel_path, boxes))
        if len(gt_data) >= max_samples:
            break

    log.info("Evaluating on %d WIDER Face images ...", len(gt_data))

    all_scores: List[float] = []
    all_tp: List[int] = []
    all_fp: List[int] = []
    total_gt = 0

    for rel_path, gt_boxes in gt_data:
        img_path = images_root / rel_path
        if not img_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        faces = detector.detect(image)
        pred_boxes = [f["box"] for f in faces]
        pred_scores = [f["confidence"] for f in faces]

        total_gt += len(gt_boxes)

        if not gt_boxes:
            all_fp.extend([1] * len(pred_boxes))
            all_tp.extend([0] * len(pred_boxes))
            all_scores.extend(pred_scores)
            continue

        matched_gt = set()
        pairs = sorted(
            [
                (score, pidx)
                for pidx, score in enumerate(pred_scores)
            ],
            reverse=True,
        )

        for score, pidx in pairs:
            best_iou = 0.0
            best_gidx = -1
            for gidx, gb in enumerate(gt_boxes):
                if gidx in matched_gt:
                    continue
                iou_val = compute_iou(pred_boxes[pidx], gb)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gidx = gidx

            if best_iou >= 0.5 and best_gidx >= 0:
                all_tp.append(1)
                all_fp.append(0)
                matched_gt.add(best_gidx)
            else:
                all_tp.append(0)
                all_fp.append(1)
            all_scores.append(score)

    if total_gt == 0:
        log.warning("No GT faces found. Check your WIDER Face path.")
        return

    # Sort by score descending
    order = sorted(range(len(all_scores)), key=lambda k: all_scores[k], reverse=True)
    tp_sorted = [all_tp[k] for k in order]
    fp_sorted = [all_fp[k] for k in order]

    tp_cum = np.cumsum(tp_sorted)
    fp_cum = np.cumsum(fp_sorted)

    recall = tp_cum / total_gt
    precision = tp_cum / (tp_cum + fp_cum + 1e-9)

    # Compute AP using 11-point interpolation
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        prec_at_thr = precision[recall >= thr]
        if len(prec_at_thr) > 0:
            ap += prec_at_thr.max()
    ap /= 11.0

    max_recall = recall[-1] if len(recall) > 0 else 0.0
    print("\n" + "=" * 50)
    print(f"WIDER Face Evaluation ({len(gt_data)} images)")
    print(f"  Total GT faces : {total_gt}")
    print(f"  AP @ IoU=0.5   : {ap:.4f}  ({ap*100:.2f}%)")
    print(f"  Max Recall     : {max_recall:.4f}  ({max_recall*100:.2f}%)")
    print("=" * 50 + "\n")


# ===========================================================================
# Device detection
# ===========================================================================

def get_device() -> str:
    """Return 'cuda' if a GPU is available and has enough VRAM, else 'cpu'."""
    try:
        import torch

        if torch.cuda.is_available():
            # Basic VRAM sanity check: require at least 1 GB free
            free, _ = torch.cuda.mem_get_info()
            if free > 1 * 1024 ** 3:
                log.info("CUDA device: %s", torch.cuda.get_device_name(0))
                return "cuda"
            else:
                log.warning("Insufficient GPU VRAM (%.0f MB free). Using CPU.", free / 1e6)
        else:
            log.info("CUDA not available. Using CPU.")
    except Exception as exc:
        log.warning("Device detection failed: %s. Using CPU.", exc)
    return "cpu"


# ===========================================================================
# CLI
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="face_detector",
        description=(
            "Production-grade ensemble face detector.\n"
            "Supports images, image folders, video files, and webcam streams.\n"
            "Models: RetinaFace + YOLOv8-face (MTCNN fallback)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input", "-i",
        metavar="INPUT",
        help=(
            "Path to: a single image, a folder of images, a video file, "
            "or a webcam index (e.g. 0)."
        ),
    )
    p.add_argument(
        "--output", "-o",
        metavar="OUTPUT",
        default=None,
        help="Path to save annotated output (image, video, or folder).",
    )
    p.add_argument(
        "--save-crops",
        metavar="DIR",
        default=None,
        help="Directory to save cropped face images.",
    )
    p.add_argument(
        "--json",
        metavar="FILE",
        default=None,
        help="Path to export detection metadata as JSON.",
    )
    p.add_argument(
        "--no-display",
        action="store_true",
        help="Suppress live preview window (useful for headless servers).",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    p.add_argument(
        "--yolo-weights",
        metavar="PATH",
        default="yolov8n-face.pt",
        help="Path to YOLOv8-face weights (default: yolov8n-face.pt).",
    )
    p.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation on WIDER Face dataset (requires --wider-root).",
    )
    p.add_argument(
        "--wider-root",
        metavar="DIR",
        default="./wider_face",
        help="Root directory of WIDER Face dataset (used with --evaluate).",
    )
    p.add_argument(
        "--eval-samples",
        type=int,
        default=500,
        metavar="N",
        help="Max number of images to evaluate (default: 500).",
    )
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        log.setLevel(logging.DEBUG)

    # Detect device
    device = get_device()

    # Load ensemble
    try:
        detector = EnsembleDetector(
            device=device,
            yolo_weights=args.yolo_weights,
        )
    except RuntimeError as exc:
        log.error("Fatal: %s", exc)
        return 1

    # Evaluation mode
    if args.evaluate:
        evaluate_wider(detector, Path(args.wider_root), args.eval_samples)
        return 0

    if args.input is None:
        parser.print_help()
        return 0

    input_src = args.input
    output_path = Path(args.output) if args.output else None
    crops_dir = Path(args.save_crops) if args.save_crops else None
    json_path = Path(args.json) if args.json else None

    json_results: Any = None

    # Route to correct processing mode
    # Try to detect if input is a webcam index
    is_webcam = False
    try:
        int(input_src)
        is_webcam = True
    except (ValueError, TypeError):
        pass

    if is_webcam:
        frame_results = process_video(
            detector, input_src, output_path, crops_dir, args.no_display
        )
        json_results = {"input_path": input_src, "frames": frame_results}

    else:
        input_path = Path(input_src)
        if not input_path.exists():
            log.error("Input not found: %s", input_path)
            return 1

        if input_path.is_dir():
            results = process_folder(detector, input_path, output_path, crops_dir)
            total = sum(r["num_faces"] for r in results)
            log.info("Folder complete. Total faces detected: %d", total)
            json_results = {
                "input_path": str(input_path),
                "total_faces": total,
                "images": results,
            }

        elif input_path.suffix.lower() in IMAGE_EXTS:
            faces_json = process_image(detector, input_path, output_path, crops_dir)
            json_results = {
                "input_path": str(input_path),
                "num_faces": len(faces_json),
                "faces": faces_json,
            }
            print(json.dumps(json_results, indent=2))

        else:
            # Assume video
            frame_results = process_video(
                detector, str(input_path), output_path, crops_dir, args.no_display
            )
            json_results = {
                "input_path": str(input_path),
                "num_frames": len(frame_results),
                "frames": frame_results,
            }

    # Write JSON if requested
    if json_path is not None and json_results is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as fh:
            json.dump(json_results, fh, indent=2)
        log.info("JSON results saved: %s", json_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
