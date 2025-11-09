#!/usr/bin/env python3
"""
high_chin_detector.py
Production-ready module to compute a normalized "High Chin Score" (0-1) and label.
Supports:
 - Mediapipe (preferred)
 - mediapipe-silicon (if official mediapipe isn't available)
 - face_recognition (fallback)
Usage examples:
  python high_chin_detector.py --image ./face.jpg
  python high_chin_detector.py --webcam
  python high_chin_detector.py --folder ./images --out results.csv
"""

from __future__ import annotations
import argparse
import logging
import math
import sys
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, List
import os
import csv
import time

# Optional heavy imports (import inside functions to allow graceful fallback)
try:
    import numpy as np
except Exception:
    raise SystemExit("Please install numpy: pip install numpy")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("high_chin_detector")


@dataclass
class ChinResult:
    score: float                # 0..1 (higher = higher chin)
    label: str                  # "High Chin" / "Balanced Chin" / "Low Chin"
    chin_ratio: float           # raw ratio used (lower-third / face_height)
    face_height: float
    chin_segment: float
    landmarks_used: Dict[str, Tuple[float, float]]  # normalized coords


# ----- Utilities -----
def euclid(p1: Tuple[float, float], p2: Tuple[float, float], img_wh: Tuple[int, int]) -> float:
    w, h = img_wh
    dx = (p1[0] - p2[0]) * w
    dy = (p1[1] - p2[1]) * h
    return math.hypot(dx, dy)


# ----- Backends: Mediapipe first, then mediapipe-silicon, then face_recognition fallback -----
class LandmarkBackend:
    def get_landmarks(self, image: "np.ndarray") -> Optional[Dict[str, Tuple[float, float]]]:
        """Return dict of normalized landmarks (x,y in 0..1) for keys:
           'forehead', 'nose_tip', 'chin', 'left_eye_top', 'right_eye_top', 'bbox_top' (optional)
        """
        raise NotImplementedError()


class MediapipeBackend(LandmarkBackend):
    def __init__(self, module_name="mediapipe"):
        self.module_name = module_name
        try:
            self.mp = __import__(module_name)
            self.mp_face_mesh = self.mp.solutions.face_mesh
            self.mp_drawing = self.mp.solutions.drawing_utils
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            logger.info(f"Using {module_name} backend")
        except Exception as e:
            raise RuntimeError(f"Failed to init mediapipe backend ({module_name}): {e}")

    def get_landmarks(self, image):
        # image is BGR numpy
        import cv2
        img_h, img_w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        lm = results.multi_face_landmarks[0].landmark
        # Mediapipe indices (FaceMesh 468):
        def idx_to_xy(idx):
            return (lm[idx].x, lm[idx].y)
        # chosen landmarks
        data = {
            "forehead": idx_to_xy(10),      # approximate top forehead
            "nose_tip": idx_to_xy(1),
            "chin": idx_to_xy(152),
            "left_eye_top": idx_to_xy(386),  # left eye top
            "right_eye_top": idx_to_xy(159),
            "left_cheek": idx_to_xy(454),
            "right_cheek": idx_to_xy(234),
        }
        return data


class FaceRecBackend(LandmarkBackend):
    def __init__(self):
        try:
            import face_recognition
            self.fr = face_recognition
            logger.info("Using face_recognition backend (fallback)")
        except Exception as e:
            raise RuntimeError(f"Failed to init face_recognition: {e}")

    def get_landmarks(self, image):
        # image is BGR numpy
        rgb = image[:, :, ::-1]
        boxes = self.fr.face_locations(rgb, model="cnn" if self._has_cnn() else "hog")
        if not boxes:
            return None
        # take first face
        top, right, bottom, left = boxes[0]
        h, w = image.shape[:2]
        landmarks = self.fr.face_landmarks(rgb, boxes=[boxes[0]])[0]
        # face_recognition provides 'chin' list and 'nose_tip' approximated from 'nose_tip' list
        chin_points = landmarks.get("chin", [])
        nose_tip_pts = landmarks.get("nose_tip", [])
        left_eye = landmarks.get("left_eye", [])
        right_eye = landmarks.get("right_eye", [])
        if not chin_points or not nose_tip_pts:
            return None
        # normalized coordinates
        def to_norm(pt):
            x, y = pt
            return (x / w, y / h)
        forehead = ( (left + right) / 2 / w, max(0.02, (top - 0.02*h) / h) )  # estimate forehead
        nose_tip = to_norm(nose_tip_pts[len(nose_tip_pts)//2])
        chin = to_norm(chin_points[-1])  # bottom-most
        left_eye_top = to_norm(left_eye[1]) if left_eye else (left/w, (top+bottom)/(2*h))
        right_eye_top = to_norm(right_eye[1]) if right_eye else (right/w, (top+bottom)/(2*h))
        data = {
            "forehead": forehead,
            "nose_tip": nose_tip,
            "chin": chin,
            "left_eye_top": left_eye_top,
            "right_eye_top": right_eye_top,
            "left_cheek": (left/w, (top+bottom)/(2*h)),
            "right_cheek": (right/w, (top+bottom)/(2*h)),
        }
        return data

    @staticmethod
    def _has_cnn():
        # face_recognition may not have cnn model available; attempt is expensive; assume HOG default
        return False


def pick_backend() -> LandmarkBackend:
    # Priority: mediapipe (official) -> mediapipe-silicon -> face_recognition
    back_candidates = [("mediapipe", MediapipeBackend), ("mediapipe_silicon", MediapipeBackend), ("face_recognition", FaceRecBackend)]
    for mod_name, cls in back_candidates:
        try:
            if mod_name in ("mediapipe", "mediapipe_silicon"):
                # try import and create
                try:
                    backend = cls(module_name=mod_name if mod_name == "mediapipe" else "mediapipe_silicon")
                except TypeError:
                    backend = cls()
            else:
                backend = cls()
            return backend
        except Exception as e:
            logger.debug(f"Backend {mod_name} not available: {e}")
    raise RuntimeError("No facial-landmark backend is available. Install mediapipe or face_recognition.")


# ----- Core Detection Logic -----
def compute_chin_score(landmarks: Dict[str, Tuple[float, float]], img_wh: Tuple[int, int]) -> ChinResult:
    """
    landmarks: dict with keys 'forehead','nose_tip','chin','left_eye_top','right_eye_top','left_cheek','right_cheek'
               each is (x_norm, y_norm)
    img_wh: (width, height)
    Returns ChinResult with score 0..1 (higher = higher chin)
    """
    required = ["forehead", "nose_tip", "chin", "left_eye_top", "right_eye_top", "left_cheek", "right_cheek"]
    for r in required:
        if r not in landmarks:
            raise ValueError(f"Missing landmark: {r}")

    w, h = img_wh
    forehead = landmarks["forehead"]
    nose = landmarks["nose_tip"]
    chin = landmarks["chin"]
    left_eye = landmarks["left_eye_top"]
    right_eye = landmarks["right_eye_top"]
    left_cheek = landmarks["left_cheek"]
    right_cheek = landmarks["right_cheek"]

    # Face height: forehead to chin (pixel distance)
    face_height = euclid((forehead[0], forehead[1]), (chin[0], chin[1]), (w, h))
    # Chin segment: nose tip to chin
    chin_segment = euclid((nose[0], nose[1]), (chin[0], chin[1]), (w, h))

    # Avoid divide by zero
    if face_height <= 1.0:
        face_height = 1.0

    chin_ratio = chin_segment / face_height  # lower third / full face height
    # Normalization: typical ranges observed: 0.25..0.4 -> map to score so that lower ratio => higher score
    # We map ratio=0.2 -> score 0.95 (very high chin), ratio=0.32 -> 0.5 balanced, ratio=0.42 -> 0.05 (low)
    # Use logistic-like mapping for stability
    # transform: score = 1 - sigmoid((chin_ratio - center)/scale)
    center = 0.32
    scale = 0.035
    # logistic
    import math
    x = (chin_ratio - center) / scale
    sigmoid = 1 / (1 + math.exp(-x))
    score = 1.0 - sigmoid  # so smaller chin_ratio -> score closer to 1

    # clamp 0..1
    score = max(0.0, min(1.0, score))

    # Thresholds (configurable)
    if score > 0.66:
        label = "High Chin"
    elif score > 0.40:
        label = "Balanced Chin"
    else:
        label = "Low Chin"

    lm_used = {
        "forehead": forehead,
        "nose_tip": nose,
        "chin": chin,
        "left_eye_top": left_eye,
        "right_eye_top": right_eye,
        "left_cheek": left_cheek,
        "right_cheek": right_cheek,
    }
    return ChinResult(score=score, label=label, chin_ratio=chin_ratio, face_height=face_height,
                      chin_segment=chin_segment, landmarks_used=lm_used)


# ----- I/O helpers -----
def process_image_file(path: str, backend: LandmarkBackend) -> Optional[ChinResult]:
    import cv2
    img = cv2.imread(path)
    if img is None:
        logger.warning(f"Failed to load image: {path}")
        return None
    lms = backend.get_landmarks(img)
    if lms is None:
        logger.info(f"No face found in {path}")
        return None
    h, w = img.shape[:2]
    return compute_chin_score(lms, (w, h))


def run_webcam(backend: LandmarkBackend):
    import cv2
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Webcam not available.")
        return
    logger.info("Press ESC to exit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        lms = backend.get_landmarks(frame)
        if lms:
            res = compute_chin_score(lms, (frame.shape[1], frame.shape[0]))
            # annotate frame
            cv2.putText(frame, f"{res.label} (score {res.score:.2f})", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"chin_ratio:{res.chin_ratio:.3f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
            # draw landmarks
            for nm, (x,y) in res.landmarks_used.items():
                cx, cy = int(x*frame.shape[1]), int(y*frame.shape[0])
                cv2.circle(frame, (cx, cy), 3, (0,255,255), -1)
        cv2.imshow("High Chin Detector", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def batch_folder(folder: str, backend: LandmarkBackend, out_csv: Optional[str] = None):
    results = []
    for fname in os.listdir(folder):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            path = os.path.join(folder, fname)
            res = process_image_file(path, backend)
            if res:
                results.append((fname, res))
    if out_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file","score","label","chin_ratio","face_height","chin_segment"])
            for fname, r in results:
                writer.writerow([fname, f"{r.score:.4f}", r.label, f"{r.chin_ratio:.4f}", f"{r.face_height:.1f}", f"{r.chin_segment:.1f}"])
        logger.info(f"Wrote results to {out_csv}")
    else:
        for fname, r in results:
            print(fname, asdict(r))


# ----- CLI -----
def main_cli():
    parser = argparse.ArgumentParser(description="High Chin Detector (production-ready)")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--folder", type=str, help="Folder to batch process")
    parser.add_argument("--webcam", action="store_true", help="Run webcam mode")
    parser.add_argument("--out", type=str, default=None, help="Output CSV for folder mode")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    backend = None
    try:
        backend = pick_backend()
    except Exception as e:
        logger.error(f"No backend available: {e}")
        sys.exit(1)

    if args.image:
        res = process_image_file(args.image, backend)
        if res:
            print("RESULT:", asdict(res))
        else:
            print("No face or failed processing.")
    elif args.folder:
        batch_folder(args.folder, backend, out_csv=args.out)
    elif args.webcam:
        run_webcam(backend)
    else:
        parser.print_help()


if __name__ == "__main__":
    main_cli()
