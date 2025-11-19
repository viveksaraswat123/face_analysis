import cv2
import numpy as np
import mediapipe as mp

mp_face = mp.solutions.face_mesh


def get_lip_ratio_mediapipe(img):

    with mp_face.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,             
        min_detection_confidence=0.6
    ) as fm:

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = fm.process(rgb)

        if not results.multi_face_landmarks:
            return None

        lm = results.multi_face_landmarks[0]
        h, w, _ = img.shape

        # Helper to convert landmark to pixel coords
        def P(i):
            p = lm.landmark[i]
            return np.array([p.x * w, p.y * h], dtype=np.float32)

        # Upper lip (take median of multiple vertical segments)
        upper_points = [
            np.linalg.norm(P(13) - P(14)),  # main upper lip height
            np.linalg.norm(P(0) - P(37)),   # extra stability point
        ]
        upper = float(np.median(upper_points))

        # Lower lip (median of multiple segments)
        lower_points = [
            np.linalg.norm(P(14) - P(17)),  # main lower lip height
            np.linalg.norm(P(15) - P(84)),
            np.linalg.norm(P(16) - P(87)),
        ]
        lower = float(np.median(lower_points))

        lip_h = (upper + lower) / 2.0

        face_h = np.linalg.norm(P(10) - P(152))

        if face_h < 1:
            return None

        ratio = lip_h / face_h
        return ratio


def map_ratio_to_0_6(ratio):


    if ratio < 0.040:
        return 0  # extremely thin
    elif ratio < 0.046:
        return 1  # thin
    elif ratio < 0.052:
        return 2  # slightly thin
    elif ratio < 0.058:
        return 3  # normal
    elif ratio < 0.063:
        return 4  # full
    elif ratio < 0.070:
        return 5  # thick
    else:
        return 6  # very thick / augmented

def get_lip_severity(img):
    ratio = get_lip_ratio_mediapipe(img)
    if ratio is None:
        return None

    severity = map_ratio_to_0_6(ratio)
    return severity
