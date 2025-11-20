import cv2
import mediapipe as mp
import numpy as np
import xgboost as xgb
import joblib

mp_face_mesh = mp.solutions.face_mesh

# Load your trained regression model (0–6 output)
model = joblib.load("severity_model.pkl")

# regions of interest
NASOLABIAL_IDX = [1, 2, 98, 327, 205, 50, 187]
MARIONETTE_IDX = [58, 288, 6, 199, 200]


def landmark_array(mesh_results, w, h):
    pts = []
    for lm in mesh_results.multi_face_landmarks[0].landmark:
        pts.append([lm.x * w, lm.y * h, lm.z])
    return np.array(pts)


def region_features(landmarks, idxs):
    region = landmarks[idxs]
    
    # geometric features
    curvature = np.mean(np.abs(np.diff(region[:, :2], axis=0)))
    depth = np.std(region[:, 2])
    length = np.linalg.norm(region[0, :2] - region[-1, :2])
    
    # symmetry (left/right variation)
    sym = np.std(region[:, 0])
    
    return np.array([curvature, depth, length, sym])


def extract_features(landmarks):
    nl = region_features(landmarks, NASOLABIAL_IDX)
    mr = region_features(landmarks, MARIONETTE_IDX)
    return np.concatenate([nl, mr])


def severity_score(features):
    pred = model.predict(features.reshape(1, -1))[0]
    return float(np.clip(pred, 0, 6))  # force into 0–6 range


# ---- run on image ----
def evaluate_image(path):
    img = cv2.imread(path)
    h, w = img.shape[:2]

    with mp_face_mesh.FaceMesh(static_image_mode=True) as mesh:
        res = mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            raise Exception("No face detected.")

        landmarks = landmark_array(res, w, h)
        feats = extract_features(landmarks)
        score = severity_score(feats)
        return score


# Example usage
if __name__ == "__main__":
    score = evaluate_image("face.jpg")
    print("Estimated cosmetic severity (0–6):", score)
