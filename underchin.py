import cv2
import mediapipe as mp
import numpy as np
import json


# Distance function using numpy for vectorized operations
def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


# Function to calculate the underchin fat analysis
def analyze_underchin_fat(image_path):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Image not found"}

    # Get image dimensions
    h, w = img.shape[:2]

    # Initialize MediaPipe FaceMesh
    mp_face = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=1
    )

    # Convert to RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = mp_face.process(rgb)

    # Check if any face landmarks are detected
    if not res.multi_face_landmarks:
        return {"error": "No face detected"}

    # Extract landmarks and calculate inter-pupillary distance (IPD)
    landmarks = [(p.x * w, p.y * h) for p in res.multi_face_landmarks[0].landmark]
    ipd = dist(landmarks[33], landmarks[263])

    # Define chin and jaw landmarks
    chin = landmarks[152]
    left_jaw = landmarks[172]
    right_jaw = landmarks[136]

    # Curvature computation using cross product (normalizing with IPD)
    curvature = np.abs(np.cross(np.array(right_jaw) - np.array(left_jaw),
                                np.array(left_jaw) - np.array(chin))) / (dist(left_jaw, right_jaw) + 1e-6)
    curvature_norm = curvature / (ipd + 1e-6)

    # Scale curvature: more curvature implies less fat
    curvature_score = (0.32 - curvature_norm) / 0.15
    curvature_score = float(np.clip(curvature_score, 0, 1))

    # Shadow intensity calculation
    under_pts = [landmarks[i] for i in range(84, 94)]
    ux, uy = np.mean(under_pts, axis=0).astype(int)
    uy = min(h - 2, uy)

    patch = img[max(0, uy - 20):min(h, uy + 20), ux - 30:ux + 30]
    if patch.size > 0:
        patch_y = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        face_y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0].mean()
        shadow = float(face_y - patch_y.mean())
    else:
        shadow = 0.0

    # Normalize shadow score
    shadow_score = (shadow - 10) / 60
    shadow_score = float(np.clip(shadow_score, 0, 1))

    # Final combined score: weighted average of shadow and curvature scores
    final_score = 0.6 * shadow_score + 0.4 * curvature_score
    severity = int(round(final_score * 6))

    # Return results in a structured format
    return {
        "feature": "Under Chin Fat",
        "severity_0_to_6": severity,
        "scores": {
            "shadow_score": shadow_score,
            "curvature_score": curvature_score
        },
        "raw_metrics": {
            "shadow_intensity": shadow,
            "curvature_norm": curvature_norm
        }
    }


# Testing the function
if __name__ == "__main__":
    image_path = "low2.png"
    print(json.dumps(analyze_underchin_fat(image_path), indent=4))
