import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

def classify_cheekbones(image_path=None, use_webcam=False):
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=not use_webcam,
                                      max_num_faces=1,
                                      refine_landmarks=True,
                                      min_detection_confidence=0.5)

    def process_frame(frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return frame, None, None, "No face detected"

        lm = results.multi_face_landmarks[0].landmark
        get = lambda i: (int(lm[i].x * w), int(lm[i].y * h))

        # key landmarks
        left_cheek, right_cheek = get(234), get(454)
        left_eye, right_eye = get(33), get(263)
        chin, nose = get(152), get(10)

        avg_cheek_y = (left_cheek[1] + right_cheek[1]) / 2
        avg_eye_y = (left_eye[1] + right_eye[1]) / 2
        face_len = chin[1] - nose[1]
        ratio = (avg_cheek_y - avg_eye_y) / face_len

        # thresholds
        if ratio >= 0.185: score, label = 0, "Very Low"
        elif ratio >= 0.160: score, label = 1, "Low"
        elif ratio >= 0.135: score, label = 2, "Below Avg"
        elif ratio >= 0.110: score, label = 3, "Average"
        elif ratio >= 0.090: score, label = 4, "Above Avg"
        elif ratio >= 0.075: score, label = 5, "High"
        else: score, label = 6, "Very High"

        # ==== Face Cropping to Fit in Frame ====
        xs = [int(l.x * w) for l in lm]
        ys = [int(l.y * h) for l in lm]
        x_min, x_max = max(min(xs) - 30, 0), min(max(xs) + 30, w)
        y_min, y_max = max(min(ys) - 30, 0), min(max(ys) + 30, h)
        cropped = frame[y_min:y_max, x_min:x_max]

        # Resize cropped face to fixed size for better viewing
        cropped = cv2.resize(cropped, (600, 600))

        # Draw landmarks (scaled to new cropped frame)
        scale_x = 600 / (x_max - x_min)
        scale_y = 600 / (y_max - y_min)
        for p in [left_cheek, right_cheek, left_eye, right_eye, chin, nose]:
            cx, cy = int((p[0] - x_min) * scale_x), int((p[1] - y_min) * scale_y)
            cv2.circle(cropped, (cx, cy), 4, (0, 255, 0), -1)

        cv2.putText(cropped, f"{label} Cheekbones ({score}/6)", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(cropped, f"Ratio: {ratio:.3f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return cropped, score, ratio, label

    # Webcam Mode
    if use_webcam:
        cap = cv2.VideoCapture(0)
        print("Press 'q' to quit")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame, _, _, _ = process_frame(frame)
            cv2.imshow("Cheekbone Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        return

    #for image testing
    if image_path:
        img = cv2.imread(image_path)
        if img is None:
            print("Image not found!")
            return
        annotated, score, ratio, label = process_frame(img)
        cv2.imshow("Cheekbone Detection (Fitted)", annotated)
        cv2.imwrite("cheekbone_result_fitted.jpg", annotated)
        print(f"Label: {label}, Score: {score}/6, Ratio: {ratio:.3f}")
        print("Saved as cheekbone_result_fitted.jpg")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


classify_cheekbones("test3.png")
# classify_cheekbones(use_webcam=True)
