import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import argparse
import logging
import random
from pathlib import Path
import json
import numpy as np
import cv2
import dlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                        ReduceLROnPlateau)


# Configuration / Constants
SEED = 42
IMAGE_SIZE = (96, 96)
DATA_DIR = "lips_dataset"
PREDICT_IMAGE = "thick2.jpg"
DLIB_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

MIN_SAMPLES_FOR_DL = 50  # if dataset smaller, fallback to geometric regression

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("lip-severity")

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# Dlib setup
if not os.path.exists(DLIB_PREDICTOR_PATH):
    logger.error(f"Dlib predictor not found at {DLIB_PREDICTOR_PATH}. Download and place it in project root.")
    # don't exit: allow user to still run geometric fallback
predictor = dlib.shape_predictor(DLIB_PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()


def extract_lip_region(image_bgr):
    """Return resized lip crop or None if not detected."""
    if image_bgr is None:
        return None
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    # take the largest face
    face = max(faces, key=lambda r: r.width() * r.height())
    landmarks = predictor(gray, face)
    xs = [landmarks.part(n).x for n in range(48, 60)]
    ys = [landmarks.part(n).y for n in range(48, 60)]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    pad_x = int(0.15 * (x_max - x_min + 1))  # proportional padding
    pad_y = int(0.25 * (y_max - y_min + 1))
    x_min, x_max = max(0, x_min - pad_x), x_max + pad_x
    y_min, y_max = max(0, y_min - pad_y), y_max + pad_y
    h, w = image_bgr.shape[:2]
    x_max, y_max = min(w - 1, x_max), min(h - 1, y_max)
    crop = image_bgr[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, IMAGE_SIZE)
    return crop

def compute_lip_ratio(image_bgr):
    """Geometric measure: mean vertical lip height / face height ratio."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    face = max(faces, key=lambda r: r.width() * r.height())
    landmarks = predictor(gray, face)
    # vertical distances
    lip_h = np.mean([
        abs(landmarks.part(51).y - landmarks.part(57).y),
        abs(landmarks.part(50).y - landmarks.part(58).y),
        abs(landmarks.part(52).y - landmarks.part(56).y)
    ])
    face_h = abs(landmarks.part(27).y - landmarks.part(8).y)
    if face_h == 0:
        return None
    return float(lip_h / face_h)

def load_images_from_folder(folder):
    """Load image paths from folder (common image extensions)."""
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    p = Path(folder)
    if not p.exists():
        logger.error(f"Dataset folder {folder} does not exist.")
        return []
    image_paths = sorted([str(x) for x in p.iterdir() if x.suffix.lower() in exts])
    return image_paths


# Dataset building
def build_dataset_auto(data_dir=DATA_DIR):
    image_paths = load_images_from_folder(data_dir)
    if len(image_paths) == 0:
        logger.warning("No images found in dataset folder.")
        return None, None
    crops = []
    ratios = []
    got = 0
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        crop = extract_lip_region(img)
        if crop is None:
            logger.debug(f"No lips for {p}; skipping.")
            continue
        r = compute_lip_ratio(img)
        if r is None:
            continue
        crops.append(img_to_array(crop) / 255.0)
        ratios.append(r)
        got += 1

    if got == 0:
        logger.error("No valid lip crops found. Check images and dlib predictor.")
        return None, None

    ratios = np.array(ratios)
    min_r, max_r = float(np.min(ratios)), float(np.max(ratios))
    # avoid zero range
    if np.isclose(max_r - min_r, 0.0):
        max_r = min_r + 1e-6

    # map ratio -> [0,6]
    scores = np.clip((ratios - min_r) / (max_r - min_r) * 6.0, 0.0, 6.0)
    crops = np.array(crops)

    # save scaler
    scaler = {"min_ratio": min_r, "max_ratio": max_r}
    with open(OUTPUTS_DIR / "scaler.json", "w") as f:
        json.dump(scaler, f)
    logger.info(f"Built dataset: {crops.shape[0]} samples (min_r={min_r:.6f}, max_r={max_r:.6f})")
    return crops, scores


# Model (Transfer learning)
def build_tf_regression_model(input_shape=(96,96,3)):
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation="linear")(x)
    model = Model(inputs=base.input, outputs=out)
    # Freeze most layers initially
    for layer in base.layers[:-20]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="mse", metrics=["mae"])
    return model

#Train pipeline
def train_pipeline():
    X, y = build_dataset_auto(DATA_DIR)
    if X is None or y is None:
        logger.error("Dataset build failed. Aborting training.")
        return

    n_samples = X.shape[0]
    if n_samples < MIN_SAMPLES_FOR_DL:
        logger.warning(f"Only {n_samples} samples found. Using geometric regression fallback.")
        logger.info("Saved scaler; you can predict geometric score using compute_lip_ratio + scaler mapping.")
        return

    # Train / val split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)
    logger.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        brightness_range=[0.8, 1.2],
        zoom_range=0.15,
        horizontal_flip=True
    )
    train_gen = datagen.flow(X_train, y_train, batch_size=16, shuffle=True)
    val_gen = datagen.flow(X_val, y_val, batch_size=16, shuffle=False)
    model = build_tf_regression_model(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    model.summary()

    # Callbacks
    ckpt_path = MODELS_DIR / "lip_severity_best.h5"
    checkpoint = ModelCheckpoint(str(ckpt_path), monitor="val_mae", save_best_only=True, verbose=1)
    early = EarlyStopping(monitor="val_mae", patience=8, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor="val_mae", factor=0.5, patience=4, verbose=1, min_lr=1e-7)

    history = model.fit(
        train_gen,
        epochs=60,
        validation_data=val_gen,
        callbacks=[checkpoint, early, reduce_lr],
        verbose=2
    )

    # Evaluation on validation set (use whole val array)
    preds = model.predict(X_val).flatten()
    mae = mean_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)
    logger.info(f"Validation MAE: {mae:.4f}, R2: {r2:.4f}")

    # Save final model and metrics
    final_model_path = MODELS_DIR / "lip_severity_model.h5"
    model.save(final_model_path)
    metrics = {"val_mae": float(mae), "val_r2": float(r2), "n_samples": int(n_samples)}
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f)

    # Save training plot
    plt.figure(figsize=(6,4))
    plt.plot(history.history.get("mae", []), label="train_mae")
    plt.plot(history.history.get("val_mae", []), label="val_mae")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.title("Training MAE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "training_mae.png")
    logger.info(f"Model saved to {final_model_path}, training plot saved to outputs/")


# Predict pipeline
def predict_image(image_path, use_geometric_if_small=True):
    scaler_path = OUTPUTS_DIR / "scaler.json"
    model_path = MODELS_DIR / "lip_severity_best.h5"
    img = cv2.imread(image_path)
    if img is None:
        logger.error("Could not read image: %s", image_path)
        return None

    ratio = compute_lip_ratio(img)
    if ratio is None:
        logger.error("No face/lips detected for %s", image_path)
        return None

    # If model exists, prefer model
    if model_path.exists():
        try:
            model = load_model(model_path)
            crop = extract_lip_region(img)
            if crop is None:
                logger.error("Could not extract lip crop for model prediction.")
                # fallback to geometric
            else:
                inp = np.expand_dims(img_to_array(crop) / 255.0, axis=0)
                score = float(model.predict(inp)[0][0])
                score = int(np.clip(round(score), 0, 6))  # ensure integer within [0,6]
                logger.info(f"Predicted score (model): {score}")
                return score

        except Exception as e:
            logger.exception("Error loading/predicting with model, will fallback to geometric mapping: %s", e)

    # Fallback: geometric mapping using saved scaler if present, otherwise use ratio -> scaled by observed global range heuristic
    if scaler_path.exists():
        with open(scaler_path, "r") as f:
            scaler = json.load(f)
            min_r, max_r = scaler["min_ratio"], scaler["max_ratio"]
            if np.isclose(max_r - min_r, 0.0):
                max_r = min_r + 1e-6
            score = np.clip((ratio - min_r) / (max_r - min_r) * 6.0, 0.0, 6.0)
            logger.info(f"Predicted score (geometric, using scaler): {score:.3f}")
            score = int(np.clip(round(score), 0, 6))
            return score

    else:
        # heuristic: assume plausible ratio range [0.015, 0.07] (depends on dataset); map in absence of scaler
        # NOTE: you should replace these when you collect more data
        min_r, max_r = 0.015, 0.07
        score = np.clip((ratio - min_r) / (max_r - min_r) * 6.0, 0.0, 6.0)
        logger.info(f"Predicted score (geometric, heuristic range): {score:.3f}")
        score = int(np.clip(round(score), 0, 6))
        return score


#cli
def parse_args():
    p = argparse.ArgumentParser(description="Lip Thickness Severity (0-6) pipeline")
    p.add_argument("--mode", choices=["train", "predict"], default="train", help="Operation mode")
    p.add_argument("--predict_image", type=str, default=PREDICT_IMAGE, help="Image path for prediction")
    return p.parse_args()


# Main

def main():
    args = parse_args()
    if args.mode == "train":
        train_pipeline()
    elif args.mode == "predict":
        score = predict_image(args.predict_image)
        if score is not None:
            category = "Average" if np.isclose(score, 3.0, atol=0.25) else ("Thin" if score < 3.0 else "Thick")
            print(f"Predicted severity: {score} -> {category}")

    else:
        logger.error("Unknown mode.")

if __name__ == "__main__":
    main()
