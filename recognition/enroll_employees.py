
import cv2
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

# ==================================================
# PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"

YUNET_MODEL = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = MODEL_DIR / "face_recognition_sface_2021dec.onnx"

OUTPUT_DIR = BASE_DIR / "recognition" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "employee_embeddings.npz"
REPORT_FILE = OUTPUT_DIR / "enrollment_report.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ==================================================
# SETTINGS
# ==================================================

DETECTION_THRESHOLD = 0.70

# Maximum images per employee to prevent one employee
# with hundreds of similar frames dominating enrollment.
MAX_IMAGES_PER_EMPLOYEE = 10

# ==================================================
# LOAD MODELS
# ==================================================

print("[INFO] Loading YuNet and SFace models...")

detector = cv2.FaceDetectorYN.create(
    str(YUNET_MODEL),
    "",
    (320, 320),
    DETECTION_THRESHOLD,
    0.30,
    5000
)

recognizer = cv2.FaceRecognizerSF.create(
    str(SFACE_MODEL),
    ""
)

print("[SUCCESS] Both models loaded.")

# ==================================================
# PROCESS DATASET
# ==================================================

embeddings = []
employee_ids = []
report = {}

for employee_dir in sorted(DATASET_DIR.iterdir()):

    if not employee_dir.is_dir():
        continue

    employee_id = employee_dir.name

    # Skip unknown identity folder
    if employee_id.lower() == "unknown":
        print(f"[SKIP] {employee_id}: unknown identity folder")
        continue

    image_paths = sorted([
        p for p in employee_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        and not p.name.lower().startswith("unknown_")
    ])

    accepted = 0
    rejected = 0
    employee_embeddings = []

    print(f"\n[INFO] Processing {employee_id}")

    for image_path in image_paths:

        if accepted >= MAX_IMAGES_PER_EMPLOYEE:
            break

        image = cv2.imread(str(image_path))

        if image is None:
            rejected += 1
            continue

        height, width = image.shape[:2]

        detector.setInputSize((width, height))

        _, faces = detector.detect(image)

        # Exactly one face required
        if faces is None or len(faces) != 1:
            rejected += 1
            continue

        face = faces[0]

        x, y, w, h = face[:4]

        # Reject very small face regions
        if w < 40 or h < 40:
            rejected += 1
            continue

        # Align face
        aligned_face = recognizer.alignCrop(image, face)

        # Generate embedding
        feature = recognizer.feature(aligned_face)

        # Normalize embedding
        feature = feature.flatten().astype(np.float32)
        norm = np.linalg.norm(feature)

        if norm == 0:
            rejected += 1
            continue

        feature = feature / norm

        employee_embeddings.append(feature)
        accepted += 1

    if employee_embeddings:

        # Keep individual embeddings for later comparison
        for feature in employee_embeddings:
            embeddings.append(feature)
            employee_ids.append(employee_id)

    report[employee_id] = {
        "accepted": accepted,
        "rejected": rejected,
        "status": "enrolled" if accepted else "needs_review"
    }

    print(
        f"[RESULT] {employee_id}: "
        f"Accepted={accepted}, Rejected={rejected}"
    )

# ==================================================
# SAVE ENROLLMENT DATABASE
# ==================================================

if not embeddings:
    print("\n[ERROR] No valid employee faces were enrolled.")
    raise SystemExit(1)

embedding_array = np.vstack(embeddings).astype(np.float32)
employee_array = np.array(employee_ids)

np.savez_compressed(
    OUTPUT_FILE,
    embeddings=embedding_array,
    employee_ids=employee_array
)

with open(REPORT_FILE, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4)

print("\n======================================")
print("[SUCCESS] ENROLLMENT COMPLETED")
print("Employees enrolled:", len(set(employee_ids)))
print("Total embeddings:", len(embedding_array))
print("Embedding dimensions:", embedding_array.shape[1])
print("Database:", OUTPUT_FILE)
print("Report:", REPORT_FILE)
print("======================================")
