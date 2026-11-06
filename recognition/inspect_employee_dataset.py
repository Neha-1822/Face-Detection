
import cv2
import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx"

# Image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Load YuNet
detector = cv2.FaceDetectorYN.create(
    str(MODEL_PATH),
    "",
    (320, 320),
    0.70,
    0.30,
    5000
)

print("\n========== EMPLOYEE DATASET AUDIT ==========\n")

total_images = 0
total_readable = 0
total_one_face = 0
total_no_face = 0
total_multiple_faces = 0
total_flagged_names = 0

for employee_dir in sorted(DATASET_DIR.iterdir()):

    if not employee_dir.is_dir():
        continue

    employee_id = employee_dir.name

    # Do not enroll the Unknown folder
    if employee_id.lower() == "unknown":
        print(f"{employee_id}: SKIPPED")
        continue

    images = [
        p for p in employee_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    one_face = 0
    no_face = 0
    multiple_faces = 0
    unreadable = 0
    flagged_names = 0

    for image_path in images:

        # Flag files with unknown names for manual review
        if image_path.name.lower().startswith("unknown_"):
            flagged_names += 1

        image = cv2.imread(str(image_path))

        if image is None:
            unreadable += 1
            continue

        total_readable += 1

        height, width = image.shape[:2]
        detector.setInputSize((width, height))

        _, faces = detector.detect(image)

        face_count = 0 if faces is None else len(faces)

        if face_count == 0:
            no_face += 1
        elif face_count == 1:
            one_face += 1
        else:
            multiple_faces += 1

    total_images += len(images)
    total_one_face += one_face
    total_no_face += no_face
    total_multiple_faces += multiple_faces
    total_flagged_names += flagged_names

    print(
        f"{employee_id}: "
        f"Images={len(images)}, "
        f"One face={one_face}, "
        f"No face={no_face}, "
        f"Multiple faces={multiple_faces}, "
        f"Unreadable={unreadable}, "
        f"Unknown-named={flagged_names}"
    )

print("\n=============== FINAL SUMMARY ===============")
print("Total images:", total_images)
print("Readable images:", total_readable)
print("Images with exactly one face:", total_one_face)
print("Images with no face:", total_no_face)
print("Images with multiple faces:", total_multiple_faces)
print("Unknown-named images:", total_flagged_names)
print("==============================================")
