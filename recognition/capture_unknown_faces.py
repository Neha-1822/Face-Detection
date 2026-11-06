
import cv2
import numpy as np
import csv
import time
from pathlib import Path
from datetime import datetime

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = BASE_DIR / "Cafeteria.avi"

FACE_MODEL = BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx"
RECOGNITION_MODEL = BASE_DIR / "models" / "face_recognition_sface_2021dec.onnx"

KNOWN_DATABASE = (
    BASE_DIR / "recognition" / "output" / "employee_embeddings.npz"
)

OUTPUT_DIR = BASE_DIR / "recognition" / "unknown_faces"
IMAGE_DIR = OUTPUT_DIR / "images"

CSV_PATH = OUTPUT_DIR / "unknown_faces.csv"
UNKNOWN_EMBEDDINGS_PATH = OUTPUT_DIR / "unknown_embeddings.npy"

# ==========================================================
# SETTINGS
# ==========================================================

FACE_SCORE_THRESHOLD = 0.70

# Same provisional recognition thresholds as your test script
RECOGNITION_THRESHOLD = 0.363
MARGIN_THRESHOLD = 0.05

# Suppress nearly identical face captures.
# A different angle may still be saved as a new sample.
DUPLICATE_THRESHOLD = 0.97

MIN_FACE_SIZE = 30
MIN_BLUR_SCORE = 5.0

# ==========================================================
# CREATE DIRECTORIES
# ==========================================================

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# LOAD EMPLOYEE DATABASE
# ==========================================================

print("[INFO] Loading employee database...")

data = np.load(KNOWN_DATABASE)

all_embeddings = data["embeddings"].astype(np.float32)
employee_ids = data["employee_ids"]

unique_ids = np.unique(employee_ids)

employee_templates = []

for employee_id in unique_ids:

    employee_vectors = all_embeddings[
        employee_ids == employee_id
    ]

    template = np.mean(employee_vectors, axis=0)

    norm = np.linalg.norm(template)

    if norm > 0:
        template = template / norm

    employee_templates.append(template)

employee_templates = np.asarray(
    employee_templates,
    dtype=np.float32
)

print(f"[SUCCESS] Employees loaded: {len(unique_ids)}")

# ==========================================================
# LOAD YUNET AND SFACE
# ==========================================================

print("[INFO] Loading face detection model...")

detector = cv2.FaceDetectorYN.create(
    str(FACE_MODEL),
    "",
    (640, 480),
    FACE_SCORE_THRESHOLD,
    0.30,
    5000
)

print("[INFO] Loading face recognition model...")

recognizer = cv2.FaceRecognizerSF.create(
    str(RECOGNITION_MODEL),
    ""
)

print("[SUCCESS] Models loaded.")

# ==========================================================
# LOAD PREVIOUSLY CAPTURED UNKNOWN EMBEDDINGS
# ==========================================================

if UNKNOWN_EMBEDDINGS_PATH.exists():

    saved_embeddings = np.load(
        UNKNOWN_EMBEDDINGS_PATH
    ).astype(np.float32)

else:

    saved_embeddings = np.empty(
        (0, 128),
        dtype=np.float32
    )

print(
    f"[INFO] Previous unknown samples: "
    f"{len(saved_embeddings)}"
)

# ==========================================================
# PREPARE CSV FILE
# ==========================================================

if not CSV_PATH.exists():

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "filename",
            "capture_datetime",
            "video_time_seconds",
            "best_employee_match",
            "best_similarity",
            "second_similarity",
            "match_margin",
            "face_detection_score"
        ])

# ==========================================================
# OPEN VIDEO
# ==========================================================

print("[INFO] Opening cafeteria video...")

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

detector.setInputSize((frame_width, frame_height))

print(f"[INFO] Video resolution: {frame_width}x{frame_height}")
print("[INFO] Unknown face collection started.")
print("[INFO] Press Q to stop.")

capture_count = 0
frame_count = 0

# ==========================================================
# PROCESS VIDEO
# ==========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("[INFO] Video completed.")
        break

    frame_count += 1

    video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    height, width = frame.shape[:2]

    detector.setInputSize((width, height))

    _, faces = detector.detect(frame)

    if faces is not None:

        for face in faces:

            x, y, w, h = face[:4].astype(int)

            detection_score = float(face[14])

            # Ignore extremely small faces
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                continue

            # --------------------------------------------------
            # ALIGN FACE AND GENERATE SFACE EMBEDDING
            # --------------------------------------------------

            face_row = face.reshape(1, 15)

            aligned_face = recognizer.alignCrop(
                frame,
                face_row
            )

            feature = recognizer.feature(aligned_face)

            feature = feature.flatten().astype(np.float32)

            norm = np.linalg.norm(feature)

            if norm == 0:
                continue

            feature = feature / norm

            # --------------------------------------------------
            # COMPARE AGAINST ENROLLED EMPLOYEES
            # --------------------------------------------------

            similarities = employee_templates @ feature

            sorted_indices = np.argsort(similarities)[::-1]

            best_index = sorted_indices[0]

            best_score = float(similarities[best_index])

            second_score = float(similarities[sorted_indices[1]])

            best_employee = str(unique_ids[best_index])

            margin = best_score - second_score

            # --------------------------------------------------
            # UNKNOWN / UNCERTAIN IDENTITY
            # --------------------------------------------------

            is_unknown = (
                best_score < RECOGNITION_THRESHOLD
                or margin < MARGIN_THRESHOLD
            )

            if is_unknown:

                # Expand crop slightly around the face
                padding = int(max(w, h) * 0.35)

                x1 = max(0, x - padding)
                y1 = max(0, y - padding)

                x2 = min(width, x + w + padding)
                y2 = min(height, y + h + padding)

                face_crop = frame[y1:y2, x1:x2]

                if face_crop.size == 0:
                    continue

                # Reject very blurry crops
                gray_crop = cv2.cvtColor(
                    face_crop,
                    cv2.COLOR_BGR2GRAY
                )

                blur_score = cv2.Laplacian(
                    gray_crop,
                    cv2.CV_64F
                ).var()

                if blur_score < MIN_BLUR_SCORE:
                    continue

                # --------------------------------------------------
                # AVOID SAVING NEAR-IDENTICAL FRAMES
                # --------------------------------------------------

                duplicate = False

                if len(saved_embeddings) > 0:

                    old_similarities = (
                        saved_embeddings @ feature
                    )

                    if np.max(old_similarities) >= DUPLICATE_THRESHOLD:
                        duplicate = True

                if not duplicate:

                    capture_count += 1

                    timestamp = datetime.now().strftime(
                        "%Y%m%d_%H%M%S"
                    )

                    filename = (
                        f"unknown_{capture_count:06d}_"
                        f"{timestamp}.jpg"
                    )

                    image_path = IMAGE_DIR / filename

                    cv2.imwrite(
                        str(image_path),
                        face_crop
                    )

                    # Store embedding to suppress duplicates
                    saved_embeddings = np.vstack([
                        saved_embeddings,
                        feature.reshape(1, -1)
                    ])

                    np.save(
                        UNKNOWN_EMBEDDINGS_PATH,
                        saved_embeddings
                    )

                    # Save capture details
                    with open(
                        CSV_PATH,
                        "a",
                        newline="",
                        encoding="utf-8"
                    ) as file:

                        writer = csv.writer(file)

                        writer.writerow([
                            filename,
                            datetime.now().isoformat(
                                timespec="seconds"
                            ),
                            round(video_time, 2),
                            best_employee,
                            round(best_score, 4),
                            round(second_score, 4),
                            round(margin, 4),
                            round(detection_score, 4)
                        ])

                    print(
                        f"[CAPTURED] {filename} | "
                        f"Best match={best_employee} | "
                        f"Score={best_score:.3f}"
                    )

                # Display unknown or uncertain face
                label = f"UNKNOWN ({best_score:.2f})"

                color = (0, 0, 255)

            else:

                label = f"{best_employee} ({best_score:.2f})"

                color = (0, 255, 0)

            # --------------------------------------------------
            # DISPLAY
            # --------------------------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            cv2.putText(
                frame,
                label,
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2
            )

    cv2.putText(
        frame,
        f"Unknown captures: {capture_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow("Unknown Face Collector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================================================
# CLEANUP
# ==========================================================

cap.release()
cv2.destroyAllWindows()

print("\n======================================")
print("UNKNOWN FACE COLLECTION COMPLETED")
print("======================================")

print(f"New captures this run: {capture_count}")
print(f"Images folder: {IMAGE_DIR}")
print(f"CSV report: {CSV_PATH}")
print(f"Saved embeddings: {UNKNOWN_EMBEDDINGS_PATH}")
