
import cv2
import numpy as np
import time
from pathlib import Path
from collections import defaultdict

# ==================================================
# PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = BASE_DIR / "Cafeteria.avi"

DATABASE_PATH = (
    BASE_DIR / "recognition" / "output" / "employee_embeddings.npz"
)

YUNET_MODEL = (
    BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx"
)

SFACE_MODEL = (
    BASE_DIR / "models" / "face_recognition_sface_2021dec.onnx"
)

# ==================================================
# RECOGNITION SETTINGS
# ==================================================

# Initial test threshold only. Must be calibrated
# using separate known-person and unknown-person footage.
SIMILARITY_THRESHOLD = 0.363

# Require some separation between the best and second-best
# employee match to reduce ambiguous predictions.
MINIMUM_MARGIN = 0.05

# ==================================================
# CHECK FILES
# ==================================================

for path in [VIDEO_PATH, DATABASE_PATH, YUNET_MODEL, SFACE_MODEL]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

# ==================================================
# LOAD ENROLLMENT DATABASE
# ==================================================

data = np.load(DATABASE_PATH)

embeddings = data["embeddings"].astype(np.float32)
employee_ids = data["employee_ids"]

# Group embeddings by employee ID.
grouped = defaultdict(list)

for embedding, employee_id in zip(embeddings, employee_ids):
    grouped[str(employee_id)].append(embedding)

# Average the templates per employee and normalize.
employee_templates = {}

for employee_id, vectors in grouped.items():

    mean_vector = np.mean(vectors, axis=0)

    norm = np.linalg.norm(mean_vector)

    if norm > 0:
        employee_templates[employee_id] = mean_vector / norm

known_ids = list(employee_templates.keys())

template_matrix = np.vstack([
    employee_templates[employee_id]
    for employee_id in known_ids
]).astype(np.float32)

print("[SUCCESS] Loaded employee database")
print("[INFO] Employees:", len(known_ids))

# ==================================================
# LOAD YUNET AND SFACE
# ==================================================

detector = cv2.FaceDetectorYN.create(
    str(YUNET_MODEL),
    "",
    (320, 320),
    0.70,
    0.30,
    5000
)

recognizer = cv2.FaceRecognizerSF.create(
    str(SFACE_MODEL),
    ""
)

print("[SUCCESS] YuNet and SFace loaded")

# ==================================================
# OPEN CCTV VIDEO
# ==================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Could not open cafeteria video")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

detector.setInputSize((width, height))

print("[INFO] Video resolution:", width, "x", height)
print("[INFO] Starting employee recognition...")

# ==================================================
# PROCESS VIDEO
# ==================================================

previous_time = time.time()

try:

    while True:

        success, frame = cap.read()

        if not success:
            print("[INFO] Video completed")
            break

        _, faces = detector.detect(frame)

        face_count = 0

        if faces is not None:

            face_count = len(faces)

            for face in faces:

                x, y, w, h = face[:4].astype(int)

                # Align the detected face
                aligned_face = recognizer.alignCrop(frame, face)

                # Generate SFace embedding
                feature = recognizer.feature(aligned_face)
                feature = feature.flatten().astype(np.float32)

                norm = np.linalg.norm(feature)

                if norm == 0:
                    continue

                feature = feature / norm

                # Compare with employee templates
                similarities = template_matrix @ feature

                sorted_indices = np.argsort(similarities)[::-1]

                best_index = sorted_indices[0]
                best_score = float(similarities[best_index])

                best_employee = known_ids[best_index]

                if len(sorted_indices) > 1:
                    second_score = float(
                        similarities[sorted_indices[1]]
                    )
                else:
                    second_score = -1.0

                margin = best_score - second_score

                # Conservative initial decision
                if (
                    best_score >= SIMILARITY_THRESHOLD
                    and margin >= MINIMUM_MARGIN
                ):
                    label = f"{best_employee} ({best_score:.2f})"
                    color = (0, 255, 0)
                else:
                    label = f"Unknown ({best_score:.2f})"
                    color = (0, 0, 255)

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
                    (x, max(y - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        # FPS
        current_time = time.time()
        fps = 1 / max(current_time - previous_time, 0.001)
        previous_time = current_time

        cv2.putText(
            frame,
            f"Faces: {face_count}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.imshow("Cafeteria Employee Recognition Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("[INFO] Stopped by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Resources released")
