import cv2
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
from seat_logic import assign_people_and_faces_to_seats

BASE_DIR = Path(__file__).resolve().parent.parent

# 1. Load seats
seats_path = BASE_DIR / "monitoring" / "seats_3rd_floor.json"
with open(seats_path, "r", encoding="utf-8") as f:
    seats = json.load(f)

# 2. Load face embeddings
db_path = BASE_DIR / "recognition" / "output" / "employee_embeddings.npz"
data = np.load(db_path)
embeddings = data["embeddings"].astype(np.float32)
employee_ids = data["employee_ids"]
grouped = defaultdict(list)
for emb, eid in zip(embeddings, employee_ids):
    grouped[str(eid)].append(emb)

employee_templates = {}
for eid, vecs in grouped.items():
    m = np.mean(vecs, axis=0)
    norm = np.linalg.norm(m)
    if norm > 0:
        employee_templates[eid] = m / norm

known_ids = list(employee_templates.keys())
template_matrix = np.vstack([employee_templates[eid] for eid in known_ids]).astype(np.float32)

# 3. Load models
yolo = YOLO(str(BASE_DIR / "yolo11n.pt"))
detector = cv2.FaceDetectorYN.create(
    str(BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx"),
    "",
    (1280, 720),
    0.35,
    0.30,
    5000
)
recognizer = cv2.FaceRecognizerSF.create(
    str(BASE_DIR / "models" / "face_recognition_sface_2021dec.onnx"),
    ""
)

# 4. Test image or video frame
video_path = BASE_DIR / "detection" / "reports" / "3rd_floor_test.mp4"
if not video_path.exists():
    video_path = BASE_DIR / "3rd Floor WS.avi"

cap = cv2.VideoCapture(str(video_path))
ret, img = cap.read()
cap.release()
if not ret or img is None:
    raise RuntimeError("Could not read frame from video")
img = cv2.resize(img, (1280, 720))
h, w = img.shape[:2]

# Run YOLO
yolo_results = yolo(img, conf=0.20, classes=[0], verbose=False)
people = [list(map(int, b.xyxy[0].tolist())) for b in yolo_results[0].boxes]

# Run YuNet
detector.setInputSize((w, h))
_, faces = detector.detect(img)
face_records = []
if faces is not None:
    for face in faces:
        fb = face[:4].astype(int)
        conf = float(face[14])
        matched_id = "Unknown"
        sim = 0.0
        try:
            aligned = recognizer.alignCrop(img, face)
            feat = recognizer.feature(aligned).flatten().astype(np.float32)
            fn = np.linalg.norm(feat)
            if fn > 0:
                feat = feat / fn
                sims = template_matrix @ feat
                best_idx = int(np.argmax(sims))
                sim = float(sims[best_idx])
                if sim >= 0.28:
                    matched_id = known_ids[best_idx]
        except Exception:
            pass
        face_records.append({"box": fb, "conf": conf, "id": matched_id, "sim": sim})

status = assign_people_and_faces_to_seats(people, face_records, seats)
print(f"Total People Detected: {len(people)}")
print(f"Total Faces Detected: {len(face_records)}")
print("=" * 70)
for sid, sdata in seats.items():
    st = status[sid]
    print(f"Seat {sid} ({sdata['name']} / {sdata.get('employee_id')}): {st['status_str']:8s} | {st['verification']}")
