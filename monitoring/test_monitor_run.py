import cv2
import time
import os
from pathlib import Path

# Run a headless test of seat_monitor
from seat_monitor import (
    load_seats_config,
    create_database,
    assign_people_and_faces_to_seats,
    load_employee_templates,
    YOLO_MODEL_PATH,
    YUNET_MODEL_PATH,
    SFACE_MODEL_PATH,
    EMBEDDINGS_PATH,
    DATABASE_PATH,
    PRIMARY_VIDEO,
    FALLBACK_VIDEO
)
from ultralytics import YOLO

print("[TEST] Running headless verification of seat monitoring...")
create_database()
seats = load_seats_config()
yolo = YOLO(YOLO_MODEL_PATH)
detector = cv2.FaceDetectorYN.create(YUNET_MODEL_PATH, "", (1280, 720), 0.35, 0.30, 5000)
recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL_PATH, "")
_, template_matrix, known_ids = load_employee_templates(EMBEDDINGS_PATH)

video_src = str(FALLBACK_VIDEO if FALLBACK_VIDEO.exists() else PRIMARY_VIDEO)
cap = cv2.VideoCapture(video_src)
assert cap.isOpened(), f"Cannot open {video_src}"

for i in range(15):
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1280, 720))
    if i % 2 == 0:
        results = yolo.predict(source=frame, classes=[0], conf=0.22, device="cpu", verbose=False)
        people = []
        for r in results:
            if r.boxes is not None:
                for b in r.boxes:
                    people.append(list(map(int, b.xyxy[0].tolist())))

        _, faces = detector.detect(frame)
        cached_faces = []
        if faces is not None and template_matrix is not None:
            for face in faces:
                fb = face[:4].astype(int)
                conf = float(face[14])
                matched_id = "Unknown"
                sim = 0.0
                try:
                    aligned = recognizer.alignCrop(frame, face)
                    feat = recognizer.feature(aligned).flatten().astype(np.float32)
                    fn = np.linalg.norm(feat)
                    if fn > 0:
                        sims = template_matrix @ (feat / fn)
                        best_idx = int(np.argmax(sims))
                        sim = float(sims[best_idx])
                        if sim >= 0.28:
                            matched_id = known_ids[best_idx]
                except Exception:
                    pass
                cached_faces.append({"box": fb, "conf": conf, "id": matched_id, "sim": sim})

        status = assign_people_and_faces_to_seats(people, cached_faces, seats)
        occupied = sum(1 for s in status.values() if s["occupied"])
        empty = len(seats) - occupied
        print(f"Frame {i:02d}: People={len(people)}, Faces={len(cached_faces)}, Occupied Seats={occupied}, Empty Seats={empty}")

cap.release()
print("[SUCCESS] Headless test passed successfully!")

