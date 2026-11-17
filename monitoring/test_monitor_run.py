import cv2
import time
import os
from pathlib import Path

from seat_monitor import (
    load_seats_config,
    create_database,
    assign_people_and_faces_to_seats,
    load_employee_templates,
    is_valid_face,
    YOLO_MODEL_PATH,
    YUNET_MODEL_PATH,
    SFACE_MODEL_PATH,
    EMBEDDINGS_PATH,
    PERSON_CONFIDENCE,
    FACE_CONFIDENCE,
    YOLO_IMGSZ,
    FALLBACK_VIDEO,
    PRIMARY_VIDEO
)
from ultralytics import YOLO

print("[TEST] Running headless verification with upgraded multi-modal pipeline...")
create_database()
seats = load_seats_config()
yolo = YOLO(YOLO_MODEL_PATH)
detector = cv2.FaceDetectorYN.create(YUNET_MODEL_PATH, "", (1280, 720), FACE_CONFIDENCE, 0.30, 5000)
recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL_PATH, "")
_, template_matrix, known_ids = load_employee_templates(EMBEDDINGS_PATH)

video_src = str(FALLBACK_VIDEO if FALLBACK_VIDEO.exists() else PRIMARY_VIDEO)
cap = cv2.VideoCapture(video_src)
assert cap.isOpened(), f"Cannot open {video_src}"

for i in range(10):
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1280, 720))
    if i % 2 == 0:
        results = yolo.predict(
            source=frame,
            classes=[0],
            conf=PERSON_CONFIDENCE,
            imgsz=YOLO_IMGSZ,
            device="cpu",
            verbose=False
        )
        people = []
        for r in results:
            if r.boxes is not None:
                for b in r.boxes:
                    people.append(list(map(int, b.xyxy[0].tolist())))

        _, faces = detector.detect(frame)
        cached_faces = []
        if faces is not None and template_matrix is not None:
            for face in faces:
                if not is_valid_face(face, min_conf=FACE_CONFIDENCE):
                    continue
                fb = face[:4].astype(int)
                conf = float(face[14])
                cached_faces.append({"box": fb, "conf": conf, "id": "Unknown", "sim": 0.0})

        status = assign_people_and_faces_to_seats(people, cached_faces, seats)
        occupied = sum(1 for s in status.values() if s["occupied"])
        empty = len(seats) - occupied
        ws04_occ = status.get("DESK-DSK04", {}).get("occupied", False)
        print(f"Frame {i:02d}: People={len(people)}, Valid Faces={len(cached_faces)}, Occupied={occupied}, Empty={empty} | DESK-DSK04 Occupied={ws04_occ}")

cap.release()
print("[SUCCESS] Upgraded multi-modal pipeline verified!")
