
import cv2
import os
import time

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

video_path = os.path.join(
    BASE_DIR, "detection", "reports", "cctv_test_25fps.mp4"
)

model_path = os.path.join(
    BASE_DIR, "models", "face_detection_yunet_2023mar.onnx"
)

SCORE_THRESHOLD = 0.70
NMS_THRESHOLD = 0.30
TOP_K = 5000

# Run face detection on every 2nd frame
DETECTION_INTERVAL = 2

# --------------------------------------------------
# CHECK FILES
# --------------------------------------------------

if not os.path.exists(video_path):
    print("[ERROR] Video not found:", video_path)
    raise SystemExit(1)

if not os.path.exists(model_path):
    print("[ERROR] YuNet model not found:", model_path)
    raise SystemExit(1)

if not hasattr(cv2, "FaceDetectorYN"):
    print("[ERROR] YuNet is not supported by OpenCV")
    raise SystemExit(1)

# --------------------------------------------------
# OPEN VIDEO
# --------------------------------------------------

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("[ERROR] Cannot open video")
    raise SystemExit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = cap.get(cv2.CAP_PROP_FPS)

print("[INFO] Resolution:", width, "x", height)
print("[INFO] Video FPS:", video_fps)

# --------------------------------------------------
# LOAD YUNET
# --------------------------------------------------

detector = cv2.FaceDetectorYN.create(
    model_path,
    "",
    (width, height),
    SCORE_THRESHOLD,
    NMS_THRESHOLD,
    TOP_K
)

print("[SUCCESS] YuNet loaded")
print("[INFO] Detection interval:", DETECTION_INTERVAL)
print("[INFO] Press Q to quit")

# --------------------------------------------------
# PROCESS VIDEO
# --------------------------------------------------

frame_number = 0
faces = None
previous_time = time.time()
display_fps = 0.0

try:
    while True:

        success, frame = cap.read()

        if not success:
            print("[INFO] Video processing completed")
            break

        frame_number += 1

        # Detect faces every 2 frames
        if frame_number % DETECTION_INTERVAL == 0 or faces is None:
            _, faces = detector.detect(frame)

        face_count = 0

        if faces is not None:

            face_count = len(faces)

            for face in faces:

                x, y, w, h = face[:4].astype(int)
                confidence = float(face[14])

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Face: {confidence:.2f}",
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # Measure display processing FPS
        current_time = time.time()
        elapsed = current_time - previous_time

        if elapsed > 0:
            display_fps = 1.0 / elapsed

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
            f"Processing FPS: {display_fps:.1f}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.imshow("YuNet - Optimized Cafeteria Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("[INFO] Detection stopped")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Resources released")
