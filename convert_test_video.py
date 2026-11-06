
import cv2
import os

input_path = (
    r"C:\Users\Kristellar\Downloads\Face-detection-main"
    r"\Face-detection-main\[CH001] 20261102063001-20261102065948.avi"
)

output_path = os.path.join("reports", "cctv_test_10sec.mp4")

os.makedirs("reports", exist_ok=True)

cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    print("[ERROR] Cannot open original CCTV video")
    raise SystemExit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("[INFO] Original video details")
print("FPS:", fps)
print("Resolution:", width, "x", height)

if fps <= 0:
    fps = 25.0
    print("[WARNING] Invalid FPS. Using 25 FPS for test.")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    print("[ERROR] Could not create MP4 file")
    cap.release()
    raise SystemExit(1)

max_frames = int(fps * 10)
frames_written = 0

print("[INFO] Converting first 10 seconds...")

while frames_written < max_frames:
    success, frame = cap.read()

    if not success:
        break

    writer.write(frame)
    frames_written += 1

cap.release()
writer.release()

print("[SUCCESS] Conversion finished")
print("Frames written:", frames_written)
print("Output:", os.path.abspath(output_path))
