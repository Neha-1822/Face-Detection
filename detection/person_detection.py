
import cv2
import time
from ultralytics import YOLO


class PersonDetector:

    def __init__(self, model_path="yolo11n.pt"):

        print("[INFO] Loading YOLO model...")

        self.model = YOLO(model_path)

        print("[SUCCESS] YOLO model loaded")

    def detect(self, frame):

        results = self.model.predict(
        source=frame,
        classes=[0],
        conf=0.25,
        imgsz=960,
        device="cpu",
        verbose=False
    )

        annotated_frame = results[0].plot()

        person_count = len(results[0].boxes)

        return annotated_frame, person_count


def main():

    # video_path = (
    #     r"C:\Users\Kristellar\Downloads\Face-detection-main"
    #     r"\Face-detection-main\[CH001] 20261102063001-20261102065948.avi"
    # )
    video_path = r"C:\Users\Kristellar\Downloads\Workplace_Monitoring\4Th Floor WS.avi"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("[ERROR] Cannot open CCTV video")
        return

    detector = PersonDetector()

    print("[INFO] Starting person detection")

    previous_time = time.time()

    try:

        while True:

            success, frame = cap.read()

            if not success:
                print("[INFO] Video completed")
                break

            # Resize frame to reduce CPU workload
            frame = cv2.resize(frame, (1280, 720))

            output, count = detector.detect(frame)

            current_time = time.time()

            fps = 1 / max(current_time - previous_time, 0.001)
            previous_time = current_time

            cv2.putText(
                output,
                f"Persons: {count}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            cv2.putText(
                output,
                f"FPS: {fps:.1f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.imshow("Workplace Person Detection", output)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("[INFO] Detection stopped")

    finally:

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
