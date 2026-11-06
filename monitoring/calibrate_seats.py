"""
Seat Calibration Tool for Workplace Monitoring
-----------------------------------------------
Use this tool to visually configure seat zones on the camera view.
- Left-click and drag: Draw a new seat box
- Right-click inside a seat: Select seat
- Press 'A' or 'N': Add/rename selected seat (prompts in terminal for seat_id, employee name, employee_id)
- Press 'D': Delete selected seat
- Press 'S': Save configuration to monitoring/seats_3rd_floor.json
- Press 'R': Reload configuration from disk
- Press 'Q' or ESC: Quit calibration
"""

import cv2
import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "monitoring" / "seats_3rd_floor.json"
DEFAULT_VIDEO = BASE_DIR / "3rd Floor WS.avi"

# Load initial image from video or file
def get_sample_frame():
    # Check for converted mp4 first
    for test_path in [
        BASE_DIR / "3rd Floor WS.avi",
        BASE_DIR / "test_3rd_floor_f0.jpg",
        BASE_DIR / "detection" / "reports" / "cctv_test_25fps.mp4"
    ]:
        if test_path.exists():
            if test_path.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                img = cv2.imread(str(test_path))
                if img is not None:
                    return cv2.resize(img, (1280, 720))
            else:
                cap = cv2.VideoCapture(str(test_path))
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        return cv2.resize(frame, (1280, 720))
    # Fallback blank
    import numpy as np
    return np.zeros((720, 1280, 3), dtype=np.uint8)

def load_seats():
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print("[ERROR] Failed to load config:", e)
    return {}

def save_seats(seats):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(seats, f, indent=4)
    print(f"[SUCCESS] Saved {len(seats)} seats to {CONFIG_PATH}")

class SeatCalibrator:
    def __init__(self):
        self.frame = get_sample_frame()
        self.seats = load_seats()
        self.drawing = False
        self.start_pt = None
        self.current_pt = None
        self.selected_sid = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_pt = (x, y)
            self.current_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            x1, y1 = self.start_pt
            x2, y2 = x, y
            box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            if (box[2] - box[0] > 20) and (box[3] - box[1] > 20):
                self.prompt_new_seat(box)
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Select seat under cursor
            self.selected_sid = None
            for sid, sdata in self.seats.items():
                sx1, sy1, sx2, sy2 = sdata["seat"]
                if sx1 <= x <= sx2 and sy1 <= y <= sy2:
                    self.selected_sid = sid
                    print(f"[SELECTED] Seat {sid}: {sdata.get('name')}")
                    break

    def prompt_new_seat(self, box):
        print("\n--- ADD NEW SEAT ---")
        default_id = f"WS-{len(self.seats) + 1:02d}"
        sid = input(f"Enter Seat ID [{default_id}]: ").strip() or default_id
        name = input("Enter Employee Name: ").strip() or "Employee"
        emp_id = input("Enter Enrolled Employee ID (e.g. KA019) [Optional]: ").strip()
        email = input("Enter Employee Email [Optional]: ").strip()
        rm_email = input("Enter Manager Email [Optional]: ").strip()

        self.seats[sid] = {
            "name": name,
            "employee_id": emp_id,
            "email": email or f"{name.lower().replace(' ', '')}@kristellar.com",
            "rm_email": rm_email or "manager@kristellar.com",
            "seat": list(box),
            "desc": f"Workstation {sid}"
        }
        self.selected_sid = sid
        print(f"[ADDED] {sid} at {box}")

    def run(self):
        cv2.namedWindow("Workstation Seat Calibrator", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Workstation Seat Calibrator", 1280, 720)
        cv2.setMouseCallback("Workstation Seat Calibrator", self.mouse_callback)

        print("\n================================================")
        print("         WORKSTATION SEAT CALIBRATION TOOL      ")
        print("================================================")
        print("Click & drag to draw a new seat box.")
        print("Right-click inside a box to select it.")
        print("Press 'S' to Save | 'D' to Delete selected | 'R' to Reload")
        print("Press 'Q' or ESC to Quit.")
        print("================================================\n")

        while True:
            display = self.frame.copy()

            # Draw existing seats
            for sid, sdata in self.seats.items():
                x1, y1, x2, y2 = sdata["seat"]
                is_sel = (sid == self.selected_sid)
                color = (0, 255, 255) if is_sel else (0, 200, 0)
                thickness = 3 if is_sel else 2

                cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
                label = f"{sid}: {sdata.get('name', '')}"
                if sdata.get("employee_id"):
                    label += f" ({sdata['employee_id']})"

                cv2.putText(
                    display, label, (x1 + 3, max(20, y1 + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
                )

            # Draw current dragging box
            if self.drawing and self.start_pt and self.current_pt:
                x1 = min(self.start_pt[0], self.current_pt[0])
                y1 = min(self.start_pt[1], self.current_pt[1])
                x2 = max(self.start_pt[0], self.current_pt[0])
                y2 = max(self.start_pt[1], self.current_pt[1])
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 165, 255), 2)

            # Top instructions bar
            cv2.rectangle(display, (0, 0), (1280, 30), (25, 25, 25), -1)
            inst = f"Seats: {len(self.seats)} | Selected: {self.selected_sid or 'None'} | Drag: New Seat | S: Save | D: Delete | R: Reload | Q: Quit"
            cv2.putText(display, inst, (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Workstation Seat Calibrator", display)
            key = cv2.waitKey(20) & 0xFF

            if key in [ord("q"), ord("Q"), 27]:
                break
            elif key in [ord("s"), ord("S")]:
                save_seats(self.seats)
            elif key in [ord("d"), ord("D")]:
                if self.selected_sid and self.selected_sid in self.seats:
                    print(f"[DELETED] Removed {self.selected_sid}")
                    del self.seats[self.selected_sid]
                    self.selected_sid = None
            elif key in [ord("r"), ord("R")]:
                self.seats = load_seats()
                print(f"[RELOADED] Loaded {len(self.seats)} seats")

        cv2.destroyAllWindows()

if __name__ == "__main__":
    calibrator = SeatCalibrator()
    calibrator.run()

