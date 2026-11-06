"""
Workplace Seat Monitoring System - 3rd Floor Workstations
==========================================================
Monitors workstations, detects people and faces, recognizes enrolled employees,
and tracks seat occupancy & employee absences according to seat_monitoring logic.

Key Features:
- Multi-Model Pipeline:
    * YOLO11 (Person Detection)
    * YuNet (Fast Face Detection)
    * SFace (Deep Face Recognition)
- Seat Occupancy & Verification:
    * Detects whether seat is occupied via person detection (robust to posture/orientation)
    * Recognizes face against enrolled employee embeddings (KA019, KA036, etc.)
    * Verifies if the occupant matches the assigned seat employee
- Absence Tracking & Alerting:
    * Debounces momentary emptiness (EMPTY_CONFIRM_SECONDS)
    * Logs absence start and end to SQLite database (seat_tracking.db)
    * Alerts if absence exceeds threshold (ABSENCE_LIMIT_SECONDS)
    * Optional SMTP email notification to Reporting Manager
- Configurable Seat Zones:
    * Loads seat coordinates from monitoring/seats_3rd_floor.json
    * Interactive calibration tool available (calibrate_seats.py)
"""

import cv2
import time
import sqlite3
import os
import json
import smtplib
import numpy as np

from pathlib import Path
from datetime import datetime
from email.message import EmailMessage
from collections import defaultdict
from ultralytics import YOLO


# ============================================================
# 1. CONFIGURATION & PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Video Source:
# Priority 1: Converted test clip or full AVI
# Priority 2: Webcam (0) or RTSP stream url ("rtsp://...")
PRIMARY_VIDEO = BASE_DIR / "3rd Floor WS.avi"
FALLBACK_VIDEO = BASE_DIR / "detection" / "reports" / "3rd_floor_test.mp4"

if FALLBACK_VIDEO.exists():
    VIDEO_SOURCE = str(FALLBACK_VIDEO)
elif PRIMARY_VIDEO.exists():
    VIDEO_SOURCE = str(PRIMARY_VIDEO)
else:
    VIDEO_SOURCE = str(BASE_DIR / "detection" / "reports" / "cctv_test_25fps.mp4")

# Set True for recorded video files, False for live webcam/RTSP
IS_RECORDED_VIDEO = (
    isinstance(VIDEO_SOURCE, str)
    and os.path.exists(VIDEO_SOURCE)
    and not str(VIDEO_SOURCE).startswith("rtsp")
)

# Models & Database paths
YOLO_MODEL_PATH = str(BASE_DIR / "yolo11n.pt")
YUNET_MODEL_PATH = str(BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx")
SFACE_MODEL_PATH = str(BASE_DIR / "models" / "face_recognition_sface_2021dec.onnx")
EMBEDDINGS_PATH = str(BASE_DIR / "recognition" / "output" / "employee_embeddings.npz")
DATABASE_PATH = str(BASE_DIR / "seat_tracking.db")
SEATS_CONFIG_PATH = str(BASE_DIR / "monitoring" / "seats_3rd_floor.json")

# Timing & Detection Parameters
DETECTION_INTERVAL = 2       # Run neural inference every N frames for speed
PERSON_CONFIDENCE = 0.22     # YOLO person confidence threshold
FACE_CONFIDENCE = 0.35       # YuNet face confidence threshold
FACE_SIMILARITY_THRESH = 0.28 # SFace cosine similarity threshold
FACE_MARGIN_THRESH = 0.03    # Separation from 2nd best employee match

# Absence Tracking Parameters
ABSENCE_LIMIT_SECONDS = 10 * 60  # Alert after 10 minutes (set e.g. 30 for testing)
EMPTY_CONFIRM_SECONDS = 5        # Debounce time before confirming absence

# Email Alerts
ENABLE_EMAIL_ALERTS = False
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_APP_PASSWORD")


# ============================================================
# 2. SEAT CONFIGURATION LOADER
# ============================================================

DEFAULT_SEATS = {
    "WS-01": {
        "name": "Neha",
        "employee_id": "KA019",
        "email": "neha@kristellar.com",
        "rm_email": "manager@kristellar.com",
        "seat": [320, 160, 560, 420],
        "desc": "Left Workstation - Front"
    },
    "WS-02": {
        "name": "Rahul",
        "employee_id": "KA012",
        "email": "rahul@kristellar.com",
        "rm_email": "manager@kristellar.com",
        "seat": [450, 70, 600, 220],
        "desc": "Left Workstation - Back"
    },
    "WS-03": {
        "name": "Employee-03",
        "employee_id": "KA026",
        "email": "emp03@kristellar.com",
        "rm_email": "manager@kristellar.com",
        "seat": [300, 460, 580, 715],
        "desc": "Foreground Quad - Left Front"
    },
    "WS-04": {
        "name": "Employee-04",
        "employee_id": "KA027",
        "email": "emp04@kristellar.com",
        "rm_email": "manager@kristellar.com",
        "seat": [480, 370, 720, 530],
        "desc": "Foreground Quad - Left Back (Blue Jacket)"
    },
    "WS-05": {
        "name": "Employee-05",
        "employee_id": "KA032",
        "email": "emp05@kristellar.com",
        "rm_email": "manager@kristellar.com",
        "seat": [680, 480, 1020, 715],
        "desc": "Foreground Quad - Right Front (Laptop & Cushion)"
    },
    "WS-06": {
        "name": "Employee-06",
        "employee_id": "KA033",
        "email": "emp06@kristellar.com",
        "rm_email": "manager@kristellar.com",
        "seat": [750, 340, 1040, 500],
        "desc": "Foreground Quad - Right Back"
    },
    "WS-07": {
        "name": "Employee-07",
        "employee_id": "KA035",
        "email": "emp07@kristellar.com",
        "rm_email": "manager@kristellar.com",
        "seat": [610, 180, 750, 340],
        "desc": "Center Mid-Left Desk"
    },
    "WS-08": {
        "name": "Employee-08",
        "employee_id": "KA037",
        "email": "emp08@kristellar.com",
        "rm_email": "manager@kristellar.com",
        "seat": [660, 60, 770, 190],
        "desc": "Center Far-Back Left"
    },
    "WS-09": {
        "name": "Employee-09",
        "employee_id": "KA041",
        "email": "emp09@kristellar.com",
        "rm_email": "manager@kristellar.com",
        "seat": [730, 50, 810, 180],
        "desc": "Center Far-Back Mid"
    },
    "WS-10": {
        "name": "Employee-10",
        "employee_id": "KA042",
        "email": "emp10@kristellar.com",
        "rm_email": "manager@kristellar.com",
        "seat": [790, 40, 880, 170],
        "desc": "Center Far-Back Right"
    },
    "WS-11": {
        "name": "Employee-11",
        "employee_id": "KA044",
        "email": "emp11@kristellar.com",
        "rm_email": "manager@kristellar.com",
        "seat": [860, 160, 1020, 360],
        "desc": "Right Mid Desk - Near (Khaki shirt)"
    },
    "WS-12": {
        "name": "Employee-12",
        "employee_id": "KA036",
        "email": "emp12@kristellar.com",
        "rm_email": "manager@kristellar.com",
        "seat": [910, 90, 1050, 230],
        "desc": "Right Mid Desk - Far (Blue shirt)"
    },
    "WS-13": {
        "name": "Employee-13",
        "employee_id": "KA045",
        "email": "emp13@kristellar.com",
        "rm_email": "manager@kristellar.com",
        "seat": [1050, 140, 1180, 270],
        "desc": "Right Deep Desk"
    }
}

def load_seats_config():
    if os.path.exists(SEATS_CONFIG_PATH):
        try:
            with open(SEATS_CONFIG_PATH, "r", encoding="utf-8") as f:
                seats = json.load(f)
                print(f"[SUCCESS] Loaded {len(seats)} seats from {SEATS_CONFIG_PATH}")
                return seats
        except Exception as e:
            print(f"[WARNING] Could not read {SEATS_CONFIG_PATH} ({e}). Using defaults.")
    return DEFAULT_SEATS


# ============================================================
# 3. DATABASE OPERATIONS
# ============================================================

def create_database():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS absences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            seat_id TEXT NOT NULL,
            employee_name TEXT NOT NULL,
            absence_start TEXT NOT NULL,
            absence_end TEXT,
            duration_seconds INTEGER,
            email_sent INTEGER DEFAULT 0,
            source TEXT,
            verified_face TEXT
        )
    """)

    # Migrate columns if missing
    columns = {row[1] for row in conn.execute("PRAGMA table_info(absences)").fetchall()}
    if "source" not in columns:
        conn.execute("ALTER TABLE absences ADD COLUMN source TEXT")
    if "verified_face" not in columns:
        conn.execute("ALTER TABLE absences ADD COLUMN verified_face TEXT")

    conn.commit()
    conn.close()
    print("[SUCCESS] Database initialized:", DATABASE_PATH)


def start_absence(seat_id, employee_name, source, verified_face=None):
    now = datetime.now()
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.execute("""
        INSERT INTO absences
        (seat_id, employee_name, absence_start, source, verified_face)
        VALUES (?, ?, ?, ?, ?)
    """, (
        seat_id,
        employee_name,
        now.isoformat(),
        str(source),
        str(verified_face or "None")
    ))
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return {
        "record_id": record_id,
        "wall_start": now,
        "empty_since": time.monotonic(),
        "video_start": None,
        "email_sent": False,
        "email_attempted": False
    }


def close_absence(record_id, duration):
    now = datetime.now()
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("""
        UPDATE absences
        SET absence_end = ?,
            duration_seconds = ?
        WHERE id = ?
    """, (
        now.isoformat(),
        int(duration),
        record_id
    ))
    conn.commit()
    conn.close()


def mark_email_sent(record_id):
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("""
        UPDATE absences
        SET email_sent = 1
        WHERE id = ?
    """, (record_id,))
    conn.commit()
    conn.close()


# ============================================================
# 4. EMAIL NOTIFICATION
# ============================================================

def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"


def send_absence_email(employee_name, seat_id, rm_email, start_time, duration):
    if not ENABLE_EMAIL_ALERTS:
        print(f"[TEST MODE] Email alert simulated for {employee_name} ({seat_id})")
        return False

    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print("[WARNING] Email credentials not configured.")
        return False

    message = EmailMessage()
    message["Subject"] = f"Workplace Absence Alert: {employee_name} ({seat_id})"
    message["From"] = SENDER_EMAIL
    message["To"] = rm_email
    message.set_content(f"""
Hello,

The 3rd Floor Workstation Seat Monitoring System has detected
that an employee's assigned workstation has remained empty
for more than {format_duration(ABSENCE_LIMIT_SECONDS)}.

Employee Name: {employee_name}
Workstation Seat ID: {seat_id}
Absence Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
Elapsed Away Duration: {format_duration(duration)}

Please verify the employee's status.

Note: An empty seat does not independently confirm that the employee
has left the workplace. This is an automated notification.
""")

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=20) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(message)
        print(f"[EMAIL SENT] Absence notification sent to {rm_email}")
        return True
    except Exception as error:
        print("[EMAIL ERROR]", error)
        return False


# ============================================================
# 5. FACE RECOGNITION DATABASE LOADER
# ============================================================

def load_employee_templates(npz_path):
    if not os.path.exists(npz_path):
        print(f"[WARNING] Embeddings file not found: {npz_path}")
        return {}, None, []

    data = np.load(npz_path)
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
    if not known_ids:
        return {}, None, []

    template_matrix = np.vstack([employee_templates[eid] for eid in known_ids]).astype(np.float32)
    print(f"[SUCCESS] Loaded {len(known_ids)} employee identity templates.")
    return employee_templates, template_matrix, known_ids


# ============================================================
# 6. SEAT & OCCUPANCY MATCHING LOGIC
# ============================================================

def assign_people_and_faces_to_seats(people, face_records, seats):
    """
    Assigns detected people and faces to workstation seats uniquely.
    Each person occupies at most one seat (best overlap / containment).
    Faces are linked to the occupant person and seat.
    """
    seat_status = {}
    for sid, sdata in seats.items():
        seat_status[sid] = {
            "occupied": False,
            "person_box": None,
            "face": None,
            "status_str": "EMPTY",
            "verification": "EMPTY"
        }

    # Match each person to best seat
    for p in people:
        px1, py1, px2, py2 = p
        pcx, pcy = (px1 + px2) // 2, (py1 + py2) // 2
        p_area = max(1, (px2 - px1) * (py2 - py1))

        best_sid = None
        best_score = 0.0

        for sid, sdata in seats.items():
            sx1, sy1, sx2, sy2 = sdata["seat"]
            s_area = max(1, (sx2 - sx1) * (sy2 - sy1))

            center_inside = (sx1 <= pcx <= sx2 and sy1 <= pcy <= sy2)

            ix1 = max(px1, sx1)
            iy1 = max(py1, sy1)
            ix2 = min(px2, sx2)
            iy2 = min(py2, sy2)

            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                overlap_p = inter_area / p_area
                overlap_s = inter_area / s_area

                score = overlap_p + (0.5 if center_inside else 0.0)

                if (center_inside or overlap_p >= 0.20 or overlap_s >= 0.15) and score > best_score:
                    best_score = score
                    best_sid = sid

        if best_sid and not seat_status[best_sid]["occupied"]:
            seat_status[best_sid]["occupied"] = True
            seat_status[best_sid]["person_box"] = p

    # Match detected faces to seats
    for f in face_records:
        fx, fy, fw, fh = f["box"]
        fcx, fcy = fx + fw // 2, fy + fh // 2

        for sid, sdata in seats.items():
            if not seat_status[sid]["occupied"]:
                continue
            pbox = seat_status[sid]["person_box"]
            sx1, sy1, sx2, sy2 = sdata["seat"]

            # Face center inside person box or seat zone
            if (pbox and pbox[0] <= fcx <= pbox[2] and pbox[1] <= fcy <= pbox[3]) or \
               (sx1 <= fcx <= sx2 and sy1 <= fcy <= sy2):
                if seat_status[sid]["face"] is None or f["conf"] > seat_status[sid]["face"]["conf"]:
                    seat_status[sid]["face"] = f

    # Determine display status & verification
    for sid, sdata in seats.items():
        info = seat_status[sid]
        assigned_emp_id = sdata.get("employee_id")

        if info["occupied"]:
            info["status_str"] = "OCCUPIED"
            if info["face"]:
                face_id = info["face"]["id"]
                sim = info["face"]["sim"]
                if assigned_emp_id and face_id == assigned_emp_id:
                    info["verification"] = f"VERIFIED: {face_id}"
                elif face_id != "Unknown":
                    info["verification"] = f"DIFFERENT: {face_id} ({sim:.2f})"
                else:
                    info["verification"] = f"UNVERIFIED FACE ({sim:.2f})"
            else:
                info["verification"] = "PERSON (NO FACE)"
        else:
            info["status_str"] = "EMPTY"
            info["verification"] = "EMPTY"

    return seat_status


def get_absence_elapsed(absence, video_seconds):
    if IS_RECORDED_VIDEO:
        if absence["video_start"] is None:
            absence["video_start"] = video_seconds
        return max(0, int(video_seconds - absence["video_start"]))
    return max(0, int(time.monotonic() - absence["empty_since"]))


# ============================================================
# 7. MAIN MONITORING PROGRAM
# ============================================================

def main():
    print("=========================================================")
    print("      3RD FLOOR WORKSTATION SEAT MONITORING SYSTEM       ")
    print("=========================================================")

    create_database()

    # 1. Load Seats
    seats = load_seats_config()

    # 2. Load Models
    print("[INFO] Loading YOLO11 person detection model...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("[SUCCESS] YOLO11 loaded.")

    print("[INFO] Loading YuNet face detector and SFace recognizer...")
    detector = cv2.FaceDetectorYN.create(
        YUNET_MODEL_PATH,
        "",
        (1280, 720),
        FACE_CONFIDENCE,
        0.30,
        5000
    )
    recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL_PATH, "")
    print("[SUCCESS] Face models loaded.")

    # 3. Load Employee Database
    _, template_matrix, known_ids = load_employee_templates(EMBEDDINGS_PATH)

    # 4. Open Video Source
    print("[INFO] Opening Video Source:", VIDEO_SOURCE)
    camera = cv2.VideoCapture(VIDEO_SOURCE)

    if not camera.isOpened():
        print(f"[ERROR] Could not open video source: {VIDEO_SOURCE}")
        return

    src_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = camera.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or video_fps > 100:
        video_fps = 25.0

    print(f"[INFO] Video Resolution: {src_w} x {src_h} @ {video_fps:.1f} FPS")
    print(f"[INFO] Monitored Workstations: {len(seats)}")
    print(f"[INFO] Absence Threshold: {format_duration(ABSENCE_LIMIT_SECONDS)}")
    print("[INFO] Press 'Q' to quit | 'P' to pause | 'S' screenshot | 'C' toggle seats")

    # State tracking
    active_absences = {}
    empty_candidates = {}

    cached_people = []
    cached_faces = []
    cached_seat_status = {}

    frame_number = 0
    show_seats_overlay = True
    show_detections_overlay = True
    paused = False
    prev_time = time.time()
    display_fps = 0.0

    cv2.namedWindow("3rd Floor Workstation - Seat Monitor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("3rd Floor Workstation - Seat Monitor", 1280, 720)

    try:
        while True:
            if not paused:
                success, raw_frame = camera.read()
                if not success:
                    print("[INFO] Video ended or stream finished.")
                    break

                frame_number += 1
                frame = cv2.resize(raw_frame, (1280, 720))

                # Video timeline
                video_seconds = camera.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if video_seconds <= 0:
                    video_seconds = frame_number / video_fps

                # Multi-Model Detection Pipeline
                if frame_number == 1 or frame_number % DETECTION_INTERVAL == 0:
                    # 1. Person Detection (YOLO)
                    yolo_results = yolo_model.predict(
                        source=frame,
                        classes=[0],
                        conf=PERSON_CONFIDENCE,
                        device="cpu",
                        verbose=False
                    )

                    cached_people = []
                    for res in yolo_results:
                        if res.boxes is None:
                            continue
                        for box in res.boxes:
                            cached_people.append(list(map(int, box.xyxy[0].tolist())))

                    # 2. Face Detection (YuNet)
                    detector.setInputSize((1280, 720))
                    _, faces = detector.detect(frame)

                    cached_faces = []
                    if faces is not None and template_matrix is not None and len(known_ids) > 0:
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
                                    s_idx = np.argsort(sims)[::-1]
                                    best_idx = s_idx[0]
                                    best_score = float(sims[best_idx])
                                    second_score = float(sims[s_idx[1]]) if len(s_idx) > 1 else -1.0
                                    margin = best_score - second_score

                                    if best_score >= FACE_SIMILARITY_THRESH and margin >= FACE_MARGIN_THRESH:
                                        matched_id = known_ids[best_idx]
                                    sim = best_score
                            except Exception:
                                pass

                            cached_faces.append({
                                "box": fb,
                                "conf": conf,
                                "id": matched_id,
                                "sim": sim
                            })

                    # 3. Match People & Faces to Workstation Seats
                    cached_seat_status = assign_people_and_faces_to_seats(
                        cached_people,
                        cached_faces,
                        seats
                    )

                # ========================================================
                # SEAT ABSENCE TRACKING LOGIC
                # ========================================================
                for seat_id, sdata in seats.items():
                    employee_name = sdata["name"]
                    sinfo = cached_seat_status.get(seat_id, {"occupied": False})
                    occupied = sinfo["occupied"]

                    if occupied:
                        empty_candidates.pop(seat_id, None)

                        # If returning from absence, close absence in DB
                        if seat_id in active_absences:
                            absence = active_absences[seat_id]
                            duration = get_absence_elapsed(absence, video_seconds)
                            close_absence(absence["record_id"], duration)

                            print(
                                f"[RETURNED] {employee_name} returned to seat {seat_id}. "
                                f"Away duration: {format_duration(duration)}"
                            )
                            del active_absences[seat_id]

                    else:
                        # Seat is currently empty -> check debouncing
                        if seat_id not in active_absences:
                            if seat_id not in empty_candidates:
                                empty_candidates[seat_id] = {
                                    "wall_start": time.monotonic(),
                                    "video_start": video_seconds
                                }

                            cand = empty_candidates[seat_id]
                            if IS_RECORDED_VIDEO:
                                empty_dur = video_seconds - cand["video_start"]
                            else:
                                empty_dur = time.monotonic() - cand["wall_start"]

                            # Debounce confirmed
                            if empty_dur >= EMPTY_CONFIRM_SECONDS:
                                active_absences[seat_id] = start_absence(
                                    seat_id,
                                    employee_name,
                                    VIDEO_SOURCE
                                )
                                if IS_RECORDED_VIDEO:
                                    active_absences[seat_id]["video_start"] = cand["video_start"]
                                else:
                                    active_absences[seat_id]["empty_since"] = cand["wall_start"]

                                print(f"[ABSENCE STARTED] {employee_name} ({seat_id}) is away.")
                                empty_candidates.pop(seat_id, None)

                        # Check elapsed absence against limit
                        if seat_id in active_absences:
                            absence = active_absences[seat_id]
                            elapsed = get_absence_elapsed(absence, video_seconds)

                            if elapsed >= ABSENCE_LIMIT_SECONDS and not absence["email_attempted"]:
                                absence["email_attempted"] = True
                                print(
                                    f"[ALERT] {employee_name} ({seat_id}) away for "
                                    f"{format_duration(elapsed)} (exceeded threshold)."
                                )
                                if not IS_RECORDED_VIDEO:
                                    sent = send_absence_email(
                                        employee_name,
                                        seat_id,
                                        sdata["rm_email"],
                                        absence["wall_start"],
                                        elapsed
                                    )
                                    if sent:
                                        mark_email_sent(absence["record_id"])
                                else:
                                    print("[TEST MODE] Absence limit reached. Real email disabled for video test.")

            # ========================================================
            # VISUAL HUD RENDERING
            # ========================================================
            display_frame = frame.copy()

            # 1. Draw detected people
            if show_detections_overlay:
                for p in cached_people:
                    px1, py1, px2, py2 = p
                    cv2.rectangle(display_frame, (px1, py1), (px2, py2), (80, 180, 80), 1)

                for f in cached_faces:
                    fx, fy, fw, fh = f["box"]
                    f_col = (0, 255, 0) if f["id"] != "Unknown" else (0, 180, 255)
                    cv2.rectangle(display_frame, (fx, fy), (fx + fw, fy + fh), f_col, 2)
                    flabel = f"{f['id']} ({f['sim']:.2f})" if f["id"] != "Unknown" else "Face"
                    cv2.putText(
                        display_frame, flabel, (fx, max(15, fy - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, f_col, 1
                    )

            # 2. Draw workstation seats
            occupied_count = 0
            empty_count = 0
            alert_count = 0

            if show_seats_overlay:
                for seat_id, sdata in seats.items():
                    sx1, sy1, sx2, sy2 = sdata["seat"]
                    emp_name = sdata["name"]
                    sinfo = cached_seat_status.get(seat_id, {"occupied": False, "verification": "EMPTY"})
                    is_occ = sinfo["occupied"]

                    if is_occ:
                        occupied_count += 1
                        if "VERIFIED" in sinfo["verification"]:
                            color = (0, 220, 0)       # Green: Verified employee
                            status_txt = f"{seat_id}: {emp_name} [VERIFIED]"
                        elif "DIFFERENT" in sinfo["verification"]:
                            color = (255, 180, 0)     # Cyan/Orange: Different employee
                            status_txt = f"{seat_id}: {emp_name} [{sinfo['verification']}]"
                        else:
                            color = (0, 200, 100)     # Soft Green: Person present
                            status_txt = f"{seat_id}: {emp_name} [OCCUPIED]"
                    else:
                        empty_count += 1
                        is_away_alert = False
                        away_text = ""

                        if seat_id in active_absences:
                            elapsed = get_absence_elapsed(active_absences[seat_id], video_seconds)
                            away_text = f" | Away: {format_duration(elapsed)}"
                            if elapsed >= ABSENCE_LIMIT_SECONDS:
                                is_away_alert = True
                                alert_count += 1

                        if is_away_alert:
                            color = (0, 0, 255)       # Bright Red: Limit exceeded
                            status_txt = f"{seat_id}: {emp_name} [ALERT{away_text}]"
                        else:
                            color = (60, 60, 230)     # Red: Empty
                            status_txt = f"{seat_id}: {emp_name} [EMPTY{away_text}]"

                    # Draw seat rectangle
                    thickness = 3 if is_occ else 2
                    cv2.rectangle(display_frame, (sx1, sy1), (sx2, sy2), color, thickness)

                    # Badge header
                    (tw, th), _ = cv2.getTextSize(status_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    badge_y = max(18, sy1)
                    cv2.rectangle(
                        display_frame,
                        (sx1, badge_y - th - 6),
                        (sx1 + tw + 6, badge_y),
                        (20, 20, 20),
                        -1
                    )
                    cv2.putText(
                        display_frame,
                        status_txt,
                        (sx1 + 3, badge_y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1
                    )

            # 3. Top HUD Status Bar
            now_t = time.time()
            if now_t - prev_time > 0:
                display_fps = 1.0 / (now_t - prev_time)
            prev_time = now_t

            # Banner background
            cv2.rectangle(display_frame, (0, 0), (1280, 36), (20, 20, 20), -1)
            cv2.line(display_frame, (0, 36), (1280, 36), (60, 60, 60), 1)

            banner_left = (
                f"3rd Floor Workstation Area  |  Seats: {len(seats)}  "
                f"[Occupied: {occupied_count} | Empty: {empty_count}]  "
                f"Alerts: {alert_count}"
            )
            cv2.putText(
                display_frame,
                banner_left,
                (15, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255) if alert_count == 0 else (0, 100, 255),
                2
            )

            banner_right = (
                f"People: {len(cached_people)} | Faces: {len(cached_faces)} | "
                f"FPS: {display_fps:.1f} | Frame: {frame_number}"
            )
            cv2.putText(
                display_frame,
                banner_right,
                (780, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (220, 220, 220),
                1
            )

            if paused:
                cv2.putText(
                    display_frame,
                    "[PAUSED - Press 'P' to resume]",
                    (520, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

            cv2.imshow("3rd Floor Workstation - Seat Monitor", display_frame)

            # Key controls
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), ord("Q"), 27]:
                print("[INFO] Quitting monitoring...")
                break
            elif key in [ord("p"), ord("P")]:
                paused = not paused
                print(f"[INFO] Monitoring {'paused' if paused else 'resumed'}.")
            elif key in [ord("c"), ord("C")]:
                show_seats_overlay = not show_seats_overlay
            elif key in [ord("d"), ord("D")]:
                show_detections_overlay = not show_detections_overlay
            elif key in [ord("s"), ord("S")]:
                snap_path = BASE_DIR / "monitoring" / f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(str(snap_path), display_frame)
                print(f"[INFO] Snapshot saved to: {snap_path}")

    except KeyboardInterrupt:
        print("[INFO] Monitoring stopped by user.")
    except Exception as error:
        print("[ERROR] Runtime failure:", error)
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera released. Resources cleared.")
        print("[INFO] Seat monitoring session ended.")


if __name__ == "__main__":
    main()
