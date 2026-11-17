"""
Workplace Cafeteria & Lounge Monitor
------------------------------------
Monitors employee presence in the cafeteria / lounge area:
1. Recognizes employees biometrically using YuNet face detection & SFace recognition.
2. Identifies employee's assigned workstation and floor (3rd or 4th Floor).
3. Verifies authorization against floor-specific tea break schedules:
   - 3rd Floor Break: 11:15 AM - 11:30 AM & 04:15 PM - 04:30 PM
   - 4th Floor Break: 11:00 AM - 11:15 AM & 04:00 PM - 04:15 PM
4. If an employee is detected in the cafeteria after their break has ended:
   - Flags person as 'POST_BREAK_OVERSTAY'
   - Automatically saves snapshot evidence
   - Sends an instant email notification to their Reporting Manager (RM)
"""

import argparse
from collections import defaultdict
import cv2
from datetime import datetime
from email.message import EmailMessage
import json
import numpy as np
import os
from pathlib import Path
import smtplib
import sys
import time

from ultralytics import YOLO

# Add parent directory for absolute path resolution
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "monitoring"))

from schedule_logic import (
    get_schedule_status,
    get_post_break_status,
    get_employee_cafeteria_status,
    get_floor_breaks,
    normalize_floor
)

# Model & Data Paths
YOLO_MODEL_PATH = str(BASE_DIR / "yolo11n.pt")
YUNET_MODEL_PATH = str(BASE_DIR / "recognition" / "models" / "face_detection_yunet_2023mar.onnx")
SFACE_MODEL_PATH = str(BASE_DIR / "recognition" / "models" / "face_recognition_sface_2021dec.onnx")
EMBEDDINGS_PATH = str(BASE_DIR / "recognition" / "output" / "employee_embeddings.npz")
DATABASE_PATH = str(BASE_DIR / "seat_tracking.db")

# Detection Thresholds
PERSON_CONFIDENCE = 0.45
FACE_CONFIDENCE = 0.70
RECOGNITION_SIMILARITY_THRESHOLD = 0.40
DETECTION_INTERVAL = 3

# Email Alerts Configuration
ENABLE_EMAIL_ALERTS = os.getenv("ENABLE_EMAIL_ALERTS", "true").lower() in ("true", "1", "yes")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "kristellardc@gmail.com")
SENDER_PASSWORD = os.getenv("SENDER_APP_PASSWORD", "").replace(" ", "")
DEFAULT_RM_EMAIL = os.getenv("DEFAULT_RM_EMAIL", "neha@kristellar.com")
ENFORCE_SCHEDULE = os.getenv("ENFORCE_SCHEDULE", "true").lower() in ("true", "1", "yes")


def build_employee_floor_registry():
    """
    Scans all workstation seat configurations to build an employee registry:
    emp_id -> { "name": ..., "floor": "3" or "4", "seat_id": ..., "rm_email": ..., "desks": [...] }
    """
    registry = {}
    config_mappings = [
        ("monitoring/seats_3rd_floor.json", "3"),
        ("monitoring/seats_3rd_floor_area2.json", "3"),
        ("monitoring/seats_4th_floor.json", "4"),
    ]

    for rel_path, fl in config_mappings:
        full_path = BASE_DIR / rel_path
        if full_path.exists():
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for sid, sinfo in data.items():
                        eid = sinfo.get("employee_id")
                        if eid:
                            eid_upper = str(eid).upper()
                            if eid_upper not in registry:
                                registry[eid_upper] = {
                                    "employee_id": eid_upper,
                                    "name": sinfo.get("name", "Employee"),
                                    "floor": fl,
                                    "seat_id": sid,
                                    "rm_email": sinfo.get("rm_email") or DEFAULT_RM_EMAIL,
                                    "desks": [(fl, sid)]
                                }
                            else:
                                registry[eid_upper]["desks"].append((fl, sid))
            except Exception as e:
                print(f"[WARNING] Could not read registry from {rel_path}: {e}")

    return registry


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


def is_valid_face(face, min_conf=0.70):
    if face is None or len(face) < 15:
        return False
    conf = float(face[14])
    if conf < min_conf:
        return False
    fw, fh = int(face[2]), int(face[3])
    if fw < 24 or fh < 24 or fw > 360 or fh > 360:
        return False
    aspect = fw / float(fh) if fh > 0 else 0
    return 0.55 <= aspect <= 1.55


def send_cafeteria_overstay_email(employee_info, break_name, break_end_time, snapshot_path=None, area_name="Cafeteria & Lounge"):
    """
    Sends an automated email notification to the Reporting Manager (RM)
    when an employee is detected in the cafeteria after their break has ended.
    """
    emp_name = employee_info["name"]
    emp_id = employee_info["employee_id"]
    floor = employee_info["floor"]
    seat_id = employee_info["seat_id"]
    rm_email = employee_info["rm_email"] or DEFAULT_RM_EMAIL

    if not ENABLE_EMAIL_ALERTS:
        print(f"[INFO] Email alerts disabled. Cafeteria alert for {emp_name} ({emp_id}) skipped.")
        return False

    now_str = datetime.now().strftime("%Y-%m-%d %I:%M %p")

    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print("\n" + "=" * 65)
        print(" [EMAIL ALERT TRIGGERED - CAFETERIA OVERSTAY SIMULATION]")
        print(f" Area                   : {area_name}")
        print(f" To (Reporting Manager) : {rm_email}")
        print(f" Subject                : [Workplace Alert - In Cafeteria After Break] {emp_name} ({emp_id}) seen in Cafeteria after {break_name}")
        print(f" Employee               : {emp_name} (ID: {emp_id}, Floor: {floor}th Floor, Desk: {seat_id})")
        print(f" Scheduled Break        : {break_name} (Ended at {break_end_time})")
        print(f" Sighted In Cafeteria   : {now_str}")
        if snapshot_path:
            print(f" Snapshot Evidence      : {snapshot_path}")
        print("=" * 65 + "\n")
        return False

    message = EmailMessage()
    message["Subject"] = f"[Workplace Alert - In Cafeteria After Break] {emp_name} ({emp_id}) seen in Cafeteria after {break_name}"
    message["From"] = SENDER_EMAIL
    message["To"] = rm_email

    plain_content = f"""Workplace Monitoring Alert - Employee In Cafeteria After Break
=================================================================
Dear Reporting Manager,

This is an automated notification from the Workplace Monitoring System ({area_name}).

The employee below has been observed in the Cafeteria after their floor's scheduled tea break ended:

  • Employee Name       : {emp_name}
  • Employee ID         : {emp_id}
  • Assigned Floor      : {floor}th Floor
  • Workstation Desk    : {seat_id}
  • Reporting Manager   : {rm_email}
  • Scheduled Break     : {break_name}
  • Break Ended At      : {break_end_time}
  • Observed At         : {now_str}

Status: The employee has not returned to their workstation and is currently detected in the cafeteria after their break ended.
Please check in with the employee to ensure normal workflow.

--
Automated Workplace Monitoring System
Kristellar Aerospace
"""
    message.set_content(plain_content)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f6f8fa; margin: 0; padding: 24px; }}
    .container {{ max-width: 580px; margin: auto; background: #ffffff; border-radius: 8px; border: 1px solid #e1e4e8; overflow: hidden; box-shadow: 0 2px 6px rgba(0,0,0,0.06); }}
    .header {{ background-color: #cf222e; color: #ffffff; padding: 18px 24px; }}
    .header h2 {{ margin: 0; font-size: 19px; }}
    .body {{ padding: 24px; }}
    .alert-banner {{ background-color: #ffebe9; border-left: 4px solid #cf222e; border-radius: 4px; padding: 12px 16px; margin-bottom: 20px; color: #cf222e; font-weight: 600; font-size: 14px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 14px; }}
    td {{ padding: 9px 12px; border-bottom: 1px solid #eaecef; font-size: 14px; }}
    td.label {{ font-weight: 600; color: #57606a; width: 160px; }}
    td.val {{ color: #24292f; }}
    .badge {{ background-color: #ffebe9; color: #cf222e; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: bold; border: 1px solid #ff8182; }}
    .footer {{ background-color: #f6f8fa; padding: 14px 24px; font-size: 12px; color: #6e7781; border-top: 1px solid #e1e4e8; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h2>Workplace Alert: Employee in Cafeteria After Break</h2>
    </div>
    <div class="body">
      <div class="alert-banner">
        ⚠️ Notice: Employee detected in cafeteria after their scheduled {break_name} ended at {break_end_time}.
      </div>
      <p style="font-size: 14px; color: #24292f; margin-top: 0;">
        Dear Reporting Manager,<br><br>
        The Workplace Monitoring System has detected that the employee below is in the <strong>{area_name}</strong> after their floor's authorized break time:
      </p>
      <table>
        <tr><td class="label">Employee Name</td><td class="val"><strong>{emp_name}</strong></td></tr>
        <tr><td class="label">Employee ID</td><td class="val"><strong>{emp_id}</strong></td></tr>
        <tr><td class="label">Assigned Floor</td><td class="val">{floor}th Floor</td></tr>
        <tr><td class="label">Assigned Desk</td><td class="val">{seat_id}</td></tr>
        <tr><td class="label">Reporting Manager</td><td class="val">{rm_email}</td></tr>
        <tr><td class="label">Scheduled Break</td><td class="val">{break_name}</td></tr>
        <tr><td class="label">Break Ended At</td><td class="val"><strong>{break_end_time}</strong></td></tr>
        <tr><td class="label">Status</td><td class="val"><span class="badge">IN CAFETERIA AFTER BREAK</span></td></tr>
      </table>
      <p style="font-size: 13px; color: #57606a; margin-top: 20px;">
        Snapshot evidence has been captured and archived by the monitoring system.
      </p>
    </div>
    <div class="footer">
      Automated Workplace Monitoring System &bull; Kristellar Aerospace
    </div>
  </div>
</body>
</html>
"""
    message.add_alternative(html_content, subtype="html")

    # Attach snapshot evidence if available
    if snapshot_path and os.path.exists(snapshot_path):
        try:
            with open(snapshot_path, "rb") as img_f:
                message.add_attachment(
                    img_f.read(),
                    maintype="image",
                    subtype="jpeg",
                    filename=os.path.basename(snapshot_path)
                )
        except Exception as e:
            print(f"[WARNING] Could not attach snapshot {snapshot_path}: {e}")

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=20) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(message)
        print(f"[EMAIL SENT] Cafeteria overstay alert sent to RM ({rm_email}) for {emp_name} ({emp_id}).")
        return True
    except Exception as error:
        print(f"[EMAIL ERROR] Failed to send email to {rm_email}: {error}")
        return False


def main():
    print("=========================================================")
    print("        WORKPLACE CAFETERIA & LOUNGE MONITOR             ")
    print("=========================================================")

    global ENABLE_EMAIL_ALERTS, ENFORCE_SCHEDULE

    parser = argparse.ArgumentParser(description="Workplace Cafeteria & Lounge Monitor")
    parser.add_argument(
        "--source",
        type=str,
        default=os.getenv("CAFETERIA_VIDEO_SOURCE", "0"),
        help="Video file path, RTSP stream URL, or webcam index (0)"
    )
    parser.add_argument(
        "--area-name",
        type=str,
        default="Kristellar Cafeteria & Lounge",
        help="Display title for the monitored cafeteria/lounge"
    )
    parser.add_argument(
        "--send-email",
        action="store_true",
        default=False,
        help="Force enable live email alerts to RM"
    )
    parser.add_argument(
        "--ignore-schedule",
        action="store_true",
        default=False,
        help="Bypass office hours schedule restrictions (for testing)"
    )
    parser.add_argument(
        "--post-break-grace",
        type=int,
        default=60,
        help="Grace period (in seconds) after break ends before sending alert (default: 60s)"
    )

    args = parser.parse_args()

    if args.send_email:
        ENABLE_EMAIL_ALERTS = True
    if args.ignore_schedule:
        ENFORCE_SCHEDULE = False

    area_title = args.area_name
    window_title = f"{area_title} - Monitor"
    grace_seconds = args.post_break_grace

    # 1. Load Models & Embeddings
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

    _, template_matrix, known_ids = load_employee_templates(EMBEDDINGS_PATH)
    employee_registry = build_employee_floor_registry()
    print(f"[SUCCESS] Employee registry mapped {len(employee_registry)} enrolled workstations across 3rd & 4th floors.")

    # 2. Open Video Source
    video_src = args.source
    if str(video_src).isdigit():
        video_src = int(video_src)
        is_recorded = False
    elif str(video_src).lower().startswith(("rtsp://", "http://", "https://")):
        is_recorded = False
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    else:
        is_recorded = os.path.exists(video_src)

    print("[INFO] Opening Video Source:", video_src)
    camera = cv2.VideoCapture(video_src)
    if not is_recorded:
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not camera.isOpened():
        print(f"[ERROR] Could not open video source: {video_src}")
        return

    ALERTS_DIR = BASE_DIR / "monitoring" / "cafeteria_alerts"
    os.makedirs(ALERTS_DIR, exist_ok=True)

    # State tracking
    alerted_overstays = set()  # (emp_id, break_tag)
    seen_overstay_start = {}   # emp_id -> timestamp first seen overstaying
    cached_people = []
    cached_faces = []

    frame_number = 0
    prev_time = time.time()
    display_fps = 25.0
    paused = False

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, 1280, 720)

    print("\n[INFO] Monitoring active. Press 'Q' to quit | 'P' to pause | 'S' screenshot.")

    try:
        while True:
            if not paused:
                ret, raw_frame = camera.read()
                if not ret:
                    if is_recorded:
                        camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        print("[WARNING] Camera stream interrupted. Attempting reconnection...")
                        time.sleep(1.0)
                        camera.release()
                        camera = cv2.VideoCapture(video_src)
                        ret, raw_frame = camera.read()
                        if not ret:
                            time.sleep(0.5)
                            continue

                frame_number += 1
                frame = cv2.resize(raw_frame, (1280, 720))

                # Multi-Model Detection Pipeline
                if frame_number == 1 or frame_number % DETECTION_INTERVAL == 0:
                    # 1. Person Detection
                    yolo_results = yolo_model.predict(
                        source=frame,
                        classes=[0],
                        conf=PERSON_CONFIDENCE,
                        imgsz=640,
                        device="cpu",
                        verbose=False
                    )
                    cached_people = []
                    for res in yolo_results:
                        if res.boxes is not None:
                            for box in res.boxes:
                                cached_people.append(list(map(int, box.xyxy[0].tolist())))

                    # 2. Face Detection & Recognition
                    detector.setInputSize((1280, 720))
                    _, faces = detector.detect(frame)
                    cached_faces = []

                    if faces is not None and template_matrix is not None and len(known_ids) > 0:
                        for face in faces:
                            if not is_valid_face(face, min_conf=FACE_CONFIDENCE):
                                continue
                            fb = face[:4].astype(int)
                            matched_id = "Unknown"
                            sim = 0.0

                            try:
                                aligned = recognizer.alignCrop(frame, face)
                                feat = recognizer.feature(aligned).flatten().astype(np.float32)
                                norm = np.linalg.norm(feat)
                                if norm > 0:
                                    feat = feat / norm
                                    scores = np.dot(template_matrix, feat)
                                    best_idx = int(np.argmax(scores))
                                    if scores[best_idx] >= RECOGNITION_SIMILARITY_THRESHOLD:
                                        matched_id = known_ids[best_idx]
                                        sim = float(scores[best_idx])
                            except Exception:
                                pass

                            cached_faces.append({
                                "box": fb,
                                "id": matched_id,
                                "sim": sim
                            })

            # Rendering
            display_frame = frame.copy()
            now_dt = datetime.now()

            # Associate detected faces to people bounding boxes
            for p in cached_people:
                px1, py1, px2, py2 = p
                matched_face = None

                for f in cached_faces:
                    fx, fy, fw, fh = f["box"]
                    fcx, fcy = fx + fw // 2, fy + fh // 2
                    if px1 <= fcx <= px2 and py1 <= fcy <= py2:
                        matched_face = f
                        break

                color = (180, 180, 180)  # Default gray
                label = "Person"

                if matched_face and matched_face["id"] != "Unknown":
                    eid = matched_face["id"].upper()
                    emp_info = employee_registry.get(eid)

                    if emp_info:
                        emp_name = emp_info["name"]
                        emp_floor = emp_info["floor"]

                        # Check cafeteria status for this employee
                        c_status = get_employee_cafeteria_status(
                            employee_floor=emp_floor,
                            now_dt=now_dt,
                            grace_seconds=grace_seconds
                        )
                        state = c_status["state"]

                        if state == "ON_BREAK":
                            color = (0, 220, 0)  # Green
                            label = f"{emp_name} ({eid}) [{emp_floor}F Break]"
                        elif state == "POST_BREAK_OVERSTAY":
                            color = (0, 0, 255)  # Bright Red
                            label = f"{emp_name} ({eid}) [OVERSTAYING AFTER BREAK]"

                            # Debounce overstay sighting (>= 10 seconds before dispatching email)
                            b_tag = c_status.get("break_tag") or "overstay"
                            if (eid, b_tag) not in alerted_overstays:
                                if eid not in seen_overstay_start:
                                    seen_overstay_start[eid] = time.time()
                                elif time.time() - seen_overstay_start[eid] >= 10:
                                    # Trigger alert email
                                    alerted_overstays.add((eid, b_tag))
                                    snap_path = str(ALERTS_DIR / f"overstay_{eid}_{int(time.time())}.jpg")
                                    cv2.imwrite(snap_path, frame)
                                    print(f"[CAFETERIA ALERT] {emp_name} ({eid}) overstaying in cafeteria after {c_status['break_name']}!")
                                    send_cafeteria_overstay_email(
                                        employee_info=emp_info,
                                        break_name=c_status["break_name"],
                                        break_end_time=c_status["break_end_time"],
                                        snapshot_path=snap_path,
                                        area_name=area_title
                                    )
                        elif state == "WORKING_HOURS_UNAUTHORIZED":
                            color = (0, 140, 255)  # Orange
                            label = f"{emp_name} ({eid}) [{emp_floor}F Work Hours]"
                        else:
                            label = f"{emp_name} ({eid})"
                    else:
                        label = f"Employee {eid}"

                cv2.rectangle(display_frame, (px1, py1), (px2, py2), color, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(display_frame, (px1, max(20, py1 - th - 6)), (px1 + tw + 6, max(20, py1)), (20, 20, 20), -1)
                cv2.putText(display_frame, label, (px1 + 3, max(18, py1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Top HUD Bar
            now_t = time.time()
            if now_t - prev_time > 0:
                display_fps = 1.0 / (now_t - prev_time)
            prev_time = now_t

            cv2.rectangle(display_frame, (0, 0), (1280, 50), (18, 18, 18), -1)
            cv2.line(display_frame, (0, 50), (1280, 50), (60, 60, 60), 1)

            now_str = now_dt.strftime("%a %b %d, %I:%M %p")
            cv2.putText(display_frame, f"{area_title}  |  {now_str}", (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1)

            # Floor break schedule tags
            s3 = get_schedule_status(now_dt, floor="3")
            s4 = get_schedule_status(now_dt, floor="4")
            s3_tag = f"3F: ON {s3.get('break_type', 'BREAK').replace('_', ' ')}" if s3.get("is_break") else "3F Breaks: 11:15-11:30 | 1:00-2:00 | 4:15-4:30"
            s4_tag = f"4F: ON {s4.get('break_type', 'BREAK').replace('_', ' ')}" if s4.get("is_break") else "4F Breaks: 11:00-11:15 | 1:00-2:00 | 4:00-4:15"
            hud_sched = f"[{s3_tag}]  [{s4_tag}]"
            cv2.putText(display_frame, hud_sched, (480, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 215, 255), 1)

            hud_metrics = f"Occupants: {len(cached_people)}  |  Identified: {len([f for f in cached_faces if f['id'] != 'Unknown'])}  |  FPS: {display_fps:.1f}"
            cv2.putText(display_frame, hud_metrics, (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 200), 1)

            if paused:
                cv2.putText(display_frame, "[PAUSED]", (580, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow(window_title, display_frame)

            try:
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                pass

            wait_time = 30 if paused else 1
            key = cv2.waitKey(wait_time) & 0xFF
            if key in [ord("q"), ord("Q"), 27]:
                break
            elif key in [ord("p"), ord("P")]:
                paused = not paused
            elif key in [ord("s"), ord("S")]:
                snap = ALERTS_DIR / f"manual_snap_{int(time.time())}.jpg"
                cv2.imwrite(str(snap), display_frame)
                print(f"[INFO] Snapshot saved: {snap}")

    except KeyboardInterrupt:
        print("[INFO] Monitoring stopped.")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera released. Session closed.")


if __name__ == "__main__":
    main()
