"""
Workplace Seat Monitoring System - 3rd Floor Workstations
==========================================================
Monitors workstations, detects people and faces, recognizes enrolled employees,
and tracks seat occupancy & employee absences according to seat_monitoring logic.

Key Features:
- Multi-Model Pipeline:
    * YOLO11 at 1280px resolution for robust seated person detection
    * YuNet with anatomical landmark validation to reject hands/keyboards/hair false positives
    * SFace deep face recognition with calibrated similarity & margin thresholds
- Seat Occupancy & Verification:
    * Occupancy confirmed by person body detection OR valid face detection in seat zone
    * Verifies if occupant matches assigned employee
    * Grace-period debouncing prevents momentary posture shifts from triggering false absence
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
import argparse
import smtplib
import numpy as np

import sys
from pathlib import Path
from datetime import datetime
from email.message import EmailMessage
from collections import defaultdict
from ultralytics import YOLO

# Ensure monitoring directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import robust assignment, clustering, and geometric face validator
from seat_logic import (
    is_valid_face,
    assign_people_and_faces_to_seats,
    filter_overlapping_people,
    detect_gathering_clusters,
    analyze_gathering_members
)

# Import office schedule and tea break manager
from schedule_logic import (
    get_schedule_status,
    get_post_break_status,
    is_shift_active,
    WEEKDAY_START_TIME,
    WEEKDAY_END_TIME,
    SATURDAY_START_TIME,
    SATURDAY_END_TIME,
    MORNING_TEA_START,
    MORNING_TEA_END,
    EVENING_TEA_START,
    EVENING_TEA_END
)


# ============================================================
# 1. CONFIGURATION & PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

def load_env_file(env_path):
    """Load environment variables from a .env file without external dependencies."""
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            if k and k not in os.environ:
                os.environ[k] = v

# Load local .env if present
load_env_file(BASE_DIR / ".env")

# Video Source Configuration:
# Set your live CCTV camera stream URL below.
# Common camera RTSP URL formats:
#   Hanwha / Wisenet : "rtsp://admin:YourPassword@172.30.0.21:554/profile2/media.smp" (or profile1)
#   Hikvision / Ezviz: "rtsp://admin:YourPassword@192.168.1.100:554/Streaming/Channels/101"
#   CP Plus / Dahua  : "rtsp://admin:YourPassword@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0"
#   Generic ONVIF    : "rtsp://admin:YourPassword@192.168.1.100:554/live/ch0"
#   USB Webcam       : 0
LIVE_CCTV_URL = ""  # <-- Put your live camera RTSP URL here

FULL_MP4_VIDEO = BASE_DIR / "3rd_floor_full.mp4"
PRIMARY_VIDEO = BASE_DIR / "3rd Floor WS.avi"
FALLBACK_VIDEO = BASE_DIR / "detection" / "reports" / "3rd_floor_test.mp4"

if LIVE_CCTV_URL:
    VIDEO_SOURCE = LIVE_CCTV_URL
else:
    if FULL_MP4_VIDEO.exists():
        DEFAULT_VIDEO_SOURCE = str(FULL_MP4_VIDEO)
    elif PRIMARY_VIDEO.exists():
        DEFAULT_VIDEO_SOURCE = str(PRIMARY_VIDEO)
    elif FALLBACK_VIDEO.exists():
        DEFAULT_VIDEO_SOURCE = str(FALLBACK_VIDEO)
    else:
        DEFAULT_VIDEO_SOURCE = str(BASE_DIR / "detection" / "reports" / "cctv_test_25fps.mp4")

    VIDEO_SOURCE = os.getenv("CCTV_VIDEO_SOURCE", DEFAULT_VIDEO_SOURCE)

# Set True for recorded video files, False for live webcam/RTSP
IS_RECORDED_VIDEO = (
    isinstance(VIDEO_SOURCE, str)
    and os.path.exists(VIDEO_SOURCE)
    and not str(VIDEO_SOURCE).startswith("rtsp")
    and not str(VIDEO_SOURCE).isdigit()
)

# Models & Database paths
YOLO_MODEL_PATH = str(BASE_DIR / "yolo11n.pt")
YUNET_MODEL_PATH = str(BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx")
SFACE_MODEL_PATH = str(BASE_DIR / "models" / "face_recognition_sface_2021dec.onnx")
EMBEDDINGS_PATH = str(BASE_DIR / "recognition" / "output" / "employee_embeddings.npz")
DATABASE_PATH = str(BASE_DIR / "seat_tracking.db")
SEATS_CONFIG_PATH = str(BASE_DIR / "monitoring" / "seats_3rd_floor.json")

# Timing & Detection Parameters
DETECTION_INTERVAL = 2          # Run neural inference every N frames for smooth speed
YOLO_IMGSZ = 1280               # High resolution inference to catch seated employees
PERSON_CONFIDENCE = 0.15        # Low threshold tuned for seated/top-down CCTV posture
FACE_CONFIDENCE = 0.52          # YuNet threshold (rejects false positives on hands/keyboard)
FACE_SIMILARITY_THRESH = 0.363  # Calibrated SFace threshold from recognition dataset
FACE_MARGIN_THRESH = 0.05       # Separation from 2nd best employee match

# Absence Tracking Parameters (Default: 60s / 1 minute for testing)
ABSENCE_LIMIT_SECONDS = int(os.getenv("ABSENCE_LIMIT_SECONDS", 60))  # 1 minute (60s)
EMPTY_CONFIRM_SECONDS = 5        # Debounce time before confirming absence
RETURN_CONFIRM_SECONDS = 4       # Debounce time to filter out hallway passersby before confirming return

# Gathering Alert Parameters (Alert when > 2 people gather for >= 5 minutes)
GATHERING_MIN_PEOPLE = int(os.getenv("GATHERING_MIN_PEOPLE", 3))           # More than 2 people (>= 3)
GATHERING_LIMIT_SECONDS = int(os.getenv("GATHERING_LIMIT_SECONDS", 300))   # 5 minutes (300s)
GATHERING_DISTANCE_THRESHOLD = int(os.getenv("GATHERING_DISTANCE", 190))   # Proximity clustering threshold in pixels
GATHERING_DEBOUNCE_SECONDS = int(os.getenv("GATHERING_DEBOUNCE_SECONDS", 10)) # Seconds before marking gathering dispersed

# Email Alerts Configuration
ENABLE_EMAIL_ALERTS = os.getenv("ENABLE_EMAIL_ALERTS", "true").lower() in ("true", "1", "yes")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "kristellardc@gmail.com")
SENDER_PASSWORD = os.getenv("SENDER_APP_PASSWORD", "").replace(" ", "")
DEFAULT_RM_EMAIL = os.getenv("DEFAULT_RM_EMAIL", "neha@kristellar.com")

# Office Operating Schedule & Shift Enforcement
# Mon-Fri: 09:30 AM - 06:30 PM | Sat: 09:30 AM - 01:30 PM | Sun: Closed
# Tea Breaks (Paused, no alerts): 11:00 - 11:30 AM & 04:00 - 04:30 PM
ENFORCE_SCHEDULE = os.getenv("ENFORCE_SCHEDULE", "true").lower() in ("true", "1", "yes")
EXIT_WHEN_CLOSED = os.getenv("EXIT_WHEN_CLOSED", "false").lower() in ("true", "1", "yes")


# ============================================================
# 2. SEAT CONFIGURATION LOADER
# ============================================================

def load_seats_config(config_path=None):
    target = Path(config_path or SEATS_CONFIG_PATH)
    if not target.is_absolute():
        target = BASE_DIR / target
    if os.path.exists(target):
        try:
            with open(target, "r", encoding="utf-8") as f:
                seats = json.load(f)
                print(f"[SUCCESS] Loaded {len(seats)} seats from {target.name}")
                return seats
        except Exception as e:
            print(f"[WARNING] Could not read {target} ({e}).")
    return {}


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

    conn.execute("""
        CREATE TABLE IF NOT EXISTS gatherings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gathering_tag TEXT,
            location_desc TEXT,
            person_count INTEGER,
            identified_employees TEXT,
            gathering_start TEXT NOT NULL,
            gathering_end TEXT,
            duration_seconds INTEGER,
            email_sent INTEGER DEFAULT 0,
            snapshot_path TEXT,
            source TEXT
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


def start_gathering_record(tag, location_desc, person_count, identified_str, source):
    now = datetime.now()
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.execute("""
        INSERT INTO gatherings
        (gathering_tag, location_desc, person_count, identified_employees, gathering_start, source)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        tag,
        location_desc,
        int(person_count),
        identified_str,
        now.isoformat(),
        str(source)
    ))
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id


def close_gathering_record(record_id, duration):
    now = datetime.now()
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("""
        UPDATE gatherings
        SET gathering_end = ?,
            duration_seconds = ?
        WHERE id = ?
    """, (
        now.isoformat(),
        int(duration),
        record_id
    ))
    conn.commit()
    conn.close()


def mark_gathering_email_sent(record_id, snapshot_path):
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("""
        UPDATE gatherings
        SET email_sent = 1,
            snapshot_path = ?
        WHERE id = ?
    """, (snapshot_path, record_id))
    conn.commit()
    conn.close()


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


def send_absence_email(employee_name, seat_id, rm_email, start_time, duration, area_name="3rd Floor Workstation Area"):
    """
    Sends an automated email notification to the Reporting Manager (RM)
    when an employee has been away from their workstation for >= 10 minutes.
    """
    if not rm_email:
        rm_email = DEFAULT_RM_EMAIL

    duration_str = format_duration(duration)
    minutes_away = max(1, int(duration // 60))
    time_str = start_time.strftime('%Y-%m-%d %H:%M:%S') if isinstance(start_time, datetime) else str(start_time)

    if not ENABLE_EMAIL_ALERTS:
        print(f"[INFO] Email alerts disabled (ENABLE_EMAIL_ALERTS=false). Alert for {employee_name} ({seat_id}) skipped.")
        return False

    # Office Schedule & Tea Break Verification
    sched = get_schedule_status(enforce=ENFORCE_SCHEDULE)
    if not sched["can_send_email"]:
        print(f"[INFO] Absence email suppressed ({sched['reason']}). Alert for {employee_name} ({seat_id}) skipped.")
        return False

    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print("\n" + "=" * 65)
        print(f" [EMAIL ALERT TRIGGERED - SIMULATION MODE]")
        print(f" Area                   : {area_name}")
        print(f" To (Reporting Manager) : {rm_email}")
        print(f" Subject                : [Workplace Alert - {area_name}] {employee_name} ({seat_id}) is not in their seat since {minutes_away} minutes")
        print(f" Workstation            : {seat_id}")
        print(f" Employee Name          : {employee_name}")
        print(f" Absence Started At     : {time_str}")
        print(f" Duration Away          : {duration_str}")
        print(f" Notice                 : Set SENDER_EMAIL and SENDER_APP_PASSWORD in .env to dispatch live emails.")
        print("=" * 65 + "\n")
        return False

    message = EmailMessage()
    message["Subject"] = f"[Workplace Alert - {area_name}] {employee_name} ({seat_id}) is not in their seat since {minutes_away} minutes"
    message["From"] = SENDER_EMAIL
    message["To"] = rm_email

    # Plain text version
    plain_content = f"""Workplace Workstation Monitoring Alert - {area_name}
===================================================
Dear Reporting Manager,

This is an automated notification from the Workplace Monitoring System ({area_name}).

The employee assigned to the workstation below has not been in their seat for the last {minutes_away} minutes:

  • Monitored Area     : {area_name}
  • Workstation ID     : {seat_id}
  • Employee Name      : {employee_name}
  • Reporting Manager  : {rm_email}
  • Absence Started At : {time_str}
  • Total Duration Away: {duration_str} (Threshold: 10 minutes)

Status: This employee is currently NOT detected at their assigned workstation.
Please verify the employee's status or confirm their availability.

--
Automated Workplace Monitoring System
Kristellar Aerospace
"""
    message.set_content(plain_content)

    # HTML version
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
      <h2>Workplace Absence Alert</h2>
    </div>
    <div class="body">
      <div class="alert-banner">
        ⚠️ Alert: Employee has not been in their seat since {minutes_away} minutes.
      </div>
      <p style="font-size: 14px; color: #24292f; margin-top: 0;">
        Dear Reporting Manager,<br><br>
        The Workplace Monitoring System ({area_name}) has detected that the employee assigned to the workstation below has been continuously absent beyond the <strong>10-minute threshold</strong>:
      </p>
      <table>
        <tr><td class="label">Workstation ID</td><td class="val"><strong>{seat_id}</strong></td></tr>
        <tr><td class="label">Employee Name</td><td class="val"><strong>{employee_name}</strong></td></tr>
        <tr><td class="label">Reporting Manager</td><td class="val">{rm_email}</td></tr>
        <tr><td class="label">Absence Started At</td><td class="val">{time_str}</td></tr>
        <tr><td class="label">Duration Away</td><td class="val" style="color: #cf222e; font-weight: bold;">{duration_str}</td></tr>
        <tr><td class="label">Current Status</td><td class="val"><span class="badge">NOT IN SEAT</span></td></tr>
      </table>
      <p style="font-size: 13px; color: #57606a; margin-top: 20px;">
        Please check in with the employee to verify their status.
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

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=20) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(message)
        print(f"[EMAIL SENT] Absence notification successfully sent to RM ({rm_email}) for {employee_name} ({seat_id}).")
        return True
    except Exception as error:
        print(f"[EMAIL ERROR] Failed to send email to {rm_email}: {error}")
        return False


def send_post_break_absence_email(employee_name, seat_id, employee_id, rm_email, break_name, break_end_time, area_name="Workstation Area"):
    """
    Sends an automated email notification to the Reporting Manager (RM)
    when an employee has not returned to their workstation after their scheduled break ended.
    """
    if not rm_email:
        rm_email = DEFAULT_RM_EMAIL

    if not ENABLE_EMAIL_ALERTS:
        print(f"[INFO] Email alerts disabled. Post-break alert for {employee_name} ({seat_id}) skipped.")
        return False

    now_str = datetime.now().strftime("%Y-%m-%d %I:%M %p")

    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print("\n" + "=" * 65)
        print(" [EMAIL ALERT TRIGGERED - POST-BREAK ABSENCE SIMULATION]")
        print(f" Area                   : {area_name}")
        print(f" To (Reporting Manager) : {rm_email}")
        print(f" Subject                : [Workplace Alert - Post-Break Absence - {area_name}] {employee_name} ({seat_id}) is not in their seat after {break_name}")
        print(f" Workstation            : {seat_id}")
        print(f" Employee Name          : {employee_name} (ID: {employee_id})")
        print(f" Break Name             : {break_name}")
        print(f" Break Ended At         : {break_end_time}")
        print(f" Verification Time      : {now_str}")
        print(f" Notice                 : Set SENDER_EMAIL and SENDER_APP_PASSWORD in .env to dispatch live emails.")
        print("=" * 65 + "\n")
        return False

    message = EmailMessage()
    message["Subject"] = f"[Workplace Alert - Post-Break Absence - {area_name}] {employee_name} ({seat_id}) is not in their seat after {break_name}"
    message["From"] = SENDER_EMAIL
    message["To"] = rm_email

    plain_content = f"""Workplace Monitoring Alert - Post-Break Absence ({area_name})
===================================================
Dear Reporting Manager,

This is an automated notification from the Workplace Monitoring System ({area_name}).

The employee assigned to the workstation below has not returned to their seat after the scheduled tea break:

  • Monitored Area     : {area_name}
  • Workstation ID     : {seat_id}
  • Employee Name      : {employee_name}
  • Employee ID        : {employee_id}
  • Reporting Manager  : {rm_email}
  • Scheduled Break    : {break_name}
  • Break Ended At     : {break_end_time}
  • Verified Absent At : {now_str}

Status: This employee is NOT in their seat after their floor's authorized break time.
Please check in with the employee to confirm their availability.

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
    .header {{ background-color: #d9381e; color: #ffffff; padding: 18px 24px; }}
    .header h2 {{ margin: 0; font-size: 19px; }}
    .body {{ padding: 24px; }}
    .alert-banner {{ background-color: #ffebe9; border-left: 4px solid #d9381e; border-radius: 4px; padding: 12px 16px; margin-bottom: 20px; color: #cf222e; font-weight: 600; font-size: 14px; }}
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
      <h2>Workplace Post-Break Absence Alert</h2>
    </div>
    <div class="body">
      <div class="alert-banner">
        ⚠️ Notice: Employee has not returned to their seat after {break_name} (ended at {break_end_time}).
      </div>
      <p style="font-size: 14px; color: #24292f; margin-top: 0;">
        Dear Reporting Manager,<br><br>
        The Workplace Monitoring System ({area_name}) has detected that the employee below was not back at their assigned workstation after the scheduled break ended:
      </p>
      <table>
        <tr><td class="label">Workstation ID</td><td class="val"><strong>{seat_id}</strong></td></tr>
        <tr><td class="label">Employee Name</td><td class="val"><strong>{employee_name}</strong> (ID: {employee_id})</td></tr>
        <tr><td class="label">Reporting Manager</td><td class="val">{rm_email}</td></tr>
        <tr><td class="label">Scheduled Break</td><td class="val">{break_name}</td></tr>
        <tr><td class="label">Break Ended At</td><td class="val"><strong>{break_end_time}</strong></td></tr>
        <tr><td class="label">Status</td><td class="val"><span class="badge">NOT IN SEAT AFTER BREAK</span></td></tr>
      </table>
      <p style="font-size: 13px; color: #57606a; margin-top: 20px;">
        Please check in with the employee to verify their whereabouts.
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

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=20) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(message)
        print(f"[EMAIL SENT] Post-break absence notification sent to RM ({rm_email}) for {employee_name} ({seat_id}).")
        return True
    except Exception as error:
        print(f"[EMAIL ERROR] Failed to send post-break email to {rm_email}: {error}")
        return False


def send_gathering_email(gathering_info, rm_email, relevant_members, duration, snapshot_path=None, area_name="3rd Floor Workstation Area"):
    """
    Sends an automated email notification to the respective Reporting Manager (RM)
    when an unauthorized employee gathering (> 2 people) persists for >= 5 minutes.
    """
    if not rm_email:
        rm_email = DEFAULT_RM_EMAIL

    duration_str = format_duration(duration)
    minutes_gathered = max(1, int(duration // 60))
    start_time = gathering_info["start_time"]
    time_str = start_time.strftime('%Y-%m-%d %H:%M:%S') if isinstance(start_time, datetime) else str(start_time)
    count = gathering_info["count"]
    location = gathering_info["location_desc"]

    all_identified = gathering_info.get("identified", [])
    unidentified_count = gathering_info.get("unidentified_count", 0)

    if relevant_members:
        target_names = ", ".join([m["name"] for m in relevant_members])
    else:
        target_names = "Employees in your work area"

    if not ENABLE_EMAIL_ALERTS:
        print(f"[INFO] Email alerts disabled. Gathering alert for {location} skipped.")
        return False

    # Office Schedule & Tea Break Verification
    sched = get_schedule_status(enforce=ENFORCE_SCHEDULE)
    if not sched["can_send_email"]:
        print(f"[INFO] Gathering email suppressed ({sched['reason']}). Incident {gathering_info['id']} skipped.")
        return False

    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print("\n" + "=" * 65)
        print(" [EMAIL ALERT TRIGGERED - GATHERING SIMULATION MODE]")
        print(f" Area                   : {area_name}")
        print(f" To (Reporting Manager) : {rm_email}")
        print(f" Subject                : [Workplace Alert - {area_name}] Employee Gathering ({count} People for {minutes_gathered} min) near {location}")
        print(f" Location               : {location}")
        print(f" People Count           : {count}")
        print(f" Relevant Team Members  : {target_names}")
        print(f" Duration Gathered      : {duration_str}")
        if snapshot_path:
            print(f" Snapshot Evidence      : {snapshot_path}")
        print("=" * 65 + "\n")
        return False

    message = EmailMessage()
    message["Subject"] = f"[Workplace Alert - {area_name}] Employee Gathering Detected ({count} People for {minutes_gathered} min) near {location}"
    message["From"] = SENDER_EMAIL
    message["To"] = rm_email

    # Plain text member list
    member_rows_plain = []
    member_rows_html = []
    if all_identified:
        for idx, m in enumerate(all_identified, 1):
            is_relevant = any(m["name"] == rm_m["name"] for rm_m in relevant_members)
            tag = " [YOUR TEAM MEMBER]" if is_relevant else ""
            member_rows_plain.append(f"  {idx}. {m['name']} (ID: {m['employee_id']}, Seat: {m['seat_id']}){tag}")
            row_style = "background-color: #fff8e6; font-weight: bold;" if is_relevant else ""
            badge = '<span style="background-color: #b35900; color: white; padding: 2px 6px; border-radius: 4px; font-size: 11px; margin-left: 6px;">Your Team</span>' if is_relevant else ''
            member_rows_html.append(
                f'<tr style="{row_style}"><td>{m["name"]}{badge}</td><td>{m["employee_id"]}</td><td>{m["seat_id"]}</td><td>{m["rm_email"]}</td></tr>'
            )
    if unidentified_count > 0:
        member_rows_plain.append(f"  + {unidentified_count} Unidentified Person(s)")
        member_rows_html.append(
            f'<tr><td colspan="4" style="color: #6e7781; font-style: italic;">+ {unidentified_count} Unidentified Person(s)</td></tr>'
        )

    plain_members_str = "\n".join(member_rows_plain) if member_rows_plain else "  Unidentified Individuals"
    html_members_str = "".join(member_rows_html)

    plain_content = f"""Workplace Monitoring Alert - Employee Gathering
===================================================
Dear Reporting Manager,

This is an automated notification from the Workplace Monitoring System ({area_name}).

A gathering of {count} people has been detected in a workstation area continuously for more than {minutes_gathered} minutes:

  • Incident Tag       : {gathering_info['id']}
  • Location           : {location}
  • People Count       : {count} individuals (Threshold: > 2 people)
  • Gathering Started  : {time_str}
  • Total Duration     : {duration_str} (Threshold: 5 minutes)
  • Reporting Manager  : {rm_email}

Detected Individuals in Gathering:
{plain_members_str}

Notice: Gatherings exceeding 5 minutes in work areas require supervisor review. Please verify with your team members to ensure normal workflow.

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
    .container {{ max-width: 600px; margin: auto; background: #ffffff; border-radius: 8px; border: 1px solid #e1e4e8; overflow: hidden; box-shadow: 0 2px 6px rgba(0,0,0,0.06); }}
    .header {{ background-color: #b35900; color: #ffffff; padding: 18px 24px; }}
    .header h2 {{ margin: 0; font-size: 19px; }}
    .body {{ padding: 24px; }}
    .alert-banner {{ background-color: #fff8e6; border-left: 4px solid #b35900; border-radius: 4px; padding: 12px 16px; margin-bottom: 20px; color: #8a4500; font-weight: 600; font-size: 14px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 14px; }}
    th {{ background-color: #f6f8fa; text-align: left; padding: 8px 10px; font-size: 12px; color: #57606a; border-bottom: 1px solid #eaecef; }}
    td {{ padding: 9px 10px; border-bottom: 1px solid #eaecef; font-size: 13px; }}
    td.label {{ font-weight: 600; color: #57606a; width: 150px; }}
    td.val {{ color: #24292f; }}
    .badge {{ background-color: #ffebe9; color: #cf222e; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: bold; border: 1px solid #ff8182; }}
    .footer {{ background-color: #f6f8fa; padding: 14px 24px; font-size: 12px; color: #6e7781; border-top: 1px solid #e1e4e8; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h2>Workplace Gathering Alert (&gt; 5 Minutes)</h2>
    </div>
    <div class="body">
      <div class="alert-banner">
        ⚠️ Notice: A gathering of {count} people has exceeded the 5-minute threshold at {location}.
      </div>
      <p style="font-size: 14px; color: #24292f; margin-top: 0;">
        Dear Reporting Manager,<br><br>
        The Workplace Monitoring System ({area_name}) detected an employee gathering persisting beyond the <strong>5-minute threshold</strong>:
      </p>
      <table>
        <tr><td class="label">Incident ID</td><td class="val"><strong>{gathering_info['id']}</strong></td></tr>
        <tr><td class="label">Location</td><td class="val"><strong>{location}</strong></td></tr>
        <tr><td class="label">People Count</td><td class="val"><span class="badge">{count} Individuals</span></td></tr>
        <tr><td class="label">Started At</td><td class="val">{time_str}</td></tr>
        <tr><td class="label">Duration Gathered</td><td class="val" style="color: #b35900; font-weight: bold;">{duration_str}</td></tr>
      </table>

      <h4 style="margin-top: 20px; margin-bottom: 8px; color: #24292f;">People Detected in Gathering:</h4>
      <table>
        <thead>
          <tr><th>Name</th><th>Employee ID</th><th>Assigned Desk</th><th>RM Email</th></tr>
        </thead>
        <tbody>
          {html_members_str}
        </tbody>
      </table>

      <p style="font-size: 12px; color: #57606a; margin-top: 20px;">
        Snapshot evidence of the gathering has been recorded and attached to this email.
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

    # Attach Snapshot if available
    if snapshot_path and os.path.exists(snapshot_path):
        try:
            with open(snapshot_path, "rb") as img_file:
                img_data = img_file.read()
                message.add_attachment(
                    img_data,
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
        print(f"[EMAIL SENT] Gathering notification sent to RM ({rm_email}) for incident {gathering_info['id']}.")
        return True
    except Exception as error:
        print(f"[EMAIL ERROR] Failed to send gathering email to {rm_email}: {error}")
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


def get_absence_elapsed(absence, video_seconds):
    if IS_RECORDED_VIDEO:
        if absence["video_start"] is None:
            absence["video_start"] = video_seconds
        return max(0, int(video_seconds - absence["video_start"]))
    return max(0, int(time.monotonic() - absence["empty_since"]))


# ============================================================
# 6. MAIN MONITORING PROGRAM
# ============================================================

def main():
    print("=========================================================")
    print("        WORKPLACE SEAT & GATHERING MONITORING SYSTEM      ")
    print("=========================================================")

    create_database()

    global VIDEO_SOURCE, IS_RECORDED_VIDEO, ABSENCE_LIMIT_SECONDS, ENABLE_EMAIL_ALERTS
    global GATHERING_LIMIT_SECONDS, GATHERING_MIN_PEOPLE, ENFORCE_SCHEDULE, EXIT_WHEN_CLOSED

    parser = argparse.ArgumentParser(description="Workplace Seat & Gathering Monitoring System")
    parser.add_argument(
        "--source",
        type=str,
        default=VIDEO_SOURCE,
        help="Video file path, RTSP stream URL, or webcam index (0)"
    )
    parser.add_argument(
        "--floor",
        type=str,
        default=None,
        help="Quick area selector: '3' (3rd Floor - 28 Seats), '3-area2' (3rd Floor Area 2 - 9 Seats), '4' (4th Floor - 24 Seats)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom seat configuration JSON file"
    )
    parser.add_argument(
        "--area-name",
        type=str,
        default=None,
        help="Display title for the monitored area"
    )
    parser.add_argument(
        "--absence-limit",
        type=int,
        default=ABSENCE_LIMIT_SECONDS,
        help="Absence limit in seconds before alert (default: 600, i.e. 10 minutes)"
    )
    parser.add_argument(
        "--gathering-limit",
        type=int,
        default=GATHERING_LIMIT_SECONDS,
        help="Gathering duration limit in seconds before alert (default: 300, i.e. 5 minutes)"
    )
    parser.add_argument(
        "--gathering-min-people",
        type=int,
        default=GATHERING_MIN_PEOPLE,
        help="Minimum people count to consider a gathering (default: 3, i.e. > 2 people)"
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
        help="Bypass office hours schedule and tea break restrictions (for 24/7 testing)"
    )
    parser.add_argument(
        "--exit-when-closed",
        action="store_true",
        default=False,
        help="Exit the program automatically when office hours end"
    )
    parser.add_argument(
        "--post-break-grace",
        type=int,
        default=60,
        help="Grace period (in seconds) after break ends before sending absence alert (default: 60s)"
    )
    args, _ = parser.parse_known_args()

    if args.send_email:
        ENABLE_EMAIL_ALERTS = True

    if args.ignore_schedule:
        ENFORCE_SCHEDULE = False
    if args.exit_when_closed:
        EXIT_WHEN_CLOSED = True

    # 1. Determine Floor & Seat Configuration Path
    if args.floor:
        fl = str(args.floor).lower().strip()
        if fl in ("3", "3f", "3rd", "3rd_floor", "3-main"):
            chosen_config = BASE_DIR / "monitoring" / "seats_3rd_floor.json"
            area_title = "3rd Floor Workstation Area 1"
        elif fl in ("3-area2", "3-2", "3a2", "area2", "3rd_area2"):
            chosen_config = BASE_DIR / "monitoring" / "seats_3rd_floor_area2.json"
            area_title = "3rd Floor Workstation Area 2"
        elif fl in ("4", "4f", "4th", "4th_floor"):
            chosen_config = BASE_DIR / "monitoring" / "seats_4th_floor.json"
            area_title = "4th Floor Workstation Area"
        else:
            chosen_config = Path(args.config) if args.config else Path(SEATS_CONFIG_PATH)
            area_title = args.area_name or f"Floor {args.floor} Workstation Area"
    elif args.config:
        chosen_config = Path(args.config)
        area_title = args.area_name or chosen_config.stem.replace("_", " ").title()
    else:
        chosen_config = Path(os.getenv("SEATS_CONFIG_PATH", SEATS_CONFIG_PATH))
        area_title = args.area_name or os.getenv("AREA_NAME", "3rd Floor Workstation Area 1")

    if args.area_name:
        area_title = args.area_name
    window_title = f"{area_title} - Seat Monitor"
    floor_id = "4" if ("4" in str(args.floor or chosen_config)) else "3"
    POST_BREAK_GRACE_SECONDS = args.post_break_grace

    seats = load_seats_config(chosen_config)
    if not seats:
        print(f"[ERROR] No seats loaded from config ({chosen_config})!")
        return

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

    if str(args.source).isdigit():
        VIDEO_SOURCE = int(args.source)
        IS_RECORDED_VIDEO = False
    elif str(args.source).lower().startswith(("rtsp://", "http://", "https://")):
        VIDEO_SOURCE = str(args.source)
        IS_RECORDED_VIDEO = False
    else:
        VIDEO_SOURCE = str(args.source)
        IS_RECORDED_VIDEO = os.path.exists(VIDEO_SOURCE)

    ABSENCE_LIMIT_SECONDS = args.absence_limit
    GATHERING_LIMIT_SECONDS = args.gathering_limit
    GATHERING_MIN_PEOPLE = args.gathering_min_people

    init_sched = get_schedule_status(enforce=ENFORCE_SCHEDULE, floor=floor_id)
    print(f"[INFO] Area Selected: {area_title} ({len(seats)} Seats)")
    print(f"[INFO] Config File: {chosen_config}")
    print(f"[INFO] Office Schedule Enforcement: {'ENABLED' if ENFORCE_SCHEDULE else 'DISABLED (Override/Testing Mode)'}")
    print(f"[INFO] Schedule State: {init_sched['status']} - {init_sched['reason']}")

    # 4. Open Video Source
    print("[INFO] Opening Video Source:", VIDEO_SOURCE)
    if isinstance(VIDEO_SOURCE, str) and VIDEO_SOURCE.lower().startswith("rtsp://"):
        # Force TCP transport to eliminate packet loss and gray artifacts on live CCTV streams
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    camera = cv2.VideoCapture(VIDEO_SOURCE)

    if not IS_RECORDED_VIDEO:
        # Minimize buffer size to ensure strictly zero latency in live real-time feeds
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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
    print(f"[INFO] Gathering Alert: > {GATHERING_MIN_PEOPLE - 1} people for >= {format_duration(GATHERING_LIMIT_SECONDS)}")
    print("[INFO] Press 'Q' to quit | 'P' to pause | 'S' screenshot | 'C' toggle seats | 'G' gatherings | 'T' toggle schedule")

    # State tracking
    active_absences = {}
    empty_candidates = {}
    return_candidates = {}
    alerted_post_breaks = set()

    # Gathering state tracking
    active_gatherings = {}
    gathering_counter = 0
    GATHERING_ALERTS_DIR = BASE_DIR / "monitoring" / "gathering_alerts"
    os.makedirs(GATHERING_ALERTS_DIR, exist_ok=True)

    cached_people = []
    cached_faces = []
    cached_seat_status = {}

    frame_number = 0
    show_seats_overlay = True
    show_gatherings_overlay = True
    show_detections_overlay = True
    paused = False
    prev_time = time.time()
    display_fps = 0.0

    window_title = f"{area_title} - Seat Monitor"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, 1280, 720)

    try:
        while True:
            if not paused:
                success, raw_frame = camera.read()
                if not success:
                    if IS_RECORDED_VIDEO:
                        # Seamless continuous loop for recorded video footage
                        print("[INFO] Video loop reached end. Seamlessly rewinding and looping...")
                        camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        success, raw_frame = camera.read()
                        if not success:
                            camera.release()
                            camera = cv2.VideoCapture(VIDEO_SOURCE)
                            success, raw_frame = camera.read()
                        if not success:
                            time.sleep(0.5)
                            continue
                    else:
                        # Auto-reconnect live camera / RTSP stream without stopping session
                        print("[WARNING] Camera stream interrupted. Attempting reconnection...")
                        time.sleep(1.0)
                        camera.release()
                        camera = cv2.VideoCapture(VIDEO_SOURCE)
                        success, raw_frame = camera.read()
                        if not success:
                            time.sleep(0.5)
                            continue

                frame_number += 1
                frame = cv2.resize(raw_frame, (1280, 720))

                # Video timeline (continuous monotonically increasing timeline across loops)
                video_seconds = frame_number / video_fps

                # ========================================================
                # OFFICE SCHEDULE & BREAK MANAGEMENT
                # ========================================================
                sched = get_schedule_status(enforce=ENFORCE_SCHEDULE, floor=floor_id)
                is_active_shift = (sched["status"] == "ACTIVE")
                is_tea_break = sched.get("is_break", sched["status"] == "TEA_BREAK")
                is_off_hours = (sched["status"] == "OFF_HOURS")

                if is_off_hours and EXIT_WHEN_CLOSED:
                    print(f"[INFO] Office operating hours ended ({sched['reason']}). Exiting session cleanly.")
                    break

                if is_off_hours:
                    # Off-hours standby mode: reset tracking states, zero email alerts
                    active_absences.clear()
                    empty_candidates.clear()
                    return_candidates.clear()
                    active_gatherings.clear()
                    cached_people = []
                    cached_faces = []
                    cached_seat_status = {
                        sid: {"occupied": False, "status_str": "OFF-HOURS", "verification": "OFF-HOURS"}
                        for sid in seats
                    }
                    time.sleep(0.04)  # Conserve system CPU/GPU during closed hours

                elif is_tea_break:
                    # Authorized tea break: clear debounce candidates and gatherings
                    empty_candidates.clear()
                    return_candidates.clear()
                    active_gatherings.clear()

                    # Freeze existing absence timers so tea break minutes are NOT counted against 10-minute threshold
                    now_mono = time.monotonic()
                    for sid, abs_data in active_absences.items():
                        if "last_freeze_mono" in abs_data:
                            freeze_delta = now_mono - abs_data["last_freeze_mono"]
                            abs_data["empty_since"] += freeze_delta
                            if abs_data.get("video_start") is not None:
                                abs_data["video_start"] += (video_seconds - abs_data.get("last_freeze_video", video_seconds))
                        abs_data["last_freeze_mono"] = now_mono
                        abs_data["last_freeze_video"] = video_seconds
                else:
                    # Active working shift: unfreeze any absences that were previously frozen
                    for sid, abs_data in active_absences.items():
                        abs_data.pop("last_freeze_mono", None)
                        abs_data.pop("last_freeze_video", None)

                # Multi-Model Detection Pipeline (Runs during active shifts and tea breaks, skipped during closed off-hours)
                if not is_off_hours and (frame_number == 1 or frame_number % DETECTION_INTERVAL == 0):
                    # 1. Person Detection (YOLO at high resolution)
                    yolo_results = yolo_model.predict(
                        source=frame,
                        classes=[0],
                        conf=PERSON_CONFIDENCE,
                        imgsz=YOLO_IMGSZ,
                        device="cpu",
                        verbose=False
                    )

                    cached_people = []
                    for res in yolo_results:
                        if res.boxes is None:
                            continue
                        for box in res.boxes:
                            cached_people.append(list(map(int, box.xyxy[0].tolist())))

                    # Deduplicate overlapping bounding boxes (NMS)
                    cached_people = filter_overlapping_people(cached_people)

                    # 2. Face Detection (YuNet with geometric validation)
                    detector.setInputSize((1280, 720))
                    _, faces = detector.detect(frame)

                    cached_faces = []
                    if faces is not None and template_matrix is not None and len(known_ids) > 0:
                        for face in faces:
                            # Reject false face detections on hands, keyboard, hair
                            if not is_valid_face(face, min_conf=FACE_CONFIDENCE):
                                continue

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

                                    # Calibrated recognition thresholds
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
                # SEAT ABSENCE TRACKING STATE MACHINE (Active Shifts Only)
                # ========================================================
                if is_active_shift:
                    for seat_id, sdata in seats.items():
                        employee_name = sdata["name"]
                        sinfo = cached_seat_status.get(seat_id, {"occupied": False})
                        occupied = sinfo["occupied"]
                        is_face_verified = sinfo.get("verification", "").startswith("VERIFIED")

                        if occupied:
                            # Clear any pending empty debounce candidates
                            empty_candidates.pop(seat_id, None)

                            # If returning from an active absence:
                            if seat_id in active_absences:
                                # If face is verified, return is immediate.
                                # Otherwise require continuous presence for RETURN_CONFIRM_SECONDS to ignore hallway passersby
                                if is_face_verified:
                                    return_dur = RETURN_CONFIRM_SECONDS + 1
                                else:
                                    if seat_id not in return_candidates:
                                        return_candidates[seat_id] = {
                                            "wall_start": time.monotonic(),
                                            "video_start": video_seconds
                                        }
                                    rcand = return_candidates[seat_id]
                                    if IS_RECORDED_VIDEO:
                                        return_dur = video_seconds - rcand["video_start"]
                                    else:
                                        return_dur = time.monotonic() - rcand["wall_start"]

                                if return_dur >= RETURN_CONFIRM_SECONDS:
                                    absence = active_absences[seat_id]
                                    duration = get_absence_elapsed(absence, video_seconds)
                                    close_absence(absence["record_id"], duration)

                                    print(
                                        f"[RETURNED] {employee_name} returned to seat {seat_id}. "
                                        f"Away duration: {format_duration(duration)}"
                                    )
                                    del active_absences[seat_id]
                                    return_candidates.pop(seat_id, None)

                        else:
                            # Clear return candidate because presence was momentary (passerby walked away)
                            return_candidates.pop(seat_id, None)

                            # Seat appears empty in this frame -> enter debounce state
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

                                # Debounce confirmed (empty for >= EMPTY_CONFIRM_SECONDS)
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

                                    print(f"[ABSENCE STARTED] {employee_name} ({seat_id}) absence confirmed.")
                                    empty_candidates.pop(seat_id, None)

                            # Check elapsed absence against alert threshold
                            if seat_id in active_absences:
                                absence = active_absences[seat_id]
                                elapsed = get_absence_elapsed(absence, video_seconds)

                                if elapsed >= ABSENCE_LIMIT_SECONDS and not absence["email_attempted"]:
                                    absence["email_attempted"] = True
                                    minutes_away = max(1, int(elapsed // 60))
                                    print(
                                        f"[ALERT] {employee_name} ({seat_id}) has not been in their seat for "
                                        f"{format_duration(elapsed)} (exceeded {format_duration(ABSENCE_LIMIT_SECONDS)} threshold)."
                                    )
                                    rm_target = sdata.get("rm_email") or DEFAULT_RM_EMAIL
                                    sent = send_absence_email(
                                        employee_name,
                                        seat_id,
                                        rm_target,
                                        absence["wall_start"],
                                        elapsed,
                                        area_name=area_title
                                    )
                                    if sent:
                                        mark_email_sent(absence["record_id"])

                # ========================================================
                # POST-BREAK ABSENCE AUDIT (Alert RM if not in seat after break)
                # ========================================================
                if is_active_shift:
                    post_break = get_post_break_status(
                        now_dt=datetime.now(),
                        floor=floor_id,
                        grace_seconds=POST_BREAK_GRACE_SECONDS,
                        window_minutes=25
                    )
                    if post_break["is_post_break"] and post_break["grace_expired"]:
                        b_tag = post_break["break_tag"]
                        for seat_id, sdata in seats.items():
                            emp_name = sdata.get("name", "Employee")
                            emp_id = sdata.get("employee_id", "Unknown")
                            rm_target = sdata.get("rm_email") or DEFAULT_RM_EMAIL

                            is_occupied = cached_seat_status.get(seat_id, {}).get("occupied", False)
                            # Alert if empty and not yet notified for this specific break
                            if not is_occupied and (seat_id, b_tag) not in alerted_post_breaks:
                                alerted_post_breaks.add((seat_id, b_tag))
                                print(
                                    f"[POST-BREAK ALERT] {emp_name} ({seat_id}) is not in their seat after "
                                    f"{post_break['break_name']} (ended at {post_break['break_end_time']})!"
                                )
                                send_post_break_absence_email(
                                    employee_name=emp_name,
                                    seat_id=seat_id,
                                    employee_id=emp_id,
                                    rm_email=rm_target,
                                    break_name=post_break["break_name"],
                                    break_end_time=post_break["break_end_time"],
                                    area_name=area_title
                                )

                # ========================================================
                # EMPLOYEE GATHERING TRACKING STATE MACHINE (Active Shifts Only)
                # ========================================================
                if is_active_shift:
                    current_clusters = detect_gathering_clusters(
                        cached_people,
                        distance_threshold=GATHERING_DISTANCE_THRESHOLD,
                        min_people=GATHERING_MIN_PEOPLE,
                        seats=seats
                    )

                    current_time_mono = time.monotonic()
                    matched_gathering_ids = set()

                    for cluster in current_clusters:
                        c_cx, c_cy = cluster["centroid"]
                        best_gid = None
                        best_dist = 9999.0

                        for gid, ginfo in active_gatherings.items():
                            gcx, gcy = ginfo["centroid"]
                            d = np.hypot(c_cx - gcx, c_cy - gcy)
                            if d < 140 and d < best_dist:
                                best_dist = d
                                best_gid = gid

                        analysis = analyze_gathering_members(
                            cluster["people_boxes"],
                            cluster["centroid"],
                            cached_faces,
                            seats,
                            default_rm_email=DEFAULT_RM_EMAIL
                        )

                        if best_gid:
                            g = active_gatherings[best_gid]
                            g["box"] = cluster["box"]
                            g["centroid"] = cluster["centroid"]
                            g["count"] = cluster["count"]
                            g["people_boxes"] = cluster["people_boxes"]
                            g["last_seen_wall"] = current_time_mono
                            g["last_seen_video"] = video_seconds
                            g["location_desc"] = analysis["location_desc"]
                            g["identified"] = analysis["identified"]
                            g["unidentified_count"] = analysis["unidentified_count"]
                            matched_gathering_ids.add(best_gid)
                        else:
                            gathering_counter += 1
                            gid = f"GATH-{gathering_counter:02d}"
                            rec_id = start_gathering_record(
                                gid,
                                analysis["location_desc"],
                                cluster["count"],
                                ", ".join([m["name"] for m in analysis["identified"]]) or "Unidentified",
                                VIDEO_SOURCE
                            )
                            active_gatherings[gid] = {
                                "id": gid,
                                "record_id": rec_id,
                                "box": cluster["box"],
                                "centroid": cluster["centroid"],
                                "count": cluster["count"],
                                "people_boxes": cluster["people_boxes"],
                                "start_time": datetime.now(),
                                "wall_start": current_time_mono,
                                "video_start": video_seconds,
                                "last_seen_wall": current_time_mono,
                                "last_seen_video": video_seconds,
                                "email_sent": False,
                                "email_attempted": False,
                                "location_desc": analysis["location_desc"],
                                "identified": analysis["identified"],
                                "unidentified_count": analysis["unidentified_count"]
                            }
                            matched_gathering_ids.add(gid)
                            print(
                                f"[GATHERING STARTED] {gid}: {cluster['count']} people gathered "
                                f"at {analysis['location_desc']}."
                            )

                    # Check dispersed gatherings
                    dispersed_gids = []
                    for gid, ginfo in active_gatherings.items():
                        if gid not in matched_gathering_ids:
                            unseen_dur = (
                                (video_seconds - ginfo["last_seen_video"])
                                if IS_RECORDED_VIDEO
                                else (current_time_mono - ginfo["last_seen_wall"])
                            )
                            if unseen_dur >= GATHERING_DEBOUNCE_SECONDS:
                                dur = (
                                    (ginfo["last_seen_video"] - ginfo["video_start"])
                                    if IS_RECORDED_VIDEO
                                    else (ginfo["last_seen_wall"] - ginfo["wall_start"])
                                )
                                dur = max(0, int(dur))
                                close_gathering_record(ginfo["record_id"], dur)
                                print(
                                    f"[GATHERING DISPERSED] {gid} ended. "
                                    f"Total gathered duration: {format_duration(dur)}."
                                )
                                dispersed_gids.append(gid)

                    for gid in dispersed_gids:
                        del active_gatherings[gid]

                    # Check duration & alert for active gatherings
                    for gid, ginfo in active_gatherings.items():
                        dur = (
                            (video_seconds - ginfo["video_start"])
                            if IS_RECORDED_VIDEO
                            else (current_time_mono - ginfo["wall_start"])
                        )
                        dur = max(0, int(dur))

                        if dur >= GATHERING_LIMIT_SECONDS and not ginfo["email_attempted"]:
                            ginfo["email_attempted"] = True
                            print(
                                f"\n[ALERT] GATHERING DETECTED ({gid}): {ginfo['count']} people gathered for "
                                f"{format_duration(dur)} at {ginfo['location_desc']} (Exceeded {format_duration(GATHERING_LIMIT_SECONDS)} threshold)!"
                            )

                            # Save snapshot with bounding box
                            snap_frame = frame.copy()
                            gx1, gy1, gx2, gy2 = ginfo["box"]
                            cv2.rectangle(snap_frame, (gx1, gy1), (gx2, gy2), (0, 0, 255), 3)
                            snap_name = f"gathering_alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{gid}.jpg"
                            snap_path = str(GATHERING_ALERTS_DIR / snap_name)
                            cv2.imwrite(snap_path, snap_frame)

                            # Group recipients by their respective RM email
                            rm_groups = defaultdict(list)
                            for member in ginfo["identified"]:
                                target_rm = member.get("rm_email") or DEFAULT_RM_EMAIL
                                rm_groups[target_rm].append(member)

                            # If no specific employee identified, notify DEFAULT_RM_EMAIL
                            if not rm_groups:
                                rm_groups[DEFAULT_RM_EMAIL] = []

                            # Dispatch email to each respective RM
                            any_sent = False
                            for rm_email, member_list in rm_groups.items():
                                sent = send_gathering_email(
                                    gathering_info=ginfo,
                                    rm_email=rm_email,
                                    relevant_members=member_list,
                                    duration=dur,
                                    snapshot_path=snap_path,
                                    area_name=area_title
                                )
                                if sent:
                                    any_sent = True

                            if any_sent:
                                ginfo["email_sent"] = True
                                mark_gathering_email_sent(ginfo["record_id"], snap_path)

            # ========================================================
            # VISUAL HUD RENDERING (SMOOTH & ANTI-FLICKER)
            # ========================================================
            display_frame = frame.copy()

            # 1. Draw detected people
            if show_detections_overlay:
                for p in cached_people:
                    px1, py1, px2, py2 = p
                    cv2.rectangle(display_frame, (px1, py1), (px2, py2), (80, 180, 80), 1)

                for f in cached_faces:
                    fx, fy, fw, fh = f["box"]
                    f_col = (0, 255, 0) if f["id"] != "Unknown" else (0, 200, 255)
                    cv2.rectangle(display_frame, (fx, fy), (fx + fw, fy + fh), f_col, 2)
                    flabel = f"{f['id']} ({f['sim']:.2f})" if f["id"] != "Unknown" else f"Face ({f['conf']:.2f})"
                    cv2.putText(
                        display_frame, flabel, (fx, max(15, fy - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, f_col, 1
                    )

            # 2. Draw active employee gatherings
            gath_alert_count = 0
            if show_gatherings_overlay:
                for gid, ginfo in active_gatherings.items():
                    gx1, gy1, gx2, gy2 = ginfo["box"]
                    g_dur = (
                        (video_seconds - ginfo["video_start"])
                        if IS_RECORDED_VIDEO
                        else (current_time_mono - ginfo["wall_start"])
                    )
                    g_dur = max(0, int(g_dur))
                    is_gath_alert = g_dur >= GATHERING_LIMIT_SECONDS

                    if is_gath_alert:
                        gath_alert_count += 1
                        g_color = (0, 0, 255)       # Bright Red: Alert threshold exceeded (> 5 min)
                        g_txt = f"ALERT GATHERING: {ginfo['count']} People [{format_duration(g_dur)}]"
                    else:
                        g_color = (0, 165, 255)     # Amber / Orange: Gathering accumulating
                        g_txt = f"GATHERING: {ginfo['count']} People [{format_duration(g_dur)}]"

                    # Enclosing box around the whole cluster
                    cv2.rectangle(display_frame, (gx1, gy1), (gx2, gy2), g_color, 2)
                    # Badge above box
                    (gtw, gth), _ = cv2.getTextSize(g_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                    gbadge_y = max(18, gy1)
                    cv2.rectangle(
                        display_frame,
                        (gx1, gbadge_y - gth - 6),
                        (gx1 + gtw + 8, gbadge_y),
                        (20, 20, 20),
                        -1
                    )
                    cv2.putText(
                        display_frame,
                        g_txt,
                        (gx1 + 4, gbadge_y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        g_color,
                        1
                    )

            # 3. Draw workstation seats
            occupied_count = 0
            empty_count = 0
            alert_count = 0

            if show_seats_overlay:
                for seat_id, sdata in seats.items():
                    sx1, sy1, sx2, sy2 = sdata["seat"]
                    emp_name = sdata["name"]
                    sinfo = cached_seat_status.get(seat_id, {"occupied": False, "verification": "EMPTY"})
                    is_occ = sinfo["occupied"]
                    is_absent = seat_id in active_absences

                    if is_absent:
                        # Confirmed Absence State - Keep counting continuously without interruption!
                        empty_count += 1
                        is_away_alert = False
                        elapsed = get_absence_elapsed(active_absences[seat_id], video_seconds)
                        away_text = f" | Away: {format_duration(elapsed)}"

                        if elapsed >= ABSENCE_LIMIT_SECONDS:
                            is_away_alert = True
                            alert_count += 1
                            color = (0, 0, 255)       # Bright Red: Limit exceeded (e.g. > 1 min)
                            status_txt = f"{seat_id}: {emp_name} [ALERT{away_text}]"
                        else:
                            color = (60, 60, 230)     # Red: Confirmed Empty
                            status_txt = f"{seat_id}: {emp_name} [EMPTY{away_text}]"

                    elif is_occ:
                        # Occupied State (confirmed seated presence)
                        occupied_count += 1
                        if sinfo["verification"].startswith("VERIFIED"):
                            color = (0, 220, 0)       # Green: Verified employee
                            status_txt = f"{seat_id}: {emp_name} [VERIFIED]"
                        elif "DIFFERENT" in sinfo["verification"]:
                            color = (255, 180, 0)     # Cyan/Orange: Different employee
                            status_txt = f"{seat_id}: {emp_name} [{sinfo['verification']}]"
                        else:
                            color = (0, 200, 100)     # Soft Green: Person / Face present
                            status_txt = f"{seat_id}: {emp_name} [OCCUPIED]"

                    else:
                        # Grace / Debounce State (0 to 5 seconds)
                        color = (0, 215, 255)         # Warm Amber / Yellow
                        status_txt = f"{seat_id}: {emp_name}"

                    # Draw seat rectangle
                    thickness = 3 if (is_occ and not is_absent) else 2
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

            # 4. Top HUD Status Bar (Dual-Deck Clean Layout)
            now_t = time.time()
            if now_t - prev_time > 0:
                display_fps = 1.0 / (now_t - prev_time)
            prev_time = now_t

            # Banner background
            cv2.rectangle(display_frame, (0, 0), (1280, 50), (18, 18, 18), -1)
            cv2.line(display_frame, (0, 50), (1280, 50), (60, 60, 60), 1)

            # Row 1: Current Date/Time + Schedule Status Pill Badge
            now_dt_str = datetime.now().strftime("%a %b %d, %I:%M %p")
            cv2.putText(
                display_frame,
                f"{area_title}  |  {now_dt_str}",
                (15, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (210, 210, 210),
                1
            )

            # Schedule Pill Badge
            sched_tag = f"[{sched['badge_text']}]"
            cv2.putText(
                display_frame,
                sched_tag,
                (550, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                sched["badge_color"],
                2
            )

            banner_top_right = (
                f"FPS: {display_fps:.1f} | Frame: {frame_number} | "
                f"Schedule: {'ON' if ENFORCE_SCHEDULE else 'OVERRIDE'}"
            )
            cv2.putText(
                display_frame,
                banner_top_right,
                (920, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (170, 170, 170),
                1
            )

            # Row 2: Live Operational Metrics or Break/Off-hours notice
            if is_off_hours:
                row2_txt = f"STANDBY: Outside Office Hours ({sched['reason']})  |  Next Shift: Mon-Fri 09:30 AM, Sat 09:30 AM"
                row2_col = (160, 160, 160)
            elif is_tea_break:
                row2_txt = f"AUTHORIZED BREAK: {sched['reason']}  |  Absence & Gathering Tracking Frozen  |  No Email Alerts"
                row2_col = (0, 165, 255)
            else:
                row2_txt = (
                    f"Seats: {len(seats)} [Occupied: {occupied_count} | Empty: {empty_count}]  |  "
                    f"Absences: {alert_count}  |  Gatherings: {len(active_gatherings)}  |  "
                    f"People: {len(cached_people)}"
                )
                row2_col = (0, 255, 255) if (alert_count == 0 and gath_alert_count == 0) else (0, 100, 255)

            cv2.putText(
                display_frame,
                row2_txt,
                (15, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.47,
                row2_col,
                1
            )

            # Off-Hours Standby Center Card
            if is_off_hours:
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (240, 250), (1040, 470), (15, 15, 15), -1)
                cv2.addWeighted(overlay, 0.78, display_frame, 0.22, 0, display_frame)
                cv2.rectangle(display_frame, (240, 250), (1040, 470), (80, 80, 180), 2)
                cv2.putText(
                    display_frame,
                    "OFFICE OPERATING HOURS CLOSED",
                    (330, 310),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.80,
                    (255, 255, 255),
                    2
                )
                cv2.putText(
                    display_frame,
                    "Mon - Fri: 09:30 AM - 06:30 PM   |   Saturday: 09:30 AM - 01:30 PM",
                    (290, 360),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.54,
                    (0, 215, 255),
                    1
                )
                cv2.putText(
                    display_frame,
                    "Camera monitoring on standby. Absence alerts resume at start of next shift.",
                    (310, 405),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (190, 190, 190),
                    1
                )
                cv2.putText(
                    display_frame,
                    "(Press 'T' on keyboard to toggle schedule enforcement for testing)",
                    (360, 442),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    (130, 130, 130),
                    1
                )

            if paused:
                cv2.putText(
                    display_frame,
                    "[PAUSED - Press 'P' to resume]",
                    (520, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

            cv2.imshow(window_title, display_frame)

            # Window close detection (user clicked 'X' button on the window)
            try:
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                    print("[INFO] Window closed by user.")
                    break
            except Exception:
                pass

            # Key controls
            wait_time = 30 if paused else 1
            key = cv2.waitKey(wait_time) & 0xFF
            if key in [ord("q"), ord("Q"), 27]:
                print("[INFO] Quitting monitoring...")
                break
            elif key in [ord("p"), ord("P")]:
                paused = not paused
                print(f"[INFO] Monitoring {'paused' if paused else 'resumed'}.")
            elif key in [ord("c"), ord("C")]:
                show_seats_overlay = not show_seats_overlay
            elif key in [ord("g"), ord("G")]:
                show_gatherings_overlay = not show_gatherings_overlay
                print(f"[INFO] Gathering overlay {'enabled' if show_gatherings_overlay else 'disabled'}.")
            elif key in [ord("t"), ord("T")]:
                ENFORCE_SCHEDULE = not ENFORCE_SCHEDULE
                print(f"[INFO] Schedule enforcement toggled: {'ENABLED' if ENFORCE_SCHEDULE else 'DISABLED (Override Mode)'}.")
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
