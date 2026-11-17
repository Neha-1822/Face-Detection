# Workplace Monitoring System

An end-to-end intelligent computer vision platform for real-time **Workstation Seat Monitoring**, **Person Detection**, **Face Detection**, and **Employee Face Recognition** using edge CCTV camera feeds.

The system continuously tracks workstation occupancy, verifies whether the assigned employee is present at their desk, debounces temporary movements, logs absence intervals to an SQLite database, and triggers management alerts when an absence exceeds configurable thresholds.

---

## Key Features

- **Multi-Model Neural Pipeline**:
  - **YOLO11 (`yolo11n.pt`)**: Real-time person detection robust to posture changes, occlusions, and employees with backs turned to the camera.
  - **YuNet (`face_detection_yunet_2023mar.onnx`)**: High-performance, lightweight face detection optimized for low-resolution and high-angle CCTV video.
  - **SFace (`face_recognition_sface_2021dec.onnx`)**: Deep facial feature extraction and cosine similarity matching against enrolled employee templates.
- **Intelligent Workstation Seat Monitoring**:
  - Real-time seat occupancy detection matching detected persons and faces to predefined desk zones.
  - **Tri-State Seat Status**:
    - `VERIFIED`: Confirms the assigned employee is present at their workstation.
    - `DIFFERENT`: Flags another enrolled employee sitting in the seat.
    - `OCCUPIED (PERSON)`: Detects human presence at the desk when face is not turned toward the ceiling camera.
    - `EMPTY`: Workstation is vacant.
- **Automated Absence Tracking & Debouncing**:
  - `EMPTY_CONFIRM_SECONDS` (default: 5s) prevents false alarms when someone briefly reaches down or leans over.
  - Absence intervals automatically recorded in `seat_tracking.db`.
  - Configurable alert threshold (`ABSENCE_LIMIT_SECONDS`, default: 10 mins).
  - Optional SMTP automated email alerts to reporting managers.
- **Visual Seat Calibration Tool**:
  - Built-in interactive GUI (`calibrate_seats.py`) allowing administrators to click and drag to define, resize, or delete seat zones directly on the camera view.

---

## Architecture Flow

```mermaid
flowchart TD
    VideoSource["CCTV Camera Feed / Video Stream<br/>(RTSP, AVI, MP4, Webcam)"] --> Preprocess["Frame Resize & Standardization<br/>(1280 x 720)"]
    
    Preprocess --> YOLO["YOLO11 Person Detector<br/>(Human Body Bounding Boxes)"]
    Preprocess --> YuNet["YuNet Face Detector<br/>(Face Bounding Boxes & Landmarks)"]
    
    YuNet --> SFace["SFace Face Recognizer<br/>(128-D Feature Embedding)"]
    SFace --> MatchDB["Cosine Similarity Match<br/>(employee_embeddings.npz)"]
    
    YOLO --> SeatLogic["Seat Occupancy & Assignment Engine<br/>(seat_logic.py)"]
    MatchDB --> SeatLogic
    
    SeatLogic --> StatusEval{"Seat Occupancy State"}
    
    StatusEval -- "Person Present + Face Verified" --> VerifiedState["Status: OCCUPIED (VERIFIED)<br/>Color: Green"]
    StatusEval -- "Person Present (No Face / Other)" --> OccupiedState["Status: OCCUPIED (PERSON)<br/>Color: Cyan / Yellow"]
    
    StatusEval -- "Seat Vacant" --> Debounce{"Empty >= 5 Seconds?"}
    Debounce -- "No" --> CandTimer["Candidate Timer Active"]
    Debounce -- "Yes" --> AbsenceStart["Absence Logged in SQLite<br/>(seat_tracking.db)"]
    
    AbsenceStart --> LimitCheck{"Away >= 10 Minutes?"}
    LimitCheck -- "Yes" --> AlertTrigger["Manager Alert (Red Alert / Email)"]
    LimitCheck -- "No" --> AwayTimer["Away Timer Running (Red Box)"]
    
    VerifiedState & OccupiedState --> ReturnCheck{"Returning from Absence?"}
    ReturnCheck -- "Yes" --> CloseAbsence["Close Absence in Database<br/>Record Total Away Duration"]
```

---

## Directory Structure

```text
Workplace_Monitoring/
│
├── monitoring/
│   ├── seat_monitor.py            # Main real-time seat monitoring application
│   ├── seat_logic.py              # Unique person-to-seat assignment & verification logic
│   ├── calibrate_seats.py         # Interactive GUI tool to draw and configure seat zones
│   ├── save_default_seats.py      # Script to save/reset default seat definitions
│   └── seats_3rd_floor.json       # Workstation coordinates & employee mapping
│
├── detection/
│   ├── person_detection.py        # Standalone YOLO person detector module
│   ├── face_detection.py          # Standalone YuNet face detector module
│   ├── convert_test_video.py      # Video remuxing/conversion utility
│   └── reports/                   # Converted sample video clips (.mp4)
│
├── recognition/
│   ├── enroll_employees.py        # Generates employee_embeddings.npz from dataset
│   ├── inspect_employee_dataset.py# Validates & audits face dataset quality
│   ├── test_recognition.py        # Standalone SFace face recognition test
│   ├── capture_unknown_faces.py   # Captures and logs unrecognized faces
│   └── output/
│       ├── employee_embeddings.npz # Normalized employee biometric templates
│       └── enrollment_report.json  # Audit report of enrolled identities
│
├── models/
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
│
├── dataset/                       # Employee image directories (e.g. KA001, KA019, etc.)
├── seat_tracking.db               # SQLite database recording employee absences
├── yolo11n.pt                     # YOLO11 neural network weights
├── requirement.txt                # Python package dependencies
├── requirements.txt               # Standard pip requirements file
├── .env.example                   # Email alerts environment configuration template
├── .gitignore                     # Git ignore rules for videos, models & credentials
└── README.md
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10 or higher
- Git (optional, for version control)
- Webcam, RTSP camera stream URL, or recorded CCTV video file (`.avi` / `.mp4`)

### 2. Clone the Repository
```bash
git clone https://github.com/Neha-1822/Face-Detection.git
cd Face-Detection
```

### 3. Create & Activate a Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage Guide

### Step 1: Employee Dataset & Face Enrollment

1. **Organize your dataset**: Place employee photos in `dataset/<EMPLOYEE_ID>/` (e.g., `dataset/KA019/photo1.jpg`).
2. **Audit dataset quality**:
   ```powershell
   python recognition/inspect_employee_dataset.py
   ```
3. **Enroll employees & generate embeddings**:
   ```powershell
   python recognition/enroll_employees.py
   ```
   *This extracts 128-D facial feature embeddings, averages them per employee, normalizes vectors, and saves `recognition/output/employee_embeddings.npz`.*

---

### Step 2: Calibrate Workstation Seats (Optional)

If your camera angle changes or you want to add/modify workstation desk zones:
```powershell
python monitoring/calibrate_seats.py
```
- **Left Click & Drag**: Draw a new workstation bounding box.
- **Terminal Prompt**: Enter Seat ID (`DESK-DSK01`), Assigned Name, and Enrolled ID (`KA001`).
- **Right Click**: Select an existing seat box.
- **`S`**: Save coordinates to `monitoring/seats_3rd_floor.json`.
- **`D`**: Delete selected seat.
- **`Q`**: Exit calibration.

---

### Step 3: Run Seat Monitoring

To start the continuous seat monitoring system:
```powershell
python monitoring/seat_monitor.py
```

#### Run Workstation Monitoring by Floor:
```powershell
# 3rd Floor Main Workstations (28 Desks):
python monitoring/seat_monitor.py --floor 3 --source "3rd Floor WS.avi"

# 3rd Floor Workstation Area 2 (9 Desks):
python monitoring/seat_monitor.py --floor 3-area2 --source "path/to/3rd_floor_area2.mp4"

# 4th Floor Workstations (24 Desks):
python monitoring/seat_monitor.py --floor 4 --source "path/to/4th_floor_ws.mp4"
```

#### Run Cafeteria & Lounge Monitor:
```powershell
python monitoring/cafeteria_monitor.py --source "rtsp://admin:password@CAFETERIA_IP:554/profile2/media.smp"
```

#### Office Break Schedules & Post-Break Alerts:
- **3rd Floor Breaks**: Morning `11:15 - 11:30 AM` | Lunch `01:00 - 02:00 PM` | Evening `04:15 - 04:30 PM`
- **4th Floor Breaks**: Morning `11:00 - 11:15 AM` | Lunch `01:00 - 02:00 PM` | Evening `04:00 - 04:15 PM`
- **Break Timer Freezing**: All breaks are paused and do NOT count against the 10-minute absence limit.
- **Post-Break Alerts**: Automatically triggers an email notification to the Reporting Manager (RM) if:
  1. An employee is **not back in their seat** after any break ends (Morning, Lunch, or Evening).
  2. An employee is **seen in the cafeteria** after their floor's break ends.
- **Continuous 10-Minute Absence Tracking**: Resumes during active work hours before and after all breaks.

#### Run with Custom Video Sources or Live Camera:
```powershell
# Run with a specific video file:
python monitoring/seat_monitor.py --source "3rd_floor_full.mp4"

# Run with a live webcam:
python monitoring/seat_monitor.py --source 0

# Run with a live RTSP IP camera:
python monitoring/seat_monitor.py --source "rtsp://username:password@192.168.1.100:554/stream1"
```

> [!NOTE]
> The monitor runs **continuously in an infinite loop** and will **never exit automatically**. For video files, it seamlessly loops playback and maintains ongoing absence timers. For live RTSP streams, it automatically reconnects on network interruptions. It will only stop when you explicitly exit.

#### Controls During Monitoring:
| Key | Action |
| :---: | :--- |
| **`Q` / `ESC`** | Quit monitoring and close session |
| **Window `[X]`** | Close window and exit session |
| **`P`** | Pause / resume video playback |
| **`S`** | Save current frame snapshot with bounding box overlays |
| **`C`** | Toggle workstation seat rectangles on/off |
| **`D`** | Toggle person/face detection boxes on/off |

---

### Step 4: Standalone Testing & Utilities

- **Test Person Detection (YOLO11)**:
  ```powershell
  python detection/person_detection.py
  ```
- **Test Face Detection (YuNet)**:
  ```powershell
  python detection/face_detection.py
  ```
- **Test Face Recognition (SFace)**:
  ```powershell
  python recognition/test_recognition.py
  ```
- **Capture Unknown Faces**:
  ```powershell
  python recognition/capture_unknown_faces.py
  ```

---

## Configuration

### Workstation Seats (`monitoring/seats_3rd_floor.json`)
Each workstation is represented as:
```json
"DESK-DSK13": {
    "name": "Employee-13",
    "employee_id": "KA036",
    "rm_email": "neha@kristellar.com",
    "seat": [941, 100, 996, 156]
}
```

### Video Source & Parameters (`monitoring/seat_monitor.py`)
- **`VIDEO_SOURCE`**: Path to video file (`3rd Floor WS.avi`), webcam index (`0`), or RTSP stream (`rtsp://user:pass@camera_ip:554/stream`).
- **`ABSENCE_LIMIT_SECONDS`**: Duration (in seconds) before absence triggers an alert (default: `600` = 10 minutes).
- **`EMPTY_CONFIRM_SECONDS`**: Debounce time before starting an absence record (default: `5` seconds).
- **`DETECTION_INTERVAL`**: Run neural models every `N` frames to maintain high processing FPS.

### Email Alerts Setup (10-Minute Absence Notification)
The system automatically sends an email to the Reporting Manager (`rm_email`) when an employee is absent for 10 minutes:
1. Copy `.env.example` to `.env`:
   ```powershell
   Copy-Item .env.example .env
   ```
2. Enter your sender email and Gmail App Password:
   ```env
   ENABLE_EMAIL_ALERTS=true
   SENDER_EMAIL=your_alerts_email@gmail.com
   SENDER_APP_PASSWORD=your_16_character_app_password
   DEFAULT_RM_EMAIL=neha@kristellar.com
   ABSENCE_LIMIT_SECONDS=600
   ```
3. If credentials are not yet entered, the system will run in **Simulation Mode** (logging the full email content and recipient details directly to the console). Once credentials are added to `.env`, real emails are dispatched immediately via TLS SMTP.

---

## Database Schema (`seat_tracking.db`)

Absence events are recorded in the `absences` table:

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | `INTEGER` | Primary key (auto-increment) |
| `seat_id` | `TEXT` | Workstation ID (e.g., `DESK-DSK13`) |
| `employee_name` | `TEXT` | Assigned employee name |
| `absence_start` | `TEXT` | ISO 8601 start timestamp |
| `absence_end` | `TEXT` | ISO 8601 end timestamp (when employee returns) |
| `duration_seconds`| `INTEGER` | Total away duration in seconds |
| `email_sent` | `INTEGER` | `1` if alert email was sent, else `0` |
| `source` | `TEXT` | Video file name or camera source stream |
| `verified_face` | `TEXT` | Biometric ID verified at the seat |

---

## Models Used

| Model | Architecture | File Size | Task |
| :--- | :--- | :--- | :--- |
| **YOLO11 Nano** | Ultralytics YOLO11n | ~5.6 MB | Human person detection & seat occupancy |
| **YuNet** | CNN Face Detector | ~232 KB | Micro-face detection (works on small CCTV faces) |
| **SFace** | ResNet-based Deep CNN | ~38.6 MB | 128-D facial feature embedding & recognition |

---

## License

This project is licensed for internal workplace monitoring and security analytics.
