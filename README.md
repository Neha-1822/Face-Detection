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
- **Terminal Prompt**: Enter Seat ID (`WS-01`), Assigned Name, and Enrolled ID (`KA019`).
- **Right Click**: Select an existing seat box.
- **`S`**: Save coordinates to `monitoring/seats_3rd_floor.json`.
- **`D`**: Delete selected seat.
- **`Q`**: Exit calibration.

---

### Step 3: Run Seat Monitoring

To start the seat monitoring system:
```powershell
python monitoring/seat_monitor.py
```

#### Hotkey Controls During Monitoring:
| Key | Action |
| :---: | :--- |
| **`Q` / `ESC`** | Quit application and release camera |
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
"WS-13": {
    "name": "Employee-13",
    "employee_id": "KA036",
    "email": "emp13@kristellar.com",
    "rm_email": "manager@kristellar.com",
    "seat": [941, 100, 996, 156],
    "desc": "Right Mid Desk (Blue Shirt)"
}
```

### Video Source & Parameters (`monitoring/seat_monitor.py`)
- **`VIDEO_SOURCE`**: Path to video file (`3rd Floor WS.avi`), webcam index (`0`), or RTSP stream (`rtsp://user:pass@camera_ip:554/stream`).
- **`ABSENCE_LIMIT_SECONDS`**: Duration (in seconds) before absence triggers an alert (default: `600` = 10 minutes).
- **`EMPTY_CONFIRM_SECONDS`**: Debounce time before starting an absence record (default: `5` seconds).
- **`DETECTION_INTERVAL`**: Run neural models every `N` frames to maintain high processing FPS.

### Email Alerts Setup
To enable email alerts when an employee's absence exceeds the threshold:
1. Copy `.env.example` to `.env`:
   ```powershell
   cp .env.example .env
   ```
2. Enter your sender email and application password:
   ```env
   SENDER_EMAIL=your_alerts_email@gmail.com
   SENDER_APP_PASSWORD=your_16_character_app_password
   ```
3. Set `ENABLE_EMAIL_ALERTS = True` in `monitoring/seat_monitor.py`.

---

## Database Schema (`seat_tracking.db`)

Absence events are recorded in the `absences` table:

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | `INTEGER` | Primary key (auto-increment) |
| `seat_id` | `TEXT` | Workstation ID (e.g., `WS-13`) |
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
