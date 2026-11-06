import cv2
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SEATS_3RD_FLOOR = {
    "WS-01": {
        "name": "Employee-01",
        "employee_id": "KA019",
        "email": "emp01@kristellar.com",
        "rm_email": "manager@kristellar.com",
        "seat": [320, 160, 560, 420],
        "desc": "Left Workstation - Front"
    },
    "WS-02": {
        "name": "Employee-02",
        "employee_id": "KA012",
        "email": "emp02@kristellar.com",
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

# Save default config file
config_path = BASE_DIR / "monitoring" / "seats_3rd_floor.json"
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(SEATS_3RD_FLOOR, f, indent=4)
print(f"Saved seat configuration to {config_path}")

img = cv2.imread(str(BASE_DIR / "test_3rd_floor_f0.jpg"))
if img is not None:
    vis = img.copy()
    for sid, sinfo in SEATS_3RD_FLOOR.items():
        x1, y1, x2, y2 = sinfo["seat"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 215, 255), 2)
        cv2.putText(vis, f"{sid}: {sinfo['name']}", (x1 + 3, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    preview_path = BASE_DIR / "monitoring" / "refined_seats_preview.jpg"
    cv2.imwrite(str(preview_path), vis)
    print(f"Saved preview to {preview_path}")
