import cv2
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# 28 Workstation Desks on 3rd Floor
SEATS_3RD_FLOOR = {
    "DESK-DSK01": {
        "name": "Staff",
        "employee_id": "KA001",
        "rm_email": "neha@kristellar.com",
        "seat": [310, 114, 450, 190]
    },
    "DESK-DSK02": {
        "name": "Aishwarya",
        "employee_id": "KA028",
        "rm_email": "neha@kristellar.com",
        "seat": [456, 66, 552, 143]
    },
    "DESK-DSK03": {
        "name": "Avilasha",
        "employee_id": "KA034",
        "rm_email": "neha@kristellar.com",
        "seat": [456, 135, 555, 288]
    },
    "DESK-DSK04": {
        "name": "Employee-04",
        "employee_id": "KA028",
        "rm_email": "neha@kristellar.com",
        "seat": [300, 485, 550, 645]
    },
    "DESK-DSK05": {
        "name": "Employee-05",
        "employee_id": "KA027",
        "rm_email": "neha@kristellar.com",
        "seat": [531, 324, 770, 467]
    },
    "DESK-DSK06": {
        "name": "Employee-06",
        "employee_id": "KA032",
        "rm_email": "neha@kristellar.com",
        "seat": [778, 387, 1051, 625]
    },
    "DESK-DSK07": {
        "name": "Employee-07",
        "employee_id": "KA035",
        "rm_email": "neha@kristellar.com",
        "seat": [565, 105, 659, 225]
    },
    "DESK-DSK08": {
        "name": "Employee-08",
        "employee_id": "KA037",
        "rm_email": "neha@kristellar.com",
        "seat": [661, 49, 718, 105]
    },
    "DESK-DSK09": {
        "name": "Employee-09",
        "employee_id": "KA041",
        "rm_email": "neha@kristellar.com",
        "seat": [725, 26, 786, 61]
    },
    "DESK-DSK10": {
        "name": "Employee-10",
        "employee_id": "KA042",
        "rm_email": "neha@kristellar.com",
        "seat": [729, 63, 776, 152]
    },
    "DESK-DSK11": {
        "name": "Employee-11",
        "employee_id": "KA043",
        "rm_email": "neha@kristellar.com",
        "seat": [792, 43, 856, 95]
    },
    "DESK-DSK12": {
        "name": "Employee-12",
        "employee_id": "KA044",
        "rm_email": "neha@kristellar.com",
        "seat": [881, 136, 958, 244]
    },
    "DESK-DSK13": {
        "name": "Employee-13",
        "employee_id": "KA036",
        "rm_email": "neha@kristellar.com",
        "seat": [941, 99, 996, 157]
    },
    "DESK-DSK14": {
        "name": "Employee-14",
        "employee_id": "KA045",
        "rm_email": "neha@kristellar.com",
        "seat": [1012, 89, 1061, 131]
    },
    "DESK-DSK15": {
        "name": "Employee-15",
        "employee_id": "KA026",
        "rm_email": "neha@kristellar.com",
        "seat": [1054, 139, 1124, 192]
    },
    "DESK-DSK16": {
        "name": "Employee-16",
        "employee_id": "KA046",
        "rm_email": "neha@kristellar.com",
        "seat": [1032, 192, 1121, 250]
    },
    "DESK-DSK17": {
        "name": "Employee-17",
        "employee_id": "KA051",
        "rm_email": "neha@kristellar.com",
        "seat": [1090, 91, 1125, 135]
    },
    "DESK-DSK18": {
        "name": "Employee-18",
        "employee_id": "KA052",
        "rm_email": "neha@kristellar.com",
        "seat": [1115, 64, 1152, 109]
    },
    "DESK-DSK19": {
        "name": "Employee-19",
        "employee_id": "KA041",
        "rm_email": "neha@kristellar.com",
        "seat": [801, 80, 880, 174]
    },
    "DESK-DSK20": {
        "name": "Employee-20",
        "employee_id": "KA042",
        "rm_email": "neha@kristellar.com",
        "seat": [801, 175, 880, 310]
    },
    "DESK-DSK21": {
        "name": "Employee-21",
        "employee_id": "KA043",
        "rm_email": "neha@kristellar.com",
        "seat": [881, 80, 940, 174]
    },
    "DESK-DSK22": {
        "name": "Employee-22",
        "employee_id": "KA044",
        "rm_email": "neha@kristellar.com",
        "seat": [881, 175, 940, 310]
    },
    "DESK-DSK23": {
        "name": "Employee-23",
        "employee_id": "KA001",
        "rm_email": "neha@kristellar.com",
        "seat": [200, 485, 300, 645]
    },
    "DESK-DSK24": {
        "name": "Employee-24",
        "employee_id": "KA002",
        "rm_email": "neha@kristellar.com",
        "seat": [200, 350, 310, 484]
    },
    "DESK-DSK25": {
        "name": "Employee-25",
        "employee_id": "KA003",
        "rm_email": "neha@kristellar.com",
        "seat": [311, 191, 455, 320]
    },
    "DESK-DSK26": {
        "name": "Employee-26",
        "employee_id": "KA005",
        "rm_email": "neha@kristellar.com",
        "seat": [997, 136, 1053, 220]
    },
    "DESK-DSK27": {
        "name": "Employee-27",
        "employee_id": "KA007",
        "rm_email": "neha@kristellar.com",
        "seat": [1126, 91, 1175, 155]
    },
    "DESK-DSK28": {
        "name": "Employee-28",
        "employee_id": "KA016",
        "rm_email": "neha@kristellar.com",
        "seat": [1153, 64, 1220, 120]
    }
}

# Save default config file
config_path = BASE_DIR / "monitoring" / "seats_3rd_floor.json"
raw = json.dumps(SEATS_3RD_FLOOR, indent=4)
compact = re.sub(r'"seat":\s*\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\]', r'"seat": [\1, \2, \3, \4]', raw)
with open(config_path, "w", encoding="utf-8") as f:
    f.write(compact)
print(f"[SUCCESS] Saved {len(SEATS_3RD_FLOOR)} workstation desks to {config_path}")

test_img = BASE_DIR / "test_3rd_floor_f0.jpg"
if test_img.exists():
    img = cv2.imread(str(test_img))
    if img is not None:
        vis = img.copy()
        for sid, sinfo in SEATS_3RD_FLOOR.items():
            x1, y1, x2, y2 = sinfo["seat"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 215, 255), 2)
            cv2.putText(vis, f"{sid}: {sinfo['name']}", (x1 + 3, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        preview_path = BASE_DIR / "monitoring" / "refined_seats_preview.jpg"
        cv2.imwrite(str(preview_path), vis)
        print(f"Saved preview to {preview_path}")
