import cv2
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# User's exact 18 red bounding boxes extracted from media_1793947534819_ae872404.jpg
scale_x = 1280.0 / 1024.0
scale_y = 720.0 / 575.0
y_offset = 3

user_boxes = [
    ("DESK-DSK01", (248, 88, 360, 149), "Staff", "KA001"),
    ("DESK-DSK02", (365, 50, 442, 111), "Aishwarya", "KA028"),
    ("DESK-DSK03", (365, 105, 444, 227), "Avilasha", "KA034"),
    ("DESK-DSK04", (240, 384, 440, 512), "Employee-04", "KA028"),
    ("DESK-DSK05", (425, 256, 616, 370), "Employee-05", "KA027"),
    ("DESK-DSK06", (622, 306, 841, 496), "Employee-06", "KA032"),
    ("DESK-DSK07", (452, 81, 527, 177), "Employee-07", "KA035"),
    ("DESK-DSK08", (529, 36, 574, 81), "Employee-08", "KA037"),
    ("DESK-DSK09", (580, 18, 629, 46), "Employee-09", "KA041"),
    ("DESK-DSK10", (583, 47, 621, 118), "Employee-10", "KA042"),
    ("DESK-DSK11", (634, 31, 685, 73), "Employee-11", "KA043"),
    ("DESK-DSK12", (705, 106, 766, 192), "Employee-12", "KA044"),
    ("DESK-DSK13", (753, 76, 797, 122), "Employee-13", "KA036"),
    ("DESK-DSK14", (810, 68, 849, 102), "Employee-14", "KA045"),
    ("DESK-DSK15", (843, 108, 899, 150), "Employee-15", "KA026"),
    ("DESK-DSK16", (826, 150, 897, 197), "Employee-16", "KA046"),
    ("DESK-DSK17", (872, 70, 900, 105), "Employee-17", "KA051"),
    ("DESK-DSK18", (892, 48, 922, 84), "Employee-18", "KA052")
]

exact_18_seats = {}
for sid, box, name, emp_id in user_boxes:
    x1, y1, x2, y2 = box
    sx1 = int(round(x1 * scale_x))
    sy1 = int(round((y1 + y_offset) * scale_y))
    sx2 = int(round(x2 * scale_x))
    sy2 = int(round((y2 + y_offset) * scale_y))
    exact_18_seats[sid] = {
        "name": name,
        "employee_id": emp_id,
        "rm_email": "neha@kristellar.com",
        "seat": [sx1, sy1, sx2, sy2]
    }

json_path = BASE_DIR / "monitoring" / "seats_3rd_floor.json"
raw = json.dumps(exact_18_seats, indent=4)
compact = re.sub(r'"seat":\s*\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\]', r'"seat": [\1, \2, \3, \4]', raw)
with open(json_path, "w", encoding="utf-8") as f:
    f.write(compact)
print(f"[SUCCESS] Saved {len(exact_18_seats)} user-specified seats to {json_path}")

# Draw preview on video frame
cap = cv2.VideoCapture(str(BASE_DIR / "3rd Floor WS.avi"))
ret, frame = cap.read()
cap.release()

if ret:
    vis = frame.copy()
    for sid, sdata in exact_18_seats.items():
        sx1, sy1, sx2, sy2 = sdata["seat"]
        # Draw red box matching user's red box
        cv2.rectangle(vis, (sx1, sy1), (sx2, sy2), (0, 0, 255), 2)
        label = f"{sid}: {sdata['name']}"
        cv2.putText(vis, label, (sx1, max(18, sy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    preview_path = BASE_DIR / "monitoring" / "exact_user_seats_preview.jpg"
    cv2.imwrite(str(preview_path), vis)
    print(f"[SUCCESS] Saved preview to {preview_path}")

