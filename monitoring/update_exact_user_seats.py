import cv2
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# User's exact 18 red bounding boxes extracted from media_1793947534819_ae872404.jpg
scale_x = 1280.0 / 1024.0
scale_y = 720.0 / 575.0
y_offset = 3

user_boxes = [
    ("WS-01", (248, 88, 360, 149), "Left Counter / Desk", "Staff", ""),
    ("WS-02", (365, 50, 442, 111), "Left Back Workstation", "Rahul", "KA012"),
    ("WS-03", (365, 105, 444, 227), "Left Front (Woman sitting)", "Neha", "KA019"),
    ("WS-04", (240, 384, 440, 512), "Foreground Left Front", "Employee-04", "KA028"),
    ("WS-05", (425, 256, 616, 370), "Foreground Center Back (Blue Jacket)", "Employee-05", "KA027"),
    ("WS-06", (622, 306, 841, 496), "Foreground Right Front (Laptop & Cushion)", "Employee-06", "KA032"),
    ("WS-07", (452, 81, 527, 177), "Center Mid-Left Desk", "Employee-07", "KA035"),
    ("WS-08", (529, 36, 574, 81), "Center Deep-Back Left", "Employee-08", "KA037"),
    ("WS-09", (580, 18, 629, 46), "Center Far-Back Wall Left", "Employee-09", "KA041"),
    ("WS-10", (583, 47, 621, 118), "Center Far-Back Wall Right", "Employee-10", "KA042"),
    ("WS-11", (634, 31, 685, 73), "Center Far-Back Right (Under IATC)", "Employee-11", "KA043"),
    ("WS-12", (705, 106, 766, 192), "Right Mid Desk (Khaki Shirt)", "Employee-12", "KA044"),
    ("WS-13", (753, 76, 797, 122), "Right Mid Desk (Blue Shirt)", "Employee-13", "KA036"),
    ("WS-14", (810, 68, 849, 102), "Right Back Desk - Top Left", "Employee-14", "KA045"),
    ("WS-15", (843, 108, 899, 150), "Right Mid-Back Desk", "Employee-15", "KA026"),
    ("WS-16", (826, 150, 897, 197), "Right Mid-Near Desk", "Employee-16", "KA046"),
    ("WS-17", (872, 70, 900, 105), "Right Back Desk - Top Mid", "Employee-17", "KA051"),
    ("WS-18", (892, 48, 922, 84), "Right Far-Back Desk (Near Door)", "Employee-18", "KA052")
]

exact_18_seats = {}
for sid, box, desc, name, emp_id in user_boxes:
    x1, y1, x2, y2 = box
    sx1 = int(round(x1 * scale_x))
    sy1 = int(round((y1 + y_offset) * scale_y))
    sx2 = int(round(x2 * scale_x))
    sy2 = int(round((y2 + y_offset) * scale_y))
    clean_name = name.lower().replace(" ", "")
    exact_18_seats[sid] = {
        "name": name,
        "employee_id": emp_id,
        "email": f"{clean_name}@kristellar.com",
        "rm_email": "manager@kristellar.com",
        "seat": [sx1, sy1, sx2, sy2],
        "desc": desc
    }

json_path = BASE_DIR / "monitoring" / "seats_3rd_floor.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(exact_18_seats, f, indent=4)
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

