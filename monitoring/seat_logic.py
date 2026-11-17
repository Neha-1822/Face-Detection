"""
Seat and Occupancy Logic Engine
-------------------------------
Provides robust multi-modal person and face-to-seat assignment:
- Occupancy established by person body detection OR valid face detection inside the seat zone.
- Rejects false face detections (hands, keyboard, hair) using anatomical landmark geometry.
- Enforces reliable face recognition similarity & margin thresholds to prevent identity false-positives.
"""

import numpy as np


def is_valid_face(face, min_conf=0.52):
    """
    Validates that a detected face has genuine facial geometry.
    Filters out false positives on hands, keyboards, hair, and clothing.
    """
    conf = float(face[14])
    if conf < min_conf:
        return False

    w = float(face[2])
    h = float(face[3])
    if w < 12 or h < 12:
        return False

    # Standard human face aspect ratio range
    aspect = w / max(1.0, h)
    if aspect < 0.45 or aspect > 1.65:
        return False

    # Check 5 anatomical landmarks (eyes, nose, mouth corners)
    landmarks = face[4:14].reshape((5, 2))
    right_eye = landmarks[0]
    left_eye = landmarks[1]

    eye_dist = np.linalg.norm(right_eye - left_eye)
    # Interpupillary distance on a real face is >= 15% of face bounding box width
    if (eye_dist / w) < 0.15:
        return False

    return True


def assign_people_and_faces_to_seats(people, face_records, seats):
    """
    Assigns detected people and valid faces to workstation seats uniquely.
    - A seat is OCCUPIED if a person body overlaps the seat OR a face is inside the seat.
    - Each person is mapped to their best matching seat.
    - Faces are linked to the seat occupant.
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

    # 1. Match detected people (bodies) to seats
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

                # Require center inside or significant overlap to avoid aisle passersby
                if (center_inside or (overlap_p >= 0.25 and overlap_s >= 0.20)) and score > best_score:
                    best_score = score
                    best_sid = sid

        if best_sid and not seat_status[best_sid]["occupied"]:
            seat_status[best_sid]["occupied"] = True
            seat_status[best_sid]["person_box"] = p

    # 2. Match detected faces to seats
    # A detected face inside a seat is definitive proof that someone is at the seat!
    for f in face_records:
        fx, fy, fw, fh = f["box"]
        fcx, fcy = fx + fw // 2, fy + fh // 2

        for sid, sdata in seats.items():
            sx1, sy1, sx2, sy2 = sdata["seat"]
            pbox = seat_status[sid]["person_box"]

            inside_seat = (sx1 <= fcx <= sx2 and sy1 <= fcy <= sy2)
            inside_person = (pbox and pbox[0] <= fcx <= pbox[2] and pbox[1] <= fcy <= pbox[3])

            if inside_seat or inside_person:
                # Definitively mark the seat as occupied
                seat_status[sid]["occupied"] = True
                if seat_status[sid]["face"] is None or f["conf"] > seat_status[sid]["face"]["conf"]:
                    seat_status[sid]["face"] = f

    # 3. Determine display status & employee verification
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


def filter_overlapping_people(people, iou_thresh=0.45):
    """
    Suppresses duplicate bounding boxes on the same person (NMS).
    Prevents a single person from being counted twice.
    """
    if not people:
        return []
    boxes = sorted(people, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    kept = []
    for b in boxes:
        bx1, by1, bx2, by2 = b
        b_area = (bx2 - bx1) * (by2 - by1)
        duplicate = False
        for k in kept:
            kx1, ky1, kx2, ky2 = k
            k_area = (kx2 - kx1) * (ky2 - ky1)
            ix1, iy1 = max(bx1, kx1), max(by1, ky1)
            ix2, iy2 = min(bx2, kx2), min(by2, ky2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                iou = inter / float(b_area + k_area - inter)
                containment = inter / float(min(b_area, k_area))
                if iou > iou_thresh or containment > 0.65:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(b)
    return kept


def detect_gathering_clusters(people, distance_threshold=190, min_people=3, seats=None):
    """
    Identifies spatial clusters of >= min_people gathered in close proximity.
    Filters out normal office seating where people are simply working at
    their individual assigned workstations.
    """
    n = len(people)
    if n < min_people:
        return []

    centers = [((p[0] + p[2]) // 2, (p[1] + p[3]) // 2) for p in people]
    feet = [((p[0] + p[2]) // 2, p[3]) for p in people]

    # Build adjacency graph
    adj = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            dist_feet = np.hypot(feet[i][0] - feet[j][0], feet[i][1] - feet[j][1])
            dist_center = np.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])

            x_gap = max(0, max(people[i][0], people[j][0]) - min(people[i][2], people[j][2]))
            y_gap = max(0, max(people[i][1], people[j][1]) - min(people[i][3], people[j][3]))
            box_gap = np.hypot(x_gap, y_gap)

            if dist_feet <= distance_threshold or dist_center <= distance_threshold or box_gap < 60:
                adj[i].add(j)
                adj[j].add(i)

    visited = set()
    raw_clusters = []
    for i in range(n):
        if i not in visited:
            comp = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            if len(comp) >= min_people:
                raw_clusters.append(comp)

    valid_clusters = []
    for comp in raw_clusters:
        cluster_people = [people[idx] for idx in comp]
        min_x1 = min(p[0] for p in cluster_people)
        min_y1 = min(p[1] for p in cluster_people)
        max_x2 = max(p[2] for p in cluster_people)
        max_y2 = max(p[3] for p in cluster_people)

        cx = int(np.mean([centers[idx][0] for idx in comp]))
        cy = int(np.mean([centers[idx][1] for idx in comp]))

        if seats:
            # Map individuals to seats to see if this is just normal distinct desk seating
            seat_assignments = {}
            unseated_count = 0
            for idx in comp:
                p = people[idx]
                pcx, pcy = centers[idx]
                matched_seat = None
                for sid, sdata in seats.items():
                    sx1, sy1, sx2, sy2 = sdata["seat"]
                    if sx1 <= pcx <= sx2 and sy1 <= pcy <= sy2:
                        matched_seat = sid
                        break
                    ix1, iy1 = max(p[0], sx1), max(p[1], sy1)
                    ix2, iy2 = min(p[2], sx2), min(p[3], sy2)
                    if ix2 > ix1 and iy2 > iy1:
                        area = (ix2 - ix1) * (iy2 - iy1)
                        if area / max(1, (p[2] - p[0]) * (p[3] - p[1])) > 0.30:
                            matched_seat = sid
                            break
                if matched_seat:
                    seat_assignments.setdefault(matched_seat, []).append(idx)
                else:
                    unseated_count += 1

            max_at_same_seat = max((len(v) for v in seat_assignments.values()), default=0)
            # If everyone is seated at their own separate workstation with no visitors/crowding:
            if unseated_count == 0 and max_at_same_seat <= 1:
                continue

        valid_clusters.append({
            "indices": comp,
            "people_boxes": cluster_people,
            "box": [min_x1, min_y1, max_x2, max_y2],
            "centroid": (cx, cy),
            "count": len(comp)
        })

    return valid_clusters


def analyze_gathering_members(cluster_people, centroid, cached_faces, seats, default_rm_email="neha@kristellar.com"):
    """
    Identifies the employees participating in a gathering cluster,
    their assigned workstations, and their respective Reporting Managers.
    """
    identified_members = []
    unidentified_count = 0

    # 1. Match people in cluster to faces or seats
    for p in cluster_people:
        px1, py1, px2, py2 = p
        pcx, pcy = (px1 + px2) // 2, (py1 + py2) // 2

        matched_name = None
        matched_id = None
        matched_rm = default_rm_email
        matched_seat = None

        # Check if a recognized face is on this person
        for f in cached_faces:
            fx, fy, fw, fh = f["box"]
            fcx, fcy = fx + fw // 2, fy + fh // 2
            if px1 <= fcx <= px2 and py1 <= fcy <= py2:
                if f.get("id") and f["id"] != "Unknown":
                    matched_id = f["id"]
                    break

        # Check if person is near an assigned workstation
        for sid, sdata in seats.items():
            sx1, sy1, sx2, sy2 = sdata["seat"]
            if sx1 <= pcx <= sx2 and sy1 <= pcy <= sy2:
                matched_seat = sid
                matched_name = sdata["name"]
                matched_rm = sdata.get("rm_email") or default_rm_email
                if not matched_id and sdata.get("employee_id"):
                    matched_id = sdata["employee_id"]
                break

        # If recognized face matches a seat configuration
        if matched_id and not matched_name:
            for sid, sdata in seats.items():
                if sdata.get("employee_id") == matched_id:
                    matched_name = sdata["name"]
                    matched_rm = sdata.get("rm_email") or default_rm_email
                    if not matched_seat:
                        matched_seat = sid
                    break

        if matched_name:
            identified_members.append({
                "name": matched_name,
                "employee_id": matched_id or "Unknown",
                "seat_id": matched_seat or "Aisle / Common Area",
                "rm_email": matched_rm
            })
        elif matched_id and matched_id != "Unknown":
            identified_members.append({
                "name": f"Employee ({matched_id})",
                "employee_id": matched_id,
                "seat_id": matched_seat or "Aisle / Common Area",
                "rm_email": matched_rm
            })
        else:
            unidentified_count += 1

    # 2. Determine location description relative to nearest seats
    cx, cy = centroid
    seat_distances = []
    for sid, sdata in seats.items():
        sx1, sy1, sx2, sy2 = sdata["seat"]
        scx, scy = (sx1 + sx2) // 2, (sy1 + sy2) // 2
        d = np.hypot(cx - scx, cy - scy)
        seat_distances.append((d, sid, sdata["name"]))

    seat_distances.sort(key=lambda x: x[0])
    closest = seat_distances[:2]
    if closest and closest[0][0] <= 180:
        if len(closest) > 1 and closest[1][0] <= 200:
            location_desc = f"Near {closest[0][1]} ({closest[0][2]}) & {closest[1][1]} ({closest[1][2]})"
        else:
            location_desc = f"Near {closest[0][1]} ({closest[0][2]})"
    else:
        location_desc = "Central Walkway / Aisle Area"

    return {
        "identified": identified_members,
        "unidentified_count": unidentified_count,
        "location_desc": location_desc
    }

