def assign_people_and_faces_to_seats(people, face_records, seats):
    """
    Assigns detected people and faces to seats uniquely.
    - Each person is assigned to the seat with the highest overlap.
    - Each face is assigned to the person (and seat) it falls within.
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

            # Check if center is in seat
            center_inside = (sx1 <= pcx <= sx2 and sy1 <= pcy <= sy2)

            # Calculate intersection
            ix1 = max(px1, sx1)
            iy1 = max(py1, sy1)
            ix2 = min(px2, sx2)
            iy2 = min(py2, sy2)

            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                overlap_p = inter_area / p_area
                overlap_s = inter_area / s_area

                # Score combines center bonus and overlap
                score = overlap_p + (0.5 if center_inside else 0.0)

                if (center_inside or overlap_p >= 0.25) and score > best_score:
                    best_score = score
                    best_sid = sid

        if best_sid and not seat_status[best_sid]["occupied"]:
            seat_status[best_sid]["occupied"] = True
            seat_status[best_sid]["person_box"] = p

    # Match faces to seats/people
    for f in face_records:
        fx, fy, fw, fh = f["box"]
        fcx, fcy = fx + fw // 2, fy + fh // 2

        # Check which seat this face falls into
        for sid, sdata in seats.items():
            if not seat_status[sid]["occupied"]:
                continue
            pbox = seat_status[sid]["person_box"]
            # Check if face is inside person box or seat box
            if (pbox[0] <= fcx <= pbox[2] and pbox[1] <= fcy <= pbox[3]) or \
               (sdata["seat"][0] <= fcx <= sdata["seat"][2] and sdata["seat"][1] <= fcy <= sdata["seat"][3]):
                if seat_status[sid]["face"] is None or f["conf"] > seat_status[sid]["face"]["conf"]:
                    seat_status[sid]["face"] = f

    # Determine status and verification
    for sid, sdata in seats.items():
        info = seat_status[sid]
        assigned_emp_id = sdata.get("employee_id")

        if info["occupied"]:
            info["status_str"] = "OCCUPIED"
            if info["face"]:
                face_id = info["face"]["id"]
                sim = info["face"]["sim"]
                if face_id == assigned_emp_id:
                    info["verification"] = f"VERIFIED: {face_id}"
                elif face_id != "Unknown":
                    info["verification"] = f"OTHER: {face_id} ({sim:.2f})"
                else:
                    info["verification"] = f"UNKNOWN FACE ({sim:.2f})"
            else:
                info["verification"] = "PERSON (NO FACE)"
        else:
            info["status_str"] = "EMPTY"
            info["verification"] = "EMPTY"

    return seat_status

