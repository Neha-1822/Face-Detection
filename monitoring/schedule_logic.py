"""
Office Schedule & Break Management Engine (Floor-Aware)
--------------------------------------------------------
Enforces official office operating hours and authorized break windows:

1. Working Hours:
   - Monday to Friday: 09:30 AM to 06:30 PM (18:30)
   - Saturday: 09:30 AM to 01:30 PM (13:30)
   - Sunday: Office Closed

2. Authorized Breaks by Floor:
   - 3rd Floor (Area 1 & Area 2):
     * Morning Tea Break : 11:15 AM to 11:30 AM (Monday - Saturday)
     * Lunch Break        : 01:00 PM to 02:00 PM (Monday - Friday)
     * Evening Tea Break : 04:15 PM to 04:30 PM (Monday - Friday)
   - 4th Floor:
     * Morning Tea Break : 11:00 AM to 11:15 AM (Monday - Saturday)
     * Lunch Break        : 01:00 PM to 02:00 PM (Monday - Friday)
     * Evening Tea Break : 04:00 PM to 04:15 PM (Monday - Friday)

3. Break Enforcement & Timers:
   - During ANY break (Tea or Lunch): Absence timers are FROZEN. Break minutes do NOT count
     towards the 10-minute absence limit.
   - Zero alerts are sent during breaks.

4. Post-Break Absence Rule:
   - When any break finishes (11:30 AM / 02:00 PM / 04:30 PM for 3F; 11:15 AM / 02:00 PM / 04:15 PM for 4F):
     If an employee is NOT back in their assigned seat or is seen in the cafeteria after break ends,
     an automated alert email is dispatched to their Reporting Manager (RM).
   - Afterwards, normal 10-minute absence tracking resumes continuously during working hours.
"""

from datetime import datetime, time, timedelta


# Official Office Shift Timings
WEEKDAY_START_TIME = time(9, 30)      # 09:30 AM
WEEKDAY_END_TIME = time(18, 30)       # 06:30 PM

SATURDAY_START_TIME = time(9, 30)     # 09:30 AM
SATURDAY_END_TIME = time(13, 30)      # 01:30 PM

# Default/Legacy break constants
MORNING_TEA_START = time(11, 15)
MORNING_TEA_END = time(11, 30)
LUNCH_BREAK_START = time(13, 0)
LUNCH_BREAK_END = time(14, 0)
EVENING_TEA_START = time(16, 15)
EVENING_TEA_END = time(16, 30)

# 3rd Floor Breaks (Area 1 & Area 2)
FLOOR_3_BREAKS = {
    "morning": {
        "start": time(11, 15),
        "end": time(11, 30),
        "name": "Morning Tea Break (11:15 - 11:30 AM)",
        "label": "Morning Tea Break",
        "end_str": "11:30 AM"
    },
    "lunch": {
        "start": time(13, 0),
        "end": time(14, 0),
        "name": "Lunch Break (01:00 - 02:00 PM)",
        "label": "Lunch Break",
        "end_str": "02:00 PM"
    },
    "evening": {
        "start": time(16, 15),
        "end": time(16, 30),
        "name": "Evening Tea Break (04:15 - 04:30 PM)",
        "label": "Evening Tea Break",
        "end_str": "04:30 PM"
    }
}

# 4th Floor Breaks
FLOOR_4_BREAKS = {
    "morning": {
        "start": time(11, 0),
        "end": time(11, 15),
        "name": "Morning Tea Break (11:00 - 11:15 AM)",
        "label": "Morning Tea Break",
        "end_str": "11:15 AM"
    },
    "lunch": {
        "start": time(13, 0),
        "end": time(14, 0),
        "name": "Lunch Break (01:00 - 02:00 PM)",
        "label": "Lunch Break",
        "end_str": "02:00 PM"
    },
    "evening": {
        "start": time(16, 0),
        "end": time(16, 15),
        "name": "Evening Tea Break (04:00 - 04:15 PM)",
        "label": "Evening Tea Break",
        "end_str": "04:15 PM"
    }
}


def normalize_floor(floor):
    """Normalizes floor argument to '3' or '4'."""
    f_str = str(floor or "3").lower().strip()
    if "4" in f_str:
        return "4"
    return "3"


def get_floor_breaks(floor="3"):
    """Returns the break schedule dictionary for the given floor."""
    f_key = normalize_floor(floor)
    if f_key == "4":
        return FLOOR_4_BREAKS, "4th Floor"
    return FLOOR_3_BREAKS, "3rd Floor"


def get_schedule_status(now_dt=None, enforce=True, floor="3"):
    """
    Evaluates whether current datetime is within working hours and authorized breaks for the floor.

    Returns a dict with:
      - status: 'ACTIVE', 'TEA_BREAK' (used for all paused breaks), or 'OFF_HOURS'
      - is_break: bool (True during any authorized morning, lunch, or evening break)
      - break_type: str ('MORNING_TEA', 'LUNCH_BREAK', 'EVENING_TEA', or None)
      - can_track: bool (True only during active working hours outside breaks)
      - can_send_email: bool (True only during active working hours outside breaks)
      - reason: str (Descriptive explanation of the state)
      - badge_text: str (Text for HUD visual display)
      - badge_color: tuple (BGR color for OpenCV rendering)
      - floor: str ('3' or '4')
      - floor_label: str ('3rd Floor' or '4th Floor')
    """
    if now_dt is None:
        now_dt = datetime.now()

    floor_key = normalize_floor(floor)
    breaks, floor_label = get_floor_breaks(floor_key)

    if not enforce:
        return {
            "status": "ACTIVE",
            "is_break": False,
            "break_type": None,
            "can_track": True,
            "can_send_email": True,
            "reason": f"Schedule enforcement bypassed (Testing / Override Mode) - {floor_label}",
            "badge_text": "SCHEDULE BYPASSED",
            "badge_color": (0, 200, 255),
            "floor": floor_key,
            "floor_label": floor_label,
            "now": now_dt
        }

    weekday = now_dt.weekday()  # Monday is 0, Sunday is 6
    curr_time = now_dt.time()

    # 1. Sunday: Entire day closed
    if weekday == 6:
        return {
            "status": "OFF_HOURS",
            "is_break": False,
            "break_type": None,
            "can_track": False,
            "can_send_email": False,
            "reason": f"Sunday - Office Closed ({floor_label})",
            "badge_text": "OFFICE CLOSED (SUNDAY)",
            "badge_color": (100, 100, 100),
            "floor": floor_key,
            "floor_label": floor_label,
            "now": now_dt
        }

    # 2. Check Working Shift Boundary
    if weekday == 5:
        # Saturday: 09:30 AM to 01:30 PM
        shift_start = SATURDAY_START_TIME
        shift_end = SATURDAY_END_TIME
        shift_label = "Saturday Shift: 09:30 AM - 01:30 PM"
    else:
        # Monday - Friday: 09:30 AM to 06:30 PM
        shift_start = WEEKDAY_START_TIME
        shift_end = WEEKDAY_END_TIME
        shift_label = "Weekday Shift: 09:30 AM - 06:30 PM"

    if curr_time < shift_start:
        return {
            "status": "OFF_HOURS",
            "is_break": False,
            "break_type": None,
            "can_track": False,
            "can_send_email": False,
            "reason": f"Before Office Hours (Opens at {shift_start.strftime('%I:%M %p')}) - {floor_label}",
            "badge_text": f"OFFICE CLOSED (Opens {shift_start.strftime('%I:%M %p')})",
            "badge_color": (80, 80, 180),
            "floor": floor_key,
            "floor_label": floor_label,
            "now": now_dt
        }

    if curr_time >= shift_end:
        return {
            "status": "OFF_HOURS",
            "is_break": False,
            "break_type": None,
            "can_track": False,
            "can_send_email": False,
            "reason": f"After Office Hours (Closed at {shift_end.strftime('%I:%M %p')}) - {floor_label}",
            "badge_text": f"OFFICE CLOSED ({shift_label})",
            "badge_color": (80, 80, 180),
            "floor": floor_key,
            "floor_label": floor_label,
            "now": now_dt
        }

    # 3. Check Morning Tea Break (Mon - Sat)
    m_break = breaks["morning"]
    if m_break["start"] <= curr_time < m_break["end"]:
        return {
            "status": "TEA_BREAK",
            "is_break": True,
            "break_type": "MORNING_TEA",
            "can_track": False,
            "can_send_email": False,
            "reason": f"{floor_label} {m_break['name']}",
            "badge_text": f"{floor_label.upper()} {m_break['name'].upper()}",
            "badge_color": (0, 165, 255),
            "floor": floor_key,
            "floor_label": floor_label,
            "now": now_dt
        }

    # 4. Check Lunch Break (01:00 PM - 02:00 PM, Mon - Fri)
    if weekday < 5:
        l_break = breaks["lunch"]
        if l_break["start"] <= curr_time < l_break["end"]:
            return {
                "status": "TEA_BREAK",  # Compatible with is_tea_break checks across monitor scripts
                "is_break": True,
                "break_type": "LUNCH_BREAK",
                "can_track": False,
                "can_send_email": False,
                "reason": f"Lunch Break (01:00 PM - 02:00 PM) - {floor_label}",
                "badge_text": "LUNCH BREAK (01:00 - 02:00 PM)",
                "badge_color": (0, 165, 255),
                "floor": floor_key,
                "floor_label": floor_label,
                "now": now_dt
            }

    # 5. Check Evening Tea Break (Mon - Fri)
    if weekday < 5:
        e_break = breaks["evening"]
        if e_break["start"] <= curr_time < e_break["end"]:
            return {
                "status": "TEA_BREAK",
                "is_break": True,
                "break_type": "EVENING_TEA",
                "can_track": False,
                "can_send_email": False,
                "reason": f"{floor_label} {e_break['name']}",
                "badge_text": f"{floor_label.upper()} {e_break['name'].upper()}",
                "badge_color": (0, 165, 255),
                "floor": floor_key,
                "floor_label": floor_label,
                "now": now_dt
            }

    # 6. Normal Active Office Working Hours
    return {
        "status": "ACTIVE",
        "is_break": False,
        "break_type": None,
        "can_track": True,
        "can_send_email": True,
        "reason": f"Active Office Working Hours ({shift_label}) - {floor_label}",
        "badge_text": f"{floor_label.upper()} - ACTIVE",
        "badge_color": (0, 200, 0),
        "floor": floor_key,
        "floor_label": floor_label,
        "now": now_dt
    }


def get_post_break_status(now_dt=None, floor="3", grace_seconds=60, window_minutes=25):
    """
    Checks if current time is within the post-break audit window right after a tea or lunch break ended.

    Returns:
      {
        "is_post_break": bool,
        "break_name": str,
        "break_end_time": str,
        "seconds_since_end": float,
        "grace_expired": bool,
        "break_tag": str
      }
    """
    if now_dt is None:
        now_dt = datetime.now()

    floor_key = normalize_floor(floor)
    breaks, floor_label = get_floor_breaks(floor_key)
    weekday = now_dt.weekday()

    if weekday == 6:  # Sunday
        return {"is_post_break": False, "break_name": "", "break_end_time": "", "seconds_since_end": 0.0, "grace_expired": False, "break_tag": ""}

    # Candidate breaks to check today: Morning Tea, Lunch Break, Evening Tea
    candidate_breaks = [("morning", breaks["morning"])]
    if weekday < 5:  # Monday - Friday has lunch break and evening tea
        candidate_breaks.append(("lunch", breaks["lunch"]))
        candidate_breaks.append(("evening", breaks["evening"]))

    for b_key, b_info in candidate_breaks:
        b_end_dt = datetime.combine(now_dt.date(), b_info["end"])
        window_end_dt = b_end_dt + timedelta(minutes=window_minutes)

        if b_end_dt <= now_dt < window_end_dt:
            elapsed = (now_dt - b_end_dt).total_seconds()
            b_tag = f"{now_dt.strftime('%Y-%m-%d')}_{b_key}_floor{floor_key}"
            return {
                "is_post_break": True,
                "break_name": b_info["label"],
                "break_end_time": b_info["end_str"],
                "seconds_since_end": elapsed,
                "grace_expired": (elapsed >= grace_seconds),
                "break_tag": b_tag
            }

    return {
        "is_post_break": False,
        "break_name": "",
        "break_end_time": "",
        "seconds_since_end": 0.0,
        "grace_expired": False,
        "break_tag": ""
    }


def is_employee_on_break(employee_floor="3", now_dt=None):
    """
    Checks if an employee of a specific floor is currently in any authorized break window
    (morning tea, lunch, or evening tea).
    """
    status = get_schedule_status(now_dt=now_dt, enforce=True, floor=employee_floor)
    return status.get("is_break", status["status"] == "TEA_BREAK")


def get_employee_cafeteria_status(employee_floor="3", now_dt=None, grace_seconds=60):
    """
    Evaluates whether an employee detected in the cafeteria is authorized to be there,
    overstaying after break, or there during work hours.

    Returns:
      - 'ON_BREAK': Legitimate authorized tea or lunch break.
      - 'POST_BREAK_OVERSTAY': Detected in cafeteria AFTER their floor's break has ended.
      - 'WORKING_HOURS_UNAUTHORIZED': Detected in cafeteria during normal work hours.
      - 'OFF_HOURS': Office closed.
    """
    if now_dt is None:
        now_dt = datetime.now()

    floor_key = normalize_floor(employee_floor)
    status = get_schedule_status(now_dt=now_dt, enforce=True, floor=floor_key)

    if status["status"] == "OFF_HOURS":
        return {
            "state": "OFF_HOURS",
            "is_authorized": False,
            "reason": status["reason"]
        }

    if status.get("is_break", status["status"] == "TEA_BREAK"):
        return {
            "state": "ON_BREAK",
            "is_authorized": True,
            "break_type": status.get("break_type", "BREAK"),
            "reason": f"Authorized {status['reason']}"
        }

    # If shift is ACTIVE, check if it's within post-break overstay window
    post = get_post_break_status(now_dt=now_dt, floor=floor_key, grace_seconds=grace_seconds, window_minutes=30)
    if post["is_post_break"] and post["grace_expired"]:
        return {
            "state": "POST_BREAK_OVERSTAY",
            "is_authorized": False,
            "break_name": post["break_name"],
            "break_end_time": post["break_end_time"],
            "seconds_since_end": post["seconds_since_end"],
            "break_tag": post["break_tag"],
            "reason": f"Overstaying in cafeteria after {post['break_name']} (ended at {post['break_end_time']})"
        }

    return {
        "state": "WORKING_HOURS_UNAUTHORIZED",
        "is_authorized": False,
        "reason": f"Seen in cafeteria during active working hours (Outside authorized {floor_key}th floor break)"
    }
