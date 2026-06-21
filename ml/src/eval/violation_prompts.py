"""Comprehensive open-vocab prompt set — every mandated violation + detection + ANPR.

Shared by the foundation-model tests (LocateAnything-3B, SAM-3). Open-vocab models take a
noun phrase and return boxes/masks, so this dict IS the experiment: which phrasing best
isolates each of our tasks. Multiple phrasings per task → we take the union (best recall).

Maps 1:1 onto the project's tasks so the Lightning runs cover EVERYTHING in one pass.
"""

# task_key -> (human label, [phrase variants], box color for annotation)
VIOLATION_PROMPTS = {
    # --- Paradigm A: instance-attribute ---
    "helmet_no":      ("Helmet: NO helmet",
                       ["motorcyclist without helmet", "motorcycle rider with bare head",
                        "person riding motorcycle not wearing a helmet"], "red"),
    "helmet_yes":     ("Helmet: wearing helmet",
                       ["motorcyclist wearing a helmet", "rider with helmet on"], "green"),
    "seatbelt_no":    ("Seatbelt: NO seatbelt",
                       ["car driver without seatbelt", "driver not wearing seat belt",
                        "person in car not wearing seatbelt"], "red"),

    # --- Paradigm B: multi-instance counting ---
    "triple_riding":  ("Triple riding",
                       ["three people riding on one motorcycle", "motorcycle carrying three riders",
                        "overloaded motorcycle with three persons"], "orange"),

    # --- Paradigm C: scene-context / temporal ---
    "red_light":      ("Red-light running",
                       ["vehicle crossing a red traffic light", "car running the red light"], "magenta"),
    "wrong_side":     ("Wrong-side driving",
                       ["vehicle driving on the wrong side of the road",
                        "car facing oncoming traffic the wrong way"], "purple"),
    "stop_line":      ("Stop-line violation",
                       ["vehicle stopped beyond the stop line",
                        "car crossing the white stop line at a junction"], "yellow"),
    "illegal_parking":("Illegal parking",
                       ["illegally parked car", "vehicle parked in a no-parking zone"], "cyan"),

    # --- Detection backbone (esp. India-specific COCO-gap classes) ---
    "auto_rickshaw":  ("Auto-rickshaw (India class)",
                       ["auto rickshaw", "three-wheeler auto", "tuk tuk"], "blue"),
    "cycle_rickshaw": ("Cycle-rickshaw (India class)",
                       ["cycle rickshaw", "pedal rickshaw"], "blue"),
    "vehicles":       ("Vehicles (general)",
                       ["car", "motorcycle", "bus", "truck", "bicycle", "person"], "white"),

    # --- ANPR ---
    "license_plate":  ("License plate",
                       ["license plate", "vehicle number plate"], "lime"),
}

# Quick list of just the 7 mandated violations (for the headline summary).
MANDATED = ["helmet_no", "seatbelt_no", "triple_riding", "red_light",
            "wrong_side", "stop_line", "illegal_parking"]
