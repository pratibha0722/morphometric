import os
import json
from math import hypot

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


# =====================================================
# CONFIGURATION
# =====================================================

INPUT_FOLDER = "hand_image"

OUTPUT_FOLDER = "hand_output1"
ANNOTATED_FOLDER = os.path.join(OUTPUT_FOLDER, "annotated")
JSON_FOLDER = os.path.join(OUTPUT_FOLDER, "json")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)
os.makedirs(JSON_FOLDER, exist_ok=True)


# =====================================================
# MEDIAPIPE
# =====================================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

drawer = mp.solutions.drawing_utils


# =====================================================
# FUNCTIONS
# =====================================================

def distance(p1, p2):
    return hypot(
        p1[0] - p2[0],
        p1[1] - p2[1]
    )


def landmark_to_pixel(landmarks, width, height):

    pts = []

    for lm in landmarks.landmark:

        pts.append((
            int(lm.x * width),
            int(lm.y * height)
        ))

    return pts


# Store all image measurements

all_results = []

# =====================================================
# HAND MEASUREMENTS
# =====================================================

def measure_hand(pts, pixels_per_mm):

    wrist = pts[0]

    thumb_tip = pts[4]
    index_tip = pts[8]
    middle_tip = pts[12]
    ring_tip = pts[16]
    pinky_tip = pts[20]

    thumb_mcp = pts[2]
    index_mcp = pts[5]
    middle_mcp = pts[9]
    ring_mcp = pts[13]
    pinky_mcp = pts[17]

    # Finger lengths

    thumb = distance(thumb_mcp, thumb_tip)

    index = distance(index_mcp, index_tip)

    middle = distance(middle_mcp, middle_tip)

    ring = distance(ring_mcp, ring_tip)

    pinky = distance(pinky_mcp, pinky_tip)

    # Hand measurements

    hand_length = distance(wrist, middle_tip)

    palm_height = distance(wrist, middle_mcp)

    palm_width = distance(index_mcp, pinky_mcp)

    hand_width = max(
        distance(pts[5], pts[17]),
        distance(pts[2], pts[17]),
        distance(pts[1], pts[17])
    )

    return {

    "hand_length_mm": round(hand_length / pixels_per_mm, 2),
    "hand_width_mm": round(hand_width / pixels_per_mm, 2),

    "palm_height_mm": round(palm_height / pixels_per_mm, 2),
    "palm_width_mm": round(palm_width / pixels_per_mm, 2),

    "thumb_mm": round(thumb / pixels_per_mm, 2),
    "index_mm": round(index / pixels_per_mm, 2),
    "middle_mm": round(middle / pixels_per_mm, 2),
    "ring_mm": round(ring / pixels_per_mm, 2),
    "pinky_mm": round(pinky / pixels_per_mm, 2)
    }
    
    # =====================================================
# PROCESS ALL IMAGES
# =====================================================

supported_formats = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
)

for filename in sorted(os.listdir(INPUT_FOLDER)):

    if not filename.lower().endswith(supported_formats):
        continue

    print(f"Processing {filename}")

    image_path = os.path.join(INPUT_FOLDER, filename)

    image = cv2.imread(image_path)

    if image is None:
        print("Cannot open image.")
        continue

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        print("No hand detected.")
        continue

    hand_landmarks = results.multi_hand_landmarks[0]

    h, w = image.shape[:2]

    pts = landmark_to_pixel(
        hand_landmarks,
        w,
        h
    )

    pixels_per_mm = 1.0
    measurements = measure_hand(pts, pixels_per_mm)

    drawer.draw_landmarks(
        image,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    y = 30

    for key, value in measurements.items():

        cv2.putText(
            image,
            f"{key}: {value:.1f}",
            (20, y),
            font,
            0.6,
            (0,255,0),
            2
        )

        y += 25

    result = {
        "image": filename
    }

    result.update(measurements)

    all_results.append(result)

    json_name = os.path.splitext(filename)[0] + ".json"

    with open(
        os.path.join(JSON_FOLDER, json_name),
        "w"
    ) as f:

        json.dump(
            result,
            f,
            indent=4
        )

    cv2.imwrite(
        os.path.join(
            ANNOTATED_FOLDER,
            filename
        ),
        image
    )

    print("Finished", filename)
    
    # =====================================================
# SAVE CSV
# =====================================================

df = pd.DataFrame(all_results)

csv_file = os.path.join(
    OUTPUT_FOLDER,
    "hand_measurements.csv"
)

df.to_csv(
    csv_file,
    index=False
)

print()
print("=" * 50)
print("Processing Complete")
print("=" * 50)
print(f"Images Processed : {len(all_results)}")
print(f"CSV Saved To     : {csv_file}")
print(f"JSON Folder      : {JSON_FOLDER}")
print(f"Annotated Images : {ANNOTATED_FOLDER}")
print("=" * 50)