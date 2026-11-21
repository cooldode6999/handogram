import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import requests
import base64
import threading
import time
import math
import os
import pyautogui
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, Response
from pydantic import BaseModel
import uvicorn

# ================= HANDOGRAM SYSTEM CONFIGURATION =================
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llava:7b"
TASK_MODEL_PATH = "hand_landmarker.task"
AUTO_TYPE_ENABLED = True
SMOOTHING_ALPHA = 0.55
MAX_DROP_FRAMES = 7
pyautogui.FAILSAFE = False
# ==================================================================

if not os.path.exists(TASK_MODEL_PATH):
    raise FileNotFoundError("hand_landmarker.task missing in folder!")

base_options = python.BaseOptions(model_asset_path=TASK_MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.60,
    min_hand_presence_confidence=0.60,
    min_tracking_confidence=0.55
)
detector = vision.HandLandmarker.create_from_options(options)

app = FastAPI(title="Handogram Spatial OS")
SCREEN_W, SCREEN_H = pyautogui.size()

INK_PALETTE = [
    {"name": "NEON CYAN",    "bgr": (255, 255, 0),   "hex": "#00ffff"},
    {"name": "AMBER GOLD",   "bgr": (0, 215, 255),   "hex": "#ffd700"},
    {"name": "CYBER PINK",   "bgr": (180, 0, 255),   "hex": "#ff00b4"},
    {"name": "MATRIX GREEN", "bgr": (50, 255, 50),   "hex": "#32ff32"},
    {"name": "ELECTRIC RED", "bgr": (60, 60, 255),   "hex": "#ff3c3c"},
    {"name": "PURE WHITE",   "bgr": (255, 255, 255), "hex": "#ffffff"},
]

SKELETON_THEMES = [
    {"name": "CYBER-PRISM", "palm": (200, 200, 200), "fingers": [(0, 165, 255), (255, 255, 0), (255, 0, 255), (0, 255, 128), (255, 100, 50)]},
    {"name": "SOLAR GOLD",  "palm": (0, 0, 220),     "fingers": [(0, 215, 255), (0, 255, 255), (0, 140, 255), (0, 215, 255), (0, 69, 255)]},
    {"name": "OVERDRIVE",   "palm": (255, 0, 128),   "fingers": [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255)]},
    {"name": "PLASMA BLUE", "palm": (0, 120, 255),   "fingers": [(0, 255, 255), (0, 180, 255), (50, 100, 255), (180, 0, 255), (255, 200, 0)]}
]

TEXT_COLORS = [
    {"name": "MINT NEON",   "hex": "#00ff9d"},
    {"name": "CYAN PULSE",  "hex": "#00e5ff"},
    {"name": "AMBER GOLD",  "hex": "#ffd700"},
    {"name": "HYPER PINK",  "hex": "#ff2a85"},
    {"name": "CRISP WHITE", "hex": "#ffffff"}
]

KB_ROWS = [
    ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "BKSP"],
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L", "ENTER"],
    ["Z", "X", "C", "V", "B", "N", "M", "SPACE", "CLR"]
]

state = {
    "system_mode": "CANVAS",
    "shape_snap": True,
    "ink_idx": 0,
    "skel_idx": 0,
    "text_color_idx": 0,
    "brush_size": 8,
    "status_text": "HANDOGRAM ONLINE // STANDBY",
    "recognized_text": "",
    "is_analyzing": False,
    "fps": 0,
    "clear_flag": False,
    "analyze_flag": False,
    "key_sound_trigger": 0,
    "last_typed_key": "",
    "scale": 1.0,
    "hands_detected": 0
}

# ================= OBJECT-BASED HOLOGRAPHIC STORAGE =================
holo_objects = []
active_stroke = []
selected_obj_idx = None
next_obj_id = 1
lock = threading.Lock()


def classify_and_create_object(pts, bgr, hex_col, thick, snap_enabled):
    """Converts raw finger points into an independent Handogram Shape or Freehand Object"""
    global next_obj_id
    if len(pts) < 4:
        return None

    arr = np.array(pts, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(arr)
    diag = math.hypot(w, h)
    start_pt, end_pt = pts[0], pts[-1]
    close_dist = math.hypot(start_pt[0] - end_pt[0], start_pt[1] - end_pt[1])

    obj = {
        "id": next_obj_id,
        "type": "path",
        "pts": list(pts),
        "bgr": bgr,
        "hex": hex_col,
        "thick": thick
    }
    next_obj_id += 1

    if not snap_enabled or diag < 35 or len(pts) < 10:
        return obj

    # 1. CLOSED SHAPE DETECTION (Circle, Triangle, Rectangle)
    if close_dist < max(45.0, diag * 0.38):
        hull = cv2.convexHull(arr)
        peri = cv2.arcLength(hull, True)
        area = cv2.contourArea(hull)
        circularity = (4 * math.pi * area) / (peri * peri + 1e-5)
        approx = cv2.approxPolyDP(hull, 0.055 * peri, True)

        if circularity > 0.68:
            cx, cy = x + w // 2, y + h // 2
            r = int((w + h) / 4)
            obj["type"] = "circle"
            obj["pts"] = [(cx, cy), (cx + r, cy)]
        elif len(approx) == 3:
            obj["type"] = "triangle"
            obj["pts"] = [tuple(approx[0][0]), tuple(approx[1][0]), tuple(approx[2][0])]
        else:
            obj["type"] = "rect"
            obj["pts"] = [(x, y), (x + w, y + h)]
    else:
        # 2. OPEN STRAIGHT LINE -> FLOWCHART ARROW
        path_len = sum(math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]) for i in range(1, len(pts)))
        if close_dist > path_len * 0.80:
            obj["type"] = "arrow"
            obj["pts"] = [start_pt, end_pt]

    return obj


def get_object_center(obj):
    pts = obj["pts"]
    if obj["type"] == "circle":
        return pts[0]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (sum(xs) // len(xs), sum(ys) // len(ys))


def render_all_objects(target_img, highlight_idx=None):
    """Renders all independent Handogram objects + current live stroke"""
    for idx, obj in enumerate(holo_objects):
        col = obj["bgr"]
        th = obj["thick"]
        pts = obj["pts"]
        is_sel = (idx == highlight_idx)

        if obj["type"] == "circle" and len(pts) >= 2:
            cx, cy = pts[0]
            r = max(10, int(math.hypot(pts[1][0] - cx, pts[1][1] - cy)))
            if is_sel:
                cv2.circle(target_img, (cx, cy), r + 6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(target_img, (cx, cy), r, col, th, cv2.LINE_AA)

        elif obj["type"] == "rect" and len(pts) >= 2:
            p1, p2 = pts[0], pts[1]
            if is_sel:
                cv2.rectangle(target_img, (p1[0] - 6, p1[1] - 6), (p2[0] + 6, p2[1] + 6), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(target_img, p1, p2, col, th, cv2.LINE_AA)

        elif obj["type"] == "triangle" and len(pts) >= 3:
            tri = np.array(pts[:3], dtype=np.int32)
            if is_sel:
                cv2.polylines(target_img, [tri], True, (255, 255, 255), th + 4, cv2.LINE_AA)
            cv2.polylines(target_img, [tri], True, col, th, cv2.LINE_AA)

        elif obj["type"] == "arrow" and len(pts) >= 2:
            if is_sel:
                cv2.arrowedLine(target_img, pts[0], pts[1], (255, 255, 255), th + 4, cv2.LINE_AA, tipLength=0.18)
            cv2.arrowedLine(target_img, pts[0], pts[1], col, th, cv2.LINE_AA, tipLength=0.18)

        else:
            if len(pts) >= 2:
                arr = np.array(pts, dtype=np.int32)
                if is_sel:
                    cv2.polylines(target_img, [arr], False, (255, 255, 255), th + 4, cv2.LINE_AA)
                cv2.polylines(target_img, [arr], False, col, th, cv2.LINE_AA)

    if len(active_stroke) >= 2:
        ink = INK_PALETTE[state["ink_idx"]]
        arr = np.array(active_stroke, dtype=np.int32)
        cv2.polylines(target_img, [arr], False, ink["bgr"], state["brush_size"], cv2.LINE_AA)


def draw_cyber_skeleton(frame, lm_list, theme_idx, label=""):
    theme = SKELETON_THEMES[theme_idx]
    palm_col = theme["palm"]
    f_cols = theme["fingers"]

    for p1, p2 in [(0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]:
        cv2.line(frame, lm_list[p1], lm_list[p2], palm_col, 2, cv2.LINE_AA)

    chains = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
    for idx, chain in enumerate(chains):
        col = f_cols[idx]
        for i in range(len(chain) - 1):
            cv2.line(frame, lm_list[chain[i]], lm_list[chain[i + 1]], col, 4, cv2.LINE_AA)
            cv2.line(frame, lm_list[chain[i]], lm_list[chain[i + 1]], (255, 255, 255), 1, cv2.LINE_AA)

    tips = [4, 8, 12, 16, 20]
    for idx, pt in enumerate(lm_list):
        if idx in tips:
            cv2.circle(frame, pt, 7, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 11, f_cols[tips.index(idx)], 2, cv2.LINE_AA)
        else:
            cv2.circle(frame, pt, 3, (20, 20, 25), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 3, palm_col, 1, cv2.LINE_AA)

    if label:
        cv2.putText(frame, label, (lm_list[0][0] - 25, lm_list[0][1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)


def get_fingers_up(lm_list, is_right=True):
    fingers = []
    if is_right:
        fingers.append(lm_list[4][0] < lm_list[3][0])
    else:
        fingers.append(lm_list[4][0] > lm_list[3][0])

    for tip in [8, 12, 16, 20]:
        fingers.append(lm_list[tip][1] < (lm_list[tip - 2][1] + 8))
    return fingers


def draw_holographic_keyboard(frame, finger_pt, is_pinching):
    h, w, _ = frame.shape
    kb_x1, kb_y1 = int(w * 0.08), int(h * 0.22)
    kb_x2, kb_y2 = int(w * 0.92), int(h * 0.82)

    overlay = frame.copy()
    cv2.rectangle(overlay, (kb_x1 - 12, kb_y1 - 12), (kb_x2 + 12, kb_y2 + 12), (10, 18, 32), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (kb_x1 - 12, kb_y1 - 12), (kb_x2 + 12, kb_y2 + 12), (0, 240, 255), 2)

    row_h = (kb_y2 - kb_y1) // len(KB_ROWS)
    hovered_key = None

    for r_idx, row in enumerate(KB_ROWS):
        y1 = kb_y1 + r_idx * row_h + 6
        y2 = y1 + row_h - 12
        total_units = sum(2.2 if k in ["SPACE", "BKSP", "ENTER"] else 1.0 for k in row)
        unit_w = (kb_x2 - kb_x1) / total_units
        curr_x = kb_x1

        for key in row:
            mult = 2.2 if key in ["SPACE", "BKSP", "ENTER"] else 1.0
            kw = int(unit_w * mult)
            x1, x2 = int(curr_x + 4), int(curr_x + kw - 4)
            curr_x += kw

            is_hover = False
            if finger_pt and (x1 <= finger_pt[0] <= x2) and (y1 <= finger_pt[1] <= y2):
                is_hover = True
                hovered_key = key

            box_col = (0, 255, 180) if (is_hover and is_pinching) else ((0, 240, 255) if is_hover else (60, 110, 140))
            thickness = -1 if (is_hover and is_pinching) else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_col, thickness, cv2.LINE_AA)

            txt_col = (10, 15, 25) if (is_hover and is_pinching) else (230, 250, 255)
            font_scale = 0.55 if len(key) > 1 else 0.75
            tx = x1 + (x2 - x1) // 2 - (len(key) * 7)
            ty = y1 + (y2 - y1) // 2 + 7
            cv2.putText(frame, key, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, txt_col, 2, cv2.LINE_AA)

    return hovered_key


def run_ollama_ocr(img_canvas):
    state["is_analyzing"] = True
    state["status_text"] = "HANDOGRAM AI SCANNING..."
    try:
        gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)
        if coords is None:
            state["recognized_text"] = "Board is empty! Write first."
            state["is_analyzing"] = False
            return

        x, y, w, h = cv2.boundingRect(coords)
        pad = 55
        cropped = thresh[max(0, y - pad):min(thresh.shape[0], y + h + pad),
                         max(0, x - pad):min(thresh.shape[1], x + w + pad)]

        cropped = cv2.dilate(cropped, np.ones((4, 4), np.uint8), iterations=1)
        cropped = cv2.GaussianBlur(cropped, (3, 3), 0)
        _, buffer = cv2.imencode('.png', cropped)
        img_b64 = base64.b64encode(buffer).decode('utf-8')

        payload = {
            "model": MODEL_NAME,
            "prompt": (
                "You are a precision OCR engine. Read the white handwritten text, numbers, or equation on the black background. "
                "Return ONLY the exact transcribed text with zero commentary, no quotes, and no extra words."
            ),
            "images": [img_b64],
            "stream": False
        }
        res = requests.post(OLLAMA_URL, json=payload, timeout=30)
        out = res.json().get("response", "").strip()
        if out:
            state["recognized_text"] = (state["recognized_text"] + " " + out).strip()
            if AUTO_TYPE_ENABLED:
                pyautogui.write(out + " ", interval=0.02)
    except Exception as e:
        state["recognized_text"] = f"Ollama Error: {str(e)[:35]}"
    finally:
        state["is_analyzing"] = False


def commit_active_stroke():
    """Finalizes current Right-Hand stroke into an independent object so future strokes stay separate"""
    global active_stroke
    if len(active_stroke) >= 4:
        ink = INK_PALETTE[state["ink_idx"]]
        new_obj = classify_and_create_object(
            active_stroke, ink["bgr"], ink["hex"], state["brush_size"], state["shape_snap"]
        )
        if new_obj:
            holo_objects.append(new_obj)
    active_stroke = []


def generate_frames():
    global holo_objects, active_stroke, selected_obj_idx
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    smooth_x, smooth_y = 0, 0
    mouse_sx, mouse_sy = 0, 0
    drop_counter = 0

    last_pinch_time = 0
    last_rock_time = 0
    last_ai_time = 0
    last_key_time = 0
    last_click_time = 0

    shaka_start_time = None
    grab_prev_pt = None
    zoom_prev_dist = None

    fps_prev = 0
    start_ms = int(time.time() * 1000)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        with lock:
            if state["clear_flag"]:
                holo_objects = []
                active_stroke = []
                selected_obj_idx = None
                state["clear_flag"] = False
                state["scale"] = 1.0

            if state["analyze_flag"] and not state["is_analyzing"]:
                state["analyze_flag"] = False
                snap_canvas = np.zeros((h, w, 3), dtype=np.uint8)
                render_all_objects(snap_canvas)
                threading.Thread(target=run_ollama_ocr, args=(snap_canvas,), daemon=True).start()

        now = time.time()
        state["fps"] = int(1 / (now - fps_prev)) if fps_prev else 0
        fps_prev = now

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = detector.detect_for_video(mp_img, int(now * 1000) - start_ms)

        hands_data = []
        if results.hand_landmarks:
            for idx, hand_lms in enumerate(results.hand_landmarks):
                lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
                handed_label = "RIGHT"
                if results.handedness and idx < len(results.handedness):
                    raw_cat = results.handedness[idx][0].category_name
                    handed_label = "RIGHT" if raw_cat == "Left" else "LEFT"

                draw_cyber_skeleton(frame, lm_list, state["skel_idx"], handed_label)
                hands_data.append({"label": handed_label, "lm": lm_list})

        state["hands_detected"] = len(hands_data)
        mode = state["system_mode"]

        # ================= MODE 1: INDEPENDENT OBJECT AIR-CANVAS =================
        if mode == "CANVAS":
            right_hand = next((hd["lm"] for hd in hands_data if hd["label"] == "RIGHT"), None)
            left_hand = next((hd["lm"] for hd in hands_data if hd["label"] == "LEFT"), None)
            if not right_hand and len(hands_data) == 1:
                right_hand = hands_data[0]["lm"]

            # --- LEFT HAND: GRAB, MOVE & SCALE ONLY THE TARGET OBJECT ---
            if left_hand:
                l_idx, l_thm = left_hand[8], left_hand[4]
                l_pinch = math.hypot(l_idx[0] - l_thm[0], l_idx[1] - l_thm[1]) < 45

                if l_pinch and len(holo_objects) > 0:
                    grab_pt = ((l_idx[0] + l_thm[0]) // 2, (l_idx[1] + l_thm[1]) // 2)
                    cv2.circle(frame, grab_pt, 22, (0, 215, 255), 3, cv2.LINE_AA)

                    with lock:
                        if selected_obj_idx is None or selected_obj_idx >= len(holo_objects):
                            best_i = len(holo_objects) - 1
                            best_d = 99999
                            for i, obj in enumerate(holo_objects):
                                oc = get_object_center(obj)
                                d = math.hypot(grab_pt[0] - oc[0], grab_pt[1] - oc[1])
                                if d < best_d:
                                    best_d = d
                                    best_i = i
                            selected_obj_idx = best_i
                            grab_prev_pt = grab_pt

                        if grab_prev_pt is not None and selected_obj_idx is not None:
                            dx = grab_pt[0] - grab_prev_pt[0]
                            dy = grab_pt[1] - grab_prev_pt[1]
                            obj = holo_objects[selected_obj_idx]
                            obj["pts"] = [(p[0] + dx, p[1] + dy) for p in obj["pts"]]
                            grab_prev_pt = grab_pt
                            state["status_text"] = f"MOVING OBJECT #{obj['id']} [{obj['type'].upper()}]"

                        if right_hand and selected_obj_idx is not None:
                            r_idx, r_thm = right_hand[8], right_hand[4]
                            if math.hypot(r_idx[0] - r_thm[0], r_idx[1] - r_thm[1]) < 48:
                                dist_hands = max(40.0, math.hypot(r_idx[0] - l_idx[0], r_idx[1] - l_idx[1]))
                                cv2.line(frame, l_idx, r_idx, (0, 255, 255), 2, cv2.LINE_AA)
                                if zoom_prev_dist is not None:
                                    ratio = dist_hands / zoom_prev_dist
                                    obj = holo_objects[selected_obj_idx]
                                    cx, cy = get_object_center(obj)
                                    obj["pts"] = [
                                        (int(cx + (p[0] - cx) * ratio), int(cy + (p[1] - cy) * ratio))
                                        for p in obj["pts"]
                                    ]
                                    state["scale"] = round(state["scale"] * ratio, 2)
                                    state["status_text"] = f"SCALING OBJECT #{obj['id']}"
                                zoom_prev_dist = dist_hands
                            else:
                                zoom_prev_dist = None
                else:
                    selected_obj_idx = None
                    grab_prev_pt = None
                    zoom_prev_dist = None
            else:
                selected_obj_idx = None
                grab_prev_pt = None
                zoom_prev_dist = None

            # --- RIGHT HAND: DRAW NEW INDEPENDENT STROKE / GESTURES ---
            if right_hand and zoom_prev_dist is None:
                drop_counter = 0
                fingers = get_fingers_up(right_hand, is_right=True)
                raw_x, raw_y = right_hand[8]
                thm_x, thm_y = right_hand[4]

                if smooth_x == 0 and smooth_y == 0:
                    smooth_x, smooth_y = raw_x, raw_y
                else:
                    smooth_x = int(SMOOTHING_ALPHA * raw_x + (1 - SMOOTHING_ALPHA) * smooth_x)
                    smooth_y = int(SMOOTHING_ALPHA * raw_y + (1 - SMOOTHING_ALPHA) * smooth_y)

                pinch_dist = math.hypot(raw_x - thm_x, raw_y - thm_y)

                # 1. SAFE WIPE GESTURE: SHAKA SIGN (Thumb + Pinky Up) held 1.0s
                if fingers[0] and not fingers[1] and not fingers[2] and not fingers[3] and fingers[4]:
                    with lock:
                        commit_active_stroke()
                    if shaka_start_time is None:
                        shaka_start_time = now
                    progress = min(1.0, (now - shaka_start_time) / 1.0)
                    state["status_text"] = f"HOLD SHAKA TO WIPE ({int(progress * 100)}%)"
                    center_h = right_hand[9]
                    cv2.ellipse(frame, center_h, (48, 48), -90, 0, int(360 * progress), (0, 0, 255), 5, cv2.LINE_AA)
                    if progress >= 1.0:
                        with lock:
                            holo_objects = []
                            active_stroke = []
                            selected_obj_idx = None
                        state["scale"] = 1.0
                        state["status_text"] = "BOARD WIPED CLEAN!"
                        shaka_start_time = None
                else:
                    shaka_start_time = None

                    # 2. PINCH (Thumb + Index) -> Color Shift
                    if pinch_dist < 38 and not fingers[2] and not fingers[3]:
                        with lock:
                            commit_active_stroke()
                        state["status_text"] = "PINCH // COLOR SHIFT"
                        if now - last_pinch_time > 0.75:
                            state["ink_idx"] = (state["ink_idx"] + 1) % len(INK_PALETTE)
                            state["skel_idx"] = (state["skel_idx"] + 1) % len(SKELETON_THEMES)
                            last_pinch_time = now

                    # 3. SPIDER-SIGN (Index + Pinky) -> Transcript Text Color
                    elif fingers[1] and not fingers[2] and not fingers[3] and fingers[4]:
                        with lock:
                            commit_active_stroke()
                        state["status_text"] = "SPIDER-SIGN // TEXT COLOR"
                        if now - last_rock_time > 0.75:
                            state["text_color_idx"] = (state["text_color_idx"] + 1) % len(TEXT_COLORS)
                            last_rock_time = now

                    # 4. ONLY INDEX FINGER UP -> Draw Brand New Independent Stroke
                    elif fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
                        ink = INK_PALETTE[state["ink_idx"]]
                        state["status_text"] = f"DRAWING NEW OBJECT [{ink['name']}]"
                        with lock:
                            active_stroke.append((smooth_x, smooth_y))
                        cv2.circle(frame, (smooth_x, smooth_y), state["brush_size"] // 2 + 5, (255, 255, 255), 2, cv2.LINE_AA)

                    # 5. PEACE SIGN (Index + Middle) -> Commit Stroke & Auto-Snap Geometric Shape!
                    elif fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
                        with lock:
                            commit_active_stroke()
                        state["status_text"] = "OBJECT LOCKED // HOVER"
                        cv2.circle(frame, (smooth_x, smooth_y), 12, (255, 220, 0), 2, cv2.LINE_AA)

                    # 6. CLOSED FIST -> AI OCR Read
                    elif not fingers[0] and not fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
                        with lock:
                            commit_active_stroke()
                        state["status_text"] = "FIST // AI READING"
                        if not state["is_analyzing"] and (now - last_ai_time > 2.2):
                            last_ai_time = now
                            snap_canvas = np.zeros((h, w, 3), dtype=np.uint8)
                            with lock:
                                render_all_objects(snap_canvas)
                            threading.Thread(target=run_ollama_ocr, args=(snap_canvas,), daemon=True).start()
                    else:
                        with lock:
                            commit_active_stroke()
                        smooth_x, smooth_y = 0, 0
            else:
                drop_counter += 1
                if drop_counter > MAX_DROP_FRAMES:
                    with lock:
                        commit_active_stroke()
                    smooth_x, smooth_y = 0, 0
                    shaka_start_time = None

        # ================= MODE 2: HOLOGRAPHIC AIR-KEYBOARD =================
        elif mode == "KEYBOARD":
            primary_hand = hands_data[0]["lm"] if hands_data else None
            finger_pt = None
            is_pinch = False
            if primary_hand:
                ix, iy = primary_hand[8]
                tx, ty = primary_hand[4]
                finger_pt = (ix, iy)
                is_pinch = math.hypot(ix - tx, iy - ty) < 38
                cv2.circle(frame, finger_pt, 12, (0, 255, 180) if is_pinch else (0, 240, 255), -1)

            hovered_key = draw_holographic_keyboard(frame, finger_pt, is_pinch)
            state["status_text"] = f"AIR-KEYBOARD [{hovered_key or 'AIM KEY'}]"

            if hovered_key and is_pinch and (now - last_key_time > 0.45):
                last_key_time = now
                state["key_sound_trigger"] += 1
                state["last_typed_key"] = hovered_key
                if hovered_key == "SPACE":
                    state["recognized_text"] += " "
                    if AUTO_TYPE_ENABLED:
                        pyautogui.press("space")
                elif hovered_key == "BKSP":
                    state["recognized_text"] = state["recognized_text"][:-1]
                    if AUTO_TYPE_ENABLED:
                        pyautogui.press("backspace")
                elif hovered_key == "ENTER":
                    state["recognized_text"] += "\n"
                    if AUTO_TYPE_ENABLED:
                        pyautogui.press("enter")
                elif hovered_key == "CLR":
                    state["recognized_text"] = ""
                else:
                    state["recognized_text"] += hovered_key
                    if AUTO_TYPE_ENABLED:
                        pyautogui.write(hovered_key.lower())

        # ================= MODE 3: SPATIAL AIR-MOUSE =================
        elif mode == "MOUSE":
            primary_hand = hands_data[0]["lm"] if hands_data else None
            pad_x, pad_y = int(w * 0.18), int(h * 0.18)
            cv2.rectangle(frame, (pad_x, pad_y), (w - pad_x, h - pad_y), (0, 240, 255), 2)

            if primary_hand:
                ix, iy = primary_hand[8]
                mx, my = primary_hand[12]
                rx, ry = primary_hand[16]
                tx, ty = primary_hand[4]

                norm_x = np.clip((ix - pad_x) / (w - 2 * pad_x), 0.0, 1.0)
                norm_y = np.clip((iy - pad_y) / (h - 2 * pad_y), 0.0, 1.0)
                target_sx = int(norm_x * SCREEN_W)
                target_sy = int(norm_y * SCREEN_H)

                if mouse_sx == 0 and mouse_sy == 0:
                    mouse_sx, mouse_sy = target_sx, target_sy
                else:
                    mouse_sx = int(0.45 * target_sx + 0.55 * mouse_sx)
                    mouse_sy = int(0.45 * target_sy + 0.55 * mouse_sy)

                pyautogui.moveTo(mouse_sx, mouse_sy)
                cv2.circle(frame, (ix, iy), 14, (0, 255, 255), 3)

                if math.hypot(tx - mx, ty - my) < 36 and (now - last_click_time > 0.4):
                    last_click_time = now
                    pyautogui.click()
                    state["key_sound_trigger"] += 1
                    state["status_text"] = "AIR-MOUSE // LEFT CLICK"
                elif math.hypot(tx - rx, ty - ry) < 36 and (now - last_click_time > 0.5):
                    last_click_time = now
                    pyautogui.rightClick()
                    state["key_sound_trigger"] += 1
                    state["status_text"] = "AIR-MOUSE // RIGHT CLICK"
                else:
                    state["status_text"] = f"AIR-MOUSE ACTIVE ({mouse_sx}, {mouse_sy})"

        # Composite Handogram Object Layer onto Live Camera Frame
        if mode == "CANVAS":
            holo_layer = np.zeros((h, w, 3), dtype=np.uint8)
            with lock:
                render_all_objects(holo_layer, highlight_idx=selected_obj_idx)
            gray_c = cv2.cvtColor(holo_layer, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_c, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            combined = cv2.add(cv2.bitwise_and(frame, frame, mask=mask_inv),
                               cv2.bitwise_and(holo_layer, holo_layer, mask=mask))
        else:
            combined = frame

        _, jpeg = cv2.imencode('.jpg', combined, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')


@app.get("/", response_class=HTMLResponse)
def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/status")
def get_status():
    return JSONResponse({
        "system_mode": state["system_mode"],
        "shape_snap": state["shape_snap"],
        "status_text": state["status_text"],
        "fps": state["fps"],
        "hands_detected": state["hands_detected"],
        "scale": state["scale"],
        "ink": INK_PALETTE[state["ink_idx"]],
        "skeleton": SKELETON_THEMES[state["skel_idx"]]["name"],
        "text_color": TEXT_COLORS[state["text_color_idx"]],
        "recognized_text": state["recognized_text"],
        "is_analyzing": state["is_analyzing"],
        "key_sound_trigger": state["key_sound_trigger"]
    })


@app.get("/export/png")
def export_png():
    holo_layer = np.zeros((720, 1280, 3), dtype=np.uint8)
    with lock:
        render_all_objects(holo_layer)
    b, g, r = cv2.split(holo_layer)
    gray = cv2.cvtColor(holo_layer, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    rgba = cv2.merge([b, g, r, alpha])
    _, buf = cv2.imencode(".png", rgba)
    return Response(content=buf.tobytes(), media_type="image/png",
                    headers={"Content-Disposition": "attachment; filename=handogram_canvas.png"})


@app.get("/export/svg")
def export_svg():
    elems = []
    with lock:
        for obj in holo_objects:
            col = obj["hex"]
            th = obj["thick"]
            pts = obj["pts"]
            if obj["type"] == "circle" and len(pts) >= 2:
                r = int(math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]))
                elems.append(f'<circle cx="{pts[0][0]}" cy="{pts[0][1]}" r="{r}" stroke="{col}" stroke-width="{th}" fill="none"/>')
            elif obj["type"] == "rect" and len(pts) >= 2:
                w, h = abs(pts[1][0] - pts[0][0]), abs(pts[1][1] - pts[0][1])
                elems.append(f'<rect x="{min(pts[0][0], pts[1][0])}" y="{min(pts[0][1], pts[1][1])}" width="{w}" height="{h}" stroke="{col}" stroke-width="{th}" fill="none"/>')
            elif obj["type"] == "triangle" and len(pts) >= 3:
                p_str = " ".join(f"{p[0]},{p[1]}" for p in pts[:3])
                elems.append(f'<polygon points="{p_str}" stroke="{col}" stroke-width="{th}" fill="none"/>')
            elif obj["type"] == "arrow" and len(pts) >= 2:
                elems.append(f'<line x1="{pts[0][0]}" y1="{pts[0][1]}" x2="{pts[1][0]}" y2="{pts[1][1]}" stroke="{col}" stroke-width="{th}"/>')
            elif len(pts) >= 2:
                d_str = "M " + " L ".join(f"{p[0]},{p[1]}" for p in pts)
                elems.append(f'<path d="{d_str}" stroke="{col}" stroke-width="{th}" fill="none" stroke-linecap="round"/>')

    svg_doc = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">
  <rect width="100%" height="100%" fill="#040812"/>
  {" ".join(elems)}
</svg>'''
    return Response(content=svg_doc, media_type="image/svg+xml",
                    headers={"Content-Disposition": "attachment; filename=handogram_flowchart.svg"})


class ControlReq(BaseModel):
    action: str
    value: int = 0


@app.post("/control")
def control(req: ControlReq):
    if req.action == "set_mode":
        modes = ["CANVAS", "KEYBOARD", "MOUSE"]
        state["system_mode"] = modes[req.value % len(modes)]
    elif req.action == "toggle_shape":
        state["shape_snap"] = not state["shape_snap"]
    elif req.action == "set_ink":
        state["ink_idx"] = req.value % len(INK_PALETTE)
    elif req.action == "set_skel":
        state["skel_idx"] = req.value % len(SKELETON_THEMES)
    elif req.action == "set_text_color":
        state["text_color_idx"] = req.value % len(TEXT_COLORS)
    elif req.action == "set_brush":
        state["brush_size"] = max(3, min(25, req.value))
    elif req.action == "clear":
        state["clear_flag"] = True
    elif req.action == "clear_text":
        state["recognized_text"] = ""
    elif req.action == "analyze":
        state["analyze_flag"] = True
    return {"ok": True}


if __name__ == "__main__":
    print("⚡ HANDOGRAM SPATIAL OS LIVE AT: http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")