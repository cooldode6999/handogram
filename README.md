# 🖐️ Handogram — Spatial Gesture & Vision OS

**Handogram** is an offline-first spatial computing studio that transforms any standard webcam into a touchless holographic workspace. Built with **Python, MediaPipe Tasks API, OpenCV, FastAPI, and local Ollama (`llava:7b`)**, it combines independent object-level air-drawing, real-time geometric shape snapping, a holographic air-keyboard, and touchless OS cursor navigation.

---

## ✨ Core Architecture & Features

### 1. Independent Object-Based Air-Canvas (Dual-Hand)
- **Zero-Drop Stroke Engine:** Uses Exponential Moving Average (EMA) smoothing and a multi-frame drop buffer so fast mid-air strokes never fragment.
- **Object-Level Manipulation:** Every completed stroke locks into an independent spatial object (`circle`, `rect`, `triangle`, `arrow`, or `path`). Grab, drag, and scale individual objects with your **left hand** without disturbing existing elements on the board.
- **Safe Wipe Gesture:** Hold the **Shaka sign (`🤙`)** for 1.0 second (visualized via a radial progress arc) to clear the board — eliminating accidental wipes during normal hand movements.

### 2. Geometric Shape & Flowchart Auto-Beautifier
- Automatically analyzes closed loops and open trajectories using OpenCV convex hull and contour approximation (`approxPolyDP` + circularity scoring) to snap rough hand-drawn circles, boxes, triangles, and lines into clean geometric primitives and directional arrows.
- Export spatial diagrams directly as **transparent `.png` overlays** or **clean `.svg` vector graphics**.

### 3. Local Vision OCR (`llava:7b`)
- Preprocesses drawn strokes via tight bounding-box cropping and morphological dilation before sending frames to a local **Ollama (`llava:7b`)** instance — converting mid-air handwriting into digital text offline and auto-typing into active desktop applications.

### 4. Holographic Air-Keyboard & Spatial Air-Mouse
- **Holo Keyboard:** Translucent on-screen QWERTY overlay with pinch-to-type detection and Web Audio API synthesized click feedback.
- **Air-Mouse:** Maps a padded inner camera zone to full screen coordinates with smooth cursor damping and thumb-to-finger tap gestures for left/right clicks.

---

## 🎮 Gesture Matrix

| Gesture | Mode | Action |
| :--- | :--- | :--- |
| ☝️ **Right Index Up** | Canvas | Draw new independent stroke |
| ✌️ **Peace Sign** | Canvas | Lock stroke & auto-snap geometric shape |
| 🤏 **Left Pinch + Drag** | Canvas | Move nearest holographic object |
| 🙌 **Both Hands Pinch** | Canvas | Scale target object up / down |
| 👌 **Right Pinch** | Canvas | Cycle ink color & 5-finger skeleton theme |
| 🤘 **Index + Pinky** | Canvas | Cycle transcript output text color |
| ✊ **Closed Fist** | Canvas | Trigger local `llava:7b` OCR & auto-type |
| 🤙 **Hold Shaka (1s)** | Canvas | Safe board wipe with radial charge ring |

---

## 🚀 Installation & Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/cooldode6999/handogram.git](https://github.com/cooldode6999/handogram.git)
   cd handogram