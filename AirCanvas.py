"""
AirCanvas v3
======================
Hand-tracking drawing app with full gesture UI.

Gestures:
  - Index finger only          → Draw
  - Index + middle up          → Hover / select toolbar items
  - Fist (all fingers curled)  → Eraser
  - Hold finger on toolbar ~1s → Activate that item

Keyboard shortcuts (fallback):
  - Z          → Undo last stroke
  - Y          → Redo
  - C          → Clear canvas
  - S          → Save canvas as PNG
  - 1–5        → Pick colour
  - [ / ]      → Brush size down / up
  - Q / ESC    → Quit

Requirements:
  pip install opencv-python mediapipe numpy scipy
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math
from collections import deque
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any, Callable

try:
    from scipy.interpolate import splprep, splev
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ═══════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════

CAMERA_INDEX   = 0
FRAME_WIDTH    = 1280
FRAME_HEIGHT   = 720

BRUSH_SIZE     = 8
MIN_BRUSH      = 2
MAX_BRUSH      = 50
ERASER_SIZE    = 40

FIST_DEBOUNCE  = 3          # frames before eraser activates
HOVER_DWELL    = 1.0        # seconds finger must hover to trigger toolbar item
SMOOTH_ALPHA   = 0.35       # exponential smoothing weight (lower = smoother)
SMOOTH_WINDOW  = 5          # moving-average window for position history

UNDO_LIMIT     = 30         # max undo steps stored

TOOLBAR_H      = 80         # height of top toolbar in pixels
TOOLBAR_Y      = 0          # top-left y of toolbar

# Shape recognition: if stroke bbox is mostly circular/square, snap it
SHAPE_MIN_POINTS = 20       # minimum stroke points to attempt recognition
SHAPE_ENABLED    = True

# ── Colour palette (BGR) ──────────────────────────────
COLORS: List[Tuple[str, Tuple[int, int, int]]] = [
    ("Red",    (0,   30,  220)),
    ("Orange", (0,   140, 255)),
    ("Yellow", (0,   220, 220)),
    ("Green",  (30,  200, 30 )),
    ("Blue",   (220, 80,  20 )),
    ("Purple", (200, 40,  150)),
    ("White",  (255, 255, 255)),
    ("Black",  (10,  10,  10 )),
]
COLOR_NAMES = [c[0] for c in COLORS]
COLOR_LIST  = [c[1] for c in COLORS]

# MediaPipe indices
TIP_IDS = [4, 8, 12, 16, 20]
MCP_IDS = [2, 5,  9, 13, 17]
PIP_IDS = [6, 10, 14, 18]

# ═══════════════════════════════════════════════════════
#  TOOLBAR DEFINITION
# ═══════════════════════════════════════════════════════

def build_toolbar(frame_w: int, toolbar_h: int) -> List[Dict[str, Any]]:
    """
    Returns a list of toolbar item dicts.
    Each item: { id, label, x1, y1, x2, y2, type, value, bgr, tx, ty }
    """
    items = []
    # Colour swatches — left block
    swatch_w = 60
    gap = 4
    x = 10
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.38
    thick = 1

    for i, (name, bgr) in enumerate(COLORS):
        items.append({
            "id": f"color_{i}", "label": name,
            "x1": x, "y1": 6, "x2": x + swatch_w, "y2": toolbar_h - 6,
            "type": "color", "value": i,
            "bgr": bgr,
        })
        x += swatch_w + gap

    # Separator gap
    x += 20

    # Tool buttons
    btn_w = 90
    tools = [
        ("id_eraser",     "Eraser",  "tool",       "eraser"),
        ("id_clear",      "Clear",   "action",     "clear"),
        ("id_save",       "Save",    "action",     "save"),
        ("id_shape_tog",  "Shape",   "toggle",     "shape"),
        ("id_undo",       "Undo",    "action",     "undo"),
        ("id_redo",       "Redo",    "action",     "redo"),
    ]
    for tid, label, ttype, tval in tools:
        items.append({
            "id": tid, "label": label,
            "x1": x, "y1": 6, "x2": x + btn_w, "y2": toolbar_h - 6,
            "type": ttype, "value": tval,
            "bgr": None,
        })
        x += btn_w + gap

    # Brush size slider — two buttons
    x += 10
    for tid, label, tval in [("id_bdown", "B-", "brush_down"),
                               ("id_bup",   "B+", "brush_up")]:
        items.append({
            "id": tid, "label": label,
            "x1": x, "y1": 6, "x2": x + 50, "y2": toolbar_h - 6,
            "type": "action", "value": tval,
            "bgr": None,
        })
        x += 54

    # Precalculate text positions for all items
    for item in items:
        (tw, th), _ = cv2.getTextSize(item["label"], font, scale, thick)
        item["tx"] = item["x1"] + ((item["x2"] - item["x1"]) - tw) // 2
        item["ty"] = item["y1"] + TOOLBAR_Y + ((item["y2"] - item["y1"]) + th) // 2

    return items


# ═══════════════════════════════════════════════════════
#  HAND DETECTOR
# ═══════════════════════════════════════════════════════

class HandDetector:
    def __init__(self, max_hands: int = 1, det_conf: float = 0.75, track_conf: float = 0.75):
        self.mp_hands  = mp.solutions.hands
        self.mp_draw   = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )
        self.results = None

    def process(self, frame_rgb: np.ndarray, draw_on: Optional[np.ndarray] = None) -> None:
        self.results = self.hands.process(frame_rgb)
        if draw_on is not None and self.results.multi_hand_landmarks:
            for lm in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    draw_on, lm,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style(),
                )

    def landmarks(self, frame: np.ndarray, hand_idx: int = 0) -> List[Tuple[int, int, int]]:
        h, w = frame.shape[:2]
        if not (self.results and self.results.multi_hand_landmarks):
            return []
        if hand_idx >= len(self.results.multi_hand_landmarks):
            return []
        hand = self.results.multi_hand_landmarks[hand_idx]
        return [(i, int(lm.x * w), int(lm.y * h))
                for i, lm in enumerate(hand.landmark)]

    def fingers_up(self, lms: List[Tuple[int, int, int]]) -> List[bool]:
        if len(lms) < 21:
            return [False] * 5
        lm = {p[0]: (p[1], p[2]) for p in lms}
        flags = [lm[TIP_IDS[0]][0] < lm[MCP_IDS[0]][0]]
        for i in range(1, 5):
            flags.append(lm[TIP_IDS[i]][1] < lm[MCP_IDS[i]][1])
        return flags

    def is_fist(self, lms: List[Tuple[int, int, int]], threshold: float = 0.8) -> bool:
        if len(lms) < 21:
            return False
        lm = {p[0]: (p[1], p[2]) for p in lms}
        curled = sum(
            1 for tip, pip in zip([8, 12, 16, 20], PIP_IDS)
            if lm[tip][1] > lm[pip][1]
        )
        return (curled / 4) >= threshold


# ═══════════════════════════════════════════════════════
#  STROKE SMOOTHER
# ═══════════════════════════════════════════════════════

class StrokeSmoother:
    """
    Three-layer smoothing pipeline:
      1. Exponential moving average (low-latency noise rejection)
      2. Weighted previous position (stabilises micro-jitter)
      3. Moving-average window over recent positions
    """
    def __init__(self, alpha: float = SMOOTH_ALPHA, window: int = SMOOTH_WINDOW):
        self.alpha   = alpha
        self.window  = window
        self.ema_x: Optional[float] = None
        self.ema_y: Optional[float] = None
        self.history: deque = deque(maxlen=window)

    def reset(self) -> None:
        self.ema_x   = None
        self.ema_y   = None
        self.history.clear()

    def smooth(self, raw_x: int, raw_y: int) -> Tuple[int, int]:
        # Layer 1: EMA
        if self.ema_x is None or self.ema_y is None:
            self.ema_x, self.ema_y = float(raw_x), float(raw_y)
        else:
            self.ema_x = self.alpha * raw_x + (1 - self.alpha) * self.ema_x
            self.ema_y = self.alpha * raw_y + (1 - self.alpha) * self.ema_y

        # Layer 2: weighted blend with previous position
        self.history.append((self.ema_x, self.ema_y))

        # Layer 3: moving average over history
        avg_x = sum(p[0] for p in self.history) / len(self.history)
        avg_y = sum(p[1] for p in self.history) / len(self.history)

        return int(round(avg_x)), int(round(avg_y))


# ═══════════════════════════════════════════════════════
#  SHAPE RECOGNISER
# ═══════════════════════════════════════════════════════

class ShapeRecogniser:
    """
    Analyses a completed stroke (list of (x,y) points) and tries to
    classify it as a circle, rectangle, triangle, or line.
    Returns (shape_name, corrected_canvas_fn) or (None, None).
    """

    @staticmethod
    def _bounding_box(pts: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return min(xs), min(ys), max(xs), max(ys)

    @staticmethod
    def _perimeter(pts: List[Tuple[int, int]]) -> float:
        total = 0.0
        for i in range(len(pts)):
            dx = pts[i][0] - pts[i-1][0]
            dy = pts[i][1] - pts[i-1][1]
            total += math.hypot(dx, dy)
        return total

    def recognise(self, pts: List[Tuple[int, int]]) -> Tuple[Optional[str], Optional[Callable]]:
        if len(pts) < SHAPE_MIN_POINTS:
            return None, None

        pts_np = np.array(pts, dtype=np.float32)
        hull   = cv2.convexHull(pts_np.astype(np.int32))
        hull_area    = cv2.contourArea(hull)
        stroke_area  = cv2.contourArea(pts_np.astype(np.int32))
        perimeter    = self._perimeter(pts)
        x1, y1, x2, y2 = self._bounding_box(pts)
        w, h = x2 - x1, y2 - y1
        if w < 10 or h < 10:
            return None, None

        # ── Circularity: 4π·area / perimeter² (= 1 for perfect circle) ──
        hull_perimeter = cv2.arcLength(hull, True)
        if hull_perimeter > 0:
            circularity = (4 * math.pi * hull_area) / (hull_perimeter ** 2)
        else:
            circularity = 0

        # ── Aspect ratio ──
        aspect = w / h if h > 0 else 1

        # ── Vertex count approximation ──
        epsilon = 0.05 * hull_perimeter
        approx  = cv2.approxPolyDP(hull, epsilon, True)
        n_verts = len(approx)

        # ── Openness: distance between first and last point ──
        start_end_dist = math.hypot(pts[0][0] - pts[-1][0],
                                    pts[0][1] - pts[-1][1])
        closed = start_end_dist < (perimeter * 0.15)

        # Decision tree
        if circularity > 0.8 and closed:
            return "circle", self._make_circle(x1, y1, x2, y2)

        if n_verts in (4, 5, 6) and closed and 0.75 <= aspect <= 1.33:
            return "square", self._make_rect(x1, y1, x2, y2)

        if n_verts in (4, 5, 6) and closed and (aspect < 0.75 or aspect > 1.33):
            return "rectangle", self._make_rect(x1, y1, x2, y2)

        if n_verts == 3 and closed:
            return "triangle", self._make_triangle(approx)

        if not closed and perimeter > 0:
            linearity = math.hypot(pts[0][0] - pts[-1][0],
                                   pts[0][1] - pts[-1][1]) / perimeter
            if linearity > 0.85:
                return "line", self._make_line(pts[0], pts[-1])

        return None, None

    @staticmethod
    def _make_circle(x1: int, y1: int, x2: int, y2: int) -> Callable:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        r  = max((x2 - x1), (y2 - y1)) // 2
        def draw(canvas: np.ndarray, color: Tuple[int, int, int], thick: int):
            cv2.circle(canvas, (cx, cy), r, color, thick)
        return draw

    @staticmethod
    def _make_rect(x1: int, y1: int, x2: int, y2: int) -> Callable:
        def draw(canvas: np.ndarray, color: Tuple[int, int, int], thick: int):
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thick)
        return draw

    @staticmethod
    def _make_triangle(approx: np.ndarray) -> Callable:
        pts = approx.reshape(-1, 2).tolist()
        # If approxPolyDP gave 4 points pick 3 most spaced
        if len(pts) > 3:
            pts = pts[:3]
        tri = np.array(pts, dtype=np.int32)
        def draw(canvas: np.ndarray, color: Tuple[int, int, int], thick: int):
            cv2.polylines(canvas, [tri], True, color, thick)
        return draw

    @staticmethod
    def _make_line(p1: Tuple[int, int], p2: Tuple[int, int]) -> Callable:
        def draw(canvas: np.ndarray, color: Tuple[int, int, int], thick: int):
            cv2.line(canvas, p1, p2, color, thick)
        return draw


# ═══════════════════════════════════════════════════════
#  CANVAS MANAGER  (undo / redo)
# ═══════════════════════════════════════════════════════

class CanvasManager:
    """
    Wraps a numpy canvas and provides snapshot-based undo/redo.
    A snapshot is saved at the END of each stroke (pen-up event).
    """
    def __init__(self, h: int, w: int):
        self.h, self.w  = h, w
        self.canvas     = np.zeros((h, w, 3), dtype=np.uint8)
        self._undo_stack: deque = deque(maxlen=UNDO_LIMIT)
        self._redo_stack: deque = deque(maxlen=UNDO_LIMIT)

    def snapshot(self) -> None:
        """Call when a stroke finishes (finger lifts)."""
        self._undo_stack.append(self.canvas.copy())
        self._redo_stack.clear()

    def undo(self) -> None:
        if not self._undo_stack:
            return
        self._redo_stack.append(self.canvas.copy())
        self.canvas[:] = self._undo_stack.pop()

    def redo(self) -> None:
        if not self._redo_stack:
            return
        self._undo_stack.append(self.canvas.copy())
        self.canvas[:] = self._redo_stack.pop()

    def clear(self) -> None:
        self.snapshot()
        self.canvas[:] = 0

    def save(self) -> str:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"painting_{ts}.png"
        cv2.imwrite(path, self.canvas)
        print(f"Saved → {path}")
        return path


# ═══════════════════════════════════════════════════════
#  STROKE INTERPOLATION (spline)
# ═══════════════════════════════════════════════════════

def interpolate_stroke(pts: List[Tuple[int, int]], color: Tuple[int, int, int], thickness: int, canvas: np.ndarray) -> None:
    """
    Draw a smooth spline through pts onto canvas.
    Falls back to polylines if scipy not available or too few points.
    """
    if len(pts) < 4:
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i-1], pts[i], color, thickness)
        return

    if SCIPY_AVAILABLE and len(pts) >= 4:
        try:
            xs = np.array([p[0] for p in pts], dtype=float)
            ys = np.array([p[1] for p in pts], dtype=float)
            # Remove duplicate consecutive points (splprep requirement)
            mask = np.ones(len(xs), dtype=bool)
            for i in range(1, len(xs)):
                if xs[i] == xs[i-1] and ys[i] == ys[i-1]:
                    mask[i] = False
            xs, ys = xs[mask], ys[mask]
            
            if len(xs) < 4:
                raise ValueError("too few unique points")
            
            k = min(3, len(xs) - 1)
            tck, _ = splprep([xs, ys], s=len(xs) * 2, k=k)
            u_new  = np.linspace(0, 1, max(len(xs) * 4, 60))
            sx, sy = splev(u_new, tck)
            
            spline_pts = [(int(round(x)), int(round(y)))
                          for x, y in zip(sx, sy)]
            for i in range(1, len(spline_pts)):
                cv2.line(canvas, spline_pts[i-1], spline_pts[i],
                         color, thickness)
            return
        except (ValueError, TypeError) as e:
            # Fall back to polyline
            pass
        except Exception as e:
            print(f"Unexpected interpolation error: {e}")

    # Fallback: simple polyline
    pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [pts_np], False, color, thickness)


# ═══════════════════════════════════════════════════════
#  APP CLASS
# ═══════════════════════════════════════════════════════

class AirCanvasApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
        if not self.cap.isOpened():
            print("Error: cannot open webcam.")
            self.running = False
            return
            
        self.running = True
        self.detector   = HandDetector()
        self.smoother   = StrokeSmoother()
        self.recogniser = ShapeRecogniser()
        self.mgr        = CanvasManager(FRAME_HEIGHT, FRAME_WIDTH)
        self.toolbar    = build_toolbar(FRAME_WIDTH, TOOLBAR_H)

        # App State
        self.active_color_idx = 0
        self.brush_size       = BRUSH_SIZE
        self.mode             = "Hover"
        self.shape_enabled    = SHAPE_ENABLED

        self.fist_count       = 0       
        self.prev_x           = 0
        self.prev_y           = 0
        self.was_drawing      = False   

        self.current_stroke: List[Tuple[int, int]] = []      

        # Toolbar hover / dwell
        self.hover_item_id: Optional[str] = None
        self.dwell_start: Optional[float] = None

    def _render_toolbar(self, frame: np.ndarray, dwell_progress: float) -> None:
        """Draw the toolbar strip onto frame (in-place)."""
        cv2.rectangle(frame, (0, TOOLBAR_Y), (FRAME_WIDTH, TOOLBAR_Y + TOOLBAR_H),
                      (30, 30, 30), -1)

        font     = cv2.FONT_HERSHEY_SIMPLEX
        scale    = 0.38
        thick    = 1

        for item in self.toolbar:
            x1, y1, x2, y2 = item["x1"], item["y1"] + TOOLBAR_Y, \
                              item["x2"], item["y2"] + TOOLBAR_Y

            if item["type"] == "color":
                fill = item["bgr"]
                is_active = (item["value"] == self.active_color_idx)
            elif item["type"] == "tool" and item["value"] == "eraser":
                fill = (80, 80, 80)
                is_active = False
            elif item["type"] == "toggle" and item["value"] == "shape":
                fill = (40, 120, 40) if self.shape_enabled else (60, 60, 60)
                is_active = self.shape_enabled
            else:
                fill = (55, 55, 55)
                is_active = False

            cv2.rectangle(frame, (x1, y1), (x2, y2), fill, -1)

            border_col = (255, 255, 255) if is_active else (100, 100, 100)
            cv2.rectangle(frame, (x1, y1), (x2, y2), border_col, 1 if not is_active else 2)

            if item["id"] == self.hover_item_id and dwell_progress > 0:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                r      = min((x2 - x1), (y2 - y1)) // 2 - 2
                angle  = int(360 * dwell_progress)
                cv2.ellipse(frame, (cx, cy), (r, r), -90, 0, angle,
                            (0, 220, 180), 2)

            txt_col  = (0, 0, 0) if item["type"] == "color" else (210, 210, 210)
            cv2.putText(frame, item["label"], (item["tx"], item["ty"]), font, scale, txt_col, thick, cv2.LINE_AA)

        bx = FRAME_WIDTH - 55
        by = TOOLBAR_Y + TOOLBAR_H // 2
        cv2.circle(frame, (bx, by), self.brush_size, COLOR_LIST[self.active_color_idx], -1)
        cv2.circle(frame, (bx, by), self.brush_size, (180, 180, 180), 1)
        cv2.putText(frame, f"{self.brush_size}px", (bx - 14, by + self.brush_size + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

    def _render_status(self, frame: np.ndarray) -> None:
        sy = FRAME_HEIGHT - 30
        color_name = COLOR_NAMES[self.active_color_idx]
        undo_count = len(self.mgr._undo_stack)
        redo_count = len(self.mgr._redo_stack)
        
        text = (f"  {self.mode}  |  {color_name}  |  Shape:{'ON' if self.shape_enabled else 'OFF'}"
                f"  |  Undo:{undo_count}  Redo:{redo_count}"
                f"  |  Z=undo  Y=redo  S=save  Q=quit")
        bar_w = min(FRAME_WIDTH, len(text) * 8 + 16)
        cv2.rectangle(frame, (0, FRAME_HEIGHT - 34), (bar_w, FRAME_HEIGHT),
                      (20, 20, 20), -1)
        col = (80, 80, 220) if self.mode == "Erasing" else \
              (80, 220, 80) if self.mode == "Drawing" else \
              (180, 180, 180)
        cv2.putText(frame, text, (8, sy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)

    def _toolbar_hit(self, x: int, y: int) -> Optional[Dict[str, Any]]:
        for item in self.toolbar:
            if item["x1"] <= x <= item["x2"] and \
               (item["y1"] + TOOLBAR_Y) <= y <= (item["y2"] + TOOLBAR_Y):
                return item
        return None

    def _execute_toolbar_action(self, item: Dict[str, Any]) -> None:
        t = item["type"]
        v = item["value"]

        if t == "color":
            self.active_color_idx = v
            print(f"Color → {COLOR_NAMES[v]}")
        elif t == "tool" and v == "eraser":
            print("Use fist gesture to erase")
        elif t == "action":
            if v == "clear":
                self.mgr.clear()
            elif v == "save":
                self.mgr.save()
            elif v == "undo":
                self.mgr.undo()
            elif v == "redo":
                self.mgr.redo()
            elif v == "brush_down":
                self.brush_size = max(MIN_BRUSH, self.brush_size - 2)
            elif v == "brush_up":
                self.brush_size = min(MAX_BRUSH, self.brush_size + 2)
        elif t == "toggle" and v == "shape":
            self.shape_enabled = not self.shape_enabled
            print(f"Shape recognition {'ON' if self.shape_enabled else 'OFF'}")

    def _finish_stroke(self, color: Tuple[int, int, int]) -> None:
        if not self.current_stroke:
            return

        thick = self.brush_size * 2

        if self.shape_enabled:
            shape_name, draw_fn = self.recogniser.recognise(self.current_stroke)
            if draw_fn is not None:
                # Remove the raw freehand drawing from the live canvas
                if self.mgr._undo_stack:
                    self.mgr.canvas[:] = self.mgr._undo_stack[-1].copy()
                draw_fn(self.mgr.canvas, color, thick)
                print(f"Shape recognised: {shape_name}")
                self.current_stroke = []
                return

        # Also remove raw stroke before smoothing if no shape recognized
        if self.mgr._undo_stack:
            self.mgr.canvas[:] = self.mgr._undo_stack[-1].copy()
        interpolate_stroke(self.current_stroke, color, thick, self.mgr.canvas)
        self.current_stroke = []

    def _handle_keys(self) -> bool:
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            return False
        elif key in (ord('z'), ord('Z')):
            self.mgr.undo()
        elif key in (ord('y'), ord('Y')):
            self.mgr.redo()
        elif key in (ord('c'), ord('C')):
            self.mgr.clear()
        elif key in (ord('s'), ord('S')):
            self.mgr.save()
        elif ord('1') <= key <= ord('8'):
            self.active_color_idx = key - ord('1')
        elif key == ord('['):
            self.brush_size = max(MIN_BRUSH, self.brush_size - 2)
        elif key == ord(']'):
            self.brush_size = min(MAX_BRUSH, self.brush_size + 2)
        return True

    def run(self) -> None:
        if not self.running:
            return
            
        print("AirCanvas v3 ready.")
        print(f"Scipy available: {SCIPY_AVAILABLE}  (smooth spline {'ON' if SCIPY_AVAILABLE else 'OFF'})")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame     = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ── Hand detection ───────────────────────────
            self.detector.process(frame_rgb, draw_on=frame)
            lms = self.detector.landmarks(frame)

            if lms:
                fingers    = self.detector.fingers_up(lms)
                raw_ix, raw_iy = lms[8][1], lms[8][2]

                # ── Fist debounce ────────────────────────
                if self.detector.is_fist(lms):
                    self.fist_count = min(self.fist_count + 1, FIST_DEBOUNCE)
                else:
                    self.fist_count = max(self.fist_count - 1, 0)
                is_erasing = self.fist_count >= FIST_DEBOUNCE

                # ── Smooth finger position ───────────────
                sx, sy = self.smoother.smooth(raw_ix, raw_iy)
                color = COLOR_LIST[self.active_color_idx]

                # ── Toolbar zone (finger in top bar) ─────
                in_toolbar = sy < TOOLBAR_H

                if in_toolbar and not is_erasing:
                    self.mode = "Toolbar"
                    if self.was_drawing:
                        if self.current_stroke:
                            self._finish_stroke(color)
                        self.smoother.reset()
                        self.was_drawing = False
                    self.prev_x = self.prev_y = 0

                    hit = self._toolbar_hit(sx, sy)
                    if hit:
                        if hit["id"] != self.hover_item_id:
                            self.hover_item_id = hit["id"]
                            self.dwell_start   = time.time()
                        else:
                            elapsed  = time.time() - (self.dwell_start or time.time())
                            progress = min(elapsed / HOVER_DWELL, 1.0)
                            if elapsed >= HOVER_DWELL:
                                self._execute_toolbar_action(hit)
                                self.hover_item_id = None
                                self.dwell_start   = None
                                progress      = 0.0
                    else:
                        self.hover_item_id = None
                        self.dwell_start   = None
                        progress      = 0.0

                elif is_erasing:
                    self.mode = "Erasing"
                    if self.was_drawing:
                        if self.current_stroke:
                            self._finish_stroke(color)
                        self.mgr.snapshot()
                        self.was_drawing = False
                    self.smoother.reset()
                    self.prev_x = self.prev_y = 0
                    cv2.circle(self.mgr.canvas, (sx, sy), ERASER_SIZE, (0, 0, 0), -1)
                    cv2.circle(frame, (sx, sy), ERASER_SIZE, (255, 255, 255), 2)
                    self.hover_item_id = None
                    progress      = 0.0

                elif fingers[1] and not fingers[2]:
                    self.mode = "Drawing"
                    self.hover_item_id = None
                    progress      = 0.0

                    if not self.was_drawing:
                        self.mgr.snapshot()
                        self.prev_x, self.prev_y = sx, sy
                        self.current_stroke = [(sx, sy)]
                        self.was_drawing    = True
                    else:
                        if self.prev_x != 0 or self.prev_y != 0:
                            cv2.line(self.mgr.canvas, (self.prev_x, self.prev_y), (sx, sy),
                                     color, self.brush_size * 2)
                        self.current_stroke.append((sx, sy))
                        self.prev_x, self.prev_y = sx, sy

                else:
                    if self.was_drawing and self.current_stroke:
                        self._finish_stroke(color)
                        self.smoother.reset()
                    self.mode      = "Hover"
                    self.was_drawing = False
                    self.prev_x = self.prev_y = 0
                    self.hover_item_id = None
                    progress      = 0.0

                # ── Cursor feedback ──────────────────────
                if not in_toolbar:
                    if is_erasing:
                        cv2.circle(frame, (sx, sy), ERASER_SIZE, (200, 200, 200), 2)
                    elif self.mode == "Drawing":
                        cv2.circle(frame, (sx, sy), self.brush_size, color, -1)
                        cv2.circle(frame, (sx, sy), self.brush_size + 2, (255, 255, 255), 1)
                    else:
                        cv2.circle(frame, (sx, sy), self.brush_size + 2, color, 1)
            else:
                self.mode = "No hand"
                self.fist_count = 0
                if self.was_drawing and self.current_stroke:
                    self._finish_stroke(COLOR_LIST[self.active_color_idx])
                    self.smoother.reset()
                self.was_drawing   = False
                self.hover_item_id = None
                progress      = 0.0
                self.prev_x = self.prev_y = 0

            # ── Composite canvas onto frame ──────────────
            canvas_gray = cv2.cvtColor(self.mgr.canvas, cv2.COLOR_BGR2GRAY)
            _, mask     = cv2.threshold(canvas_gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv    = cv2.bitwise_not(mask)
            frame_bg    = cv2.bitwise_and(frame, frame, mask=mask_inv)
            canvas_fg   = cv2.bitwise_and(self.mgr.canvas, self.mgr.canvas, mask=mask)
            combined    = cv2.add(frame_bg, canvas_fg)

            # ── Draw toolbar ─────────────────────────────
            dwell_prog = 0.0
            if self.hover_item_id and self.dwell_start:
                dwell_prog = min((time.time() - self.dwell_start) / HOVER_DWELL, 1.0)
            self._render_toolbar(combined, dwell_prog)

            # ── Status bar ───────────────────────────────
            self._render_status(combined)

            cv2.imshow("AirCanvas v3", combined)

            # ── Keyboard controls ────────────────────────
            if not self._handle_keys():
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print("AirCanvas v3 closed.")


# ═══════════════════════════════════════════════════════
#  MAIN ENTRY
# ═══════════════════════════════════════════════════════

def main():
    app = AirCanvasApp()
    app.run()

if __name__ == "__main__":
    main()