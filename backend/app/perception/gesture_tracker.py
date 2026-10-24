import math
import time
from typing import Dict, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from app.schemas import GestureState


class GestureTracker:
    """
    Real-time Hand Gesture Recognition Module using MediaPipe Hands.
    Extracts 21 3D landmarks, classifies gestures with confidence scores,
    and applies Exponential Moving Average (EMA) cursor smoothing.
    """

    def __init__(self) -> None:
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.70,
            min_tracking_confidence=0.70,
        )

        # Screen dimensions for cursor mapping
        self.screen_w, self.screen_h = pyautogui.size()

        # EMA Cursor Smoothing State
        self.smoothing_alpha: float = 0.35
        self.smooth_x: float = self.screen_w / 2.0
        self.smooth_y: float = self.screen_h / 2.0

        # Active bounding margin (so user doesn't have to reach camera edges)
        self.frame_margin: float = 0.15

        # Debounce & Scroll tracking
        self.last_gesture: str = "NONE"
        self.gesture_hold_count: int = 0
        self.prev_scroll_y: Optional[float] = None

        # User Personalization Mapping (Loaded dynamically from DB)
        self.gesture_map: Dict[str, str] = {
            "POINT": "MOVE_CURSOR",
            "PINCH": "CLICK",
            "TWO_FINGERS": "SCROLL",
            "PALM": "STOP",
            "FIST": "GRAB",
            "THUMBS_UP": "CONFIRM",
            "THUMBS_DOWN": "CANCEL",
        }

    def update_gesture_mappings(self, new_map: Dict[str, str]) -> None:
        if new_map:
            self.gesture_map.update(new_map)

    @staticmethod
    def _dist(lm1, lm2) -> float:
        return math.hypot(lm1.x - lm2.x, lm1.y - lm2.y)

    def process_frame(
        self, rgb_frame: np.ndarray, bgr_canvas: Optional[np.ndarray] = None
    ) -> Tuple[GestureState, float]:
        """
        Processes an RGB frame and returns (GestureState, scroll_delta).
        Optionally draws hand landmarks onto bgr_canvas for the dashboard preview.
        """
        now = time.time()
        results = self.hands.process(rgb_frame)
        scroll_delta = 0.0

        if not results.multi_hand_landmarks:
            self.prev_scroll_y = None
            self.last_gesture = "NONE"
            return (
                GestureState(
                    gesture="NONE",
                    mapped_action="NONE",
                    cursor_x=int(self.smooth_x),
                    cursor_y=int(self.smooth_y),
                    confidence=0.0,
                    timestamp=now,
                ),
                0.0,
            )

        hand_lms = results.multi_hand_landmarks[0]
        handedness_score = (
            results.multi_handedness[0].classification[0].score
            if results.multi_handedness
            else 0.90
        )

        if bgr_canvas is not None:
            self.mp_drawing.draw_landmarks(
                bgr_canvas, hand_lms, self.mp_hands.HAND_CONNECTIONS
            )

        lm = hand_lms.landmark

        # Normalize distances by palm scale (Wrist 0 to Middle MCP 9)
        palm_size = max(self._dist(lm[0], lm[9]), 0.05)

        # Map Index Finger Tip (Landmark 8) to Screen Coordinates with active margin box
        norm_x = np.clip(
            (lm[8].x - self.frame_margin) / (1.0 - 2 * self.frame_margin), 0.0, 1.0
        )
        norm_y = np.clip(
            (lm[8].y - self.frame_margin) / (1.0 - 2 * self.frame_margin), 0.0, 1.0
        )
        target_x = norm_x * self.screen_w
        target_y = norm_y * self.screen_h

        # Apply Exponential Moving Average (EMA) filter for silky cursor control
        self.smooth_x = (self.smoothing_alpha * target_x) + (
            (1.0 - self.smoothing_alpha) * self.smooth_x
        )
        self.smooth_y = (self.smoothing_alpha * target_y) + (
            (1.0 - self.smoothing_alpha) * self.smooth_y
        )

        # Classify the exact hand gesture and calculate confidence
        gesture_name, raw_conf = self._classify_landmarks(lm, palm_size)
        final_conf = round(min(1.0, (handedness_score * 0.5) + (raw_conf * 0.5)), 3)

        # Compute vertical scroll delta if TWO_FINGERS gesture is active
        if gesture_name == "TWO_FINGERS":
            mid_y = (lm[8].y + lm[12].y) / 2.0
            if self.prev_scroll_y is not None:
                dy = self.prev_scroll_y - mid_y
                if abs(dy) > 0.008:
                    scroll_delta = dy * 1200.0
            self.prev_scroll_y = mid_y
        else:
            self.prev_scroll_y = None

        mapped_action = self.gesture_map.get(gesture_name, "NONE")

        state = GestureState(
            gesture=gesture_name,
            mapped_action=mapped_action,
            cursor_x=int(self.smooth_x),
            cursor_y=int(self.smooth_y),
            confidence=final_conf,
            timestamp=now,
        )
        return state, scroll_delta

    def _classify_landmarks(self, lm, palm_size: float) -> Tuple[str, float]:
        """
        Deterministic geometric classifier for the 7 SYNTRA gestures:
        1. PINCH       -> Thumb tip (4) & Index tip (8) touching
        2. THUMBS_UP   -> Thumb extended straight up, 4 fingers curled
        3. THUMBS_DOWN -> Thumb extended straight down, 4 fingers curled
        4. PALM        -> All 5 fingers open and extended
        5. TWO_FINGERS -> Index (8) & Middle (12) extended, Ring & Pinky curled
        6. POINT       -> Only Index (8) extended
        7. FIST        -> All 4 fingers curled into palm
        """
        # Finger extension checks (tip higher than PIP joint in image Y coordinates)
        index_up = lm[8].y < lm[6].y - 0.015
        middle_up = lm[12].y < lm[10].y - 0.015
        ring_up = lm[16].y < lm[14].y - 0.015
        pinky_up = lm[20].y < lm[18].y - 0.015

        # Finger curled checks (tip close to wrist or below PIP)
        index_curled = lm[8].y > lm[6].y
        middle_curled = lm[12].y > lm[10].y
        ring_curled = lm[16].y > lm[14].y
        pinky_curled = lm[20].y > lm[18].y
        four_fingers_curled = index_curled and middle_curled and ring_curled and pinky_curled

        # 1. PINCH Detection (Thumb Tip 4 to Index Tip 8 normalized by palm size)
        pinch_ratio = self._dist(lm[4], lm[8]) / palm_size
        if pinch_ratio < 0.32 and (not ring_up) and (not pinky_up):
            conf = np.clip(1.0 - (pinch_ratio / 0.35), 0.85, 0.99)
            return "PINCH", float(conf)

        # 2. THUMBS_UP & 3. THUMBS_DOWN Detection
        thumb_vertical_dist = abs(lm[4].y - lm[2].y) / palm_size
        if four_fingers_curled and thumb_vertical_dist > 0.35:
            if lm[4].y < lm[2].y - 0.03 and lm[4].y < lm[5].y:
                return "THUMBS_UP", 0.95
            if lm[4].y > lm[2].y + 0.03 and lm[4].y > lm[17].y:
                return "THUMBS_DOWN", 0.94

        # 4. OPEN PALM Detection (All 4 fingers + Thumb spread)
        thumb_open = self._dist(lm[4], lm[17]) / palm_size > 0.85
        if index_up and middle_up and ring_up and pinky_up and thumb_open:
            return "PALM", 0.97

        # 5. TWO FINGERS (Scroll) Detection
        if index_up and middle_up and ring_curled and pinky_curled:
            return "TWO_FINGERS", 0.93

        # 6. INDEX FINGER (Point / Cursor Movement)
        if index_up and middle_curled and ring_curled and pinky_curled:
            return "POINT", 0.95

        # 7. FIST (Grab / Hold)
        if four_fingers_curled and (self._dist(lm[4], lm[10]) / palm_size < 0.55):
            return "FIST", 0.92

        return "NONE", 0.0


gesture_tracker = GestureTracker()