import json
import math
import time
from datetime import datetime, timezone
from typing import List, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from app.config import settings
from app.perception.screen_inspector import screen_inspector
from app.schemas import CalibrationPointInput, GazeState


class GazeTracker:
    """
    Real-Time Gaze Tracking Module using MediaPipe FaceMesh (with Iris refinement).
    1. Extracts iris center ratios + head-pose anchor features.
    2. Maps eye features to (X, Y) screen coordinates (supports 9-Point Polynomial Calibration).
    3. Identifies Gaze Target via ScreenInspector.
    4. Tracks Gaze Dwell progress for hands-free selection (Demo 3).
    """

    # MediaPipe FaceMesh Iris & Eye Corner Landmark Indices
    LEFT_IRIS = 468
    RIGHT_IRIS = 473
    LEFT_EYE_INNER = 133
    LEFT_EYE_OUTER = 33
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374
    NOSE_TIP = 1

    def __init__(self) -> None:
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.65,
        )

        self.screen_w, self.screen_h = pyautogui.size()

        # Exponential Moving Average (EMA) smoothing to prevent eye tremor jitter
        self.smooth_alpha: float = 0.22
        self.smooth_x: float = self.screen_w / 2.0
        self.smooth_y: float = self.screen_h / 2.0

        # Raw normalized feature coordinates (used during 9-Point Calibration)
        self.latest_raw_fx: float = 0.5
        self.latest_raw_fy: float = 0.5

        # Dwell Time Tracking (Demo 3: Look & Hold Gaze)
        self.dwell_time_ms: int = settings.DEFAULT_DWELL_TIME_MS
        self._dwell_target: Optional[str] = None
        self._dwell_start_time: float = time.time()
        self._dwell_triggered_for_current: bool = False

        # 9-Point Polynomial Calibration Models
        self.poly = PolynomialFeatures(degree=2)
        self.model_x: Optional[Ridge] = None
        self.model_y: Optional[Ridge] = None
        self.is_calibrated: bool = False
        self._load_calibration()

    def set_dwell_time(self, dwell_ms: int) -> None:
        self.dwell_time_ms = max(400, dwell_ms)

    def process_frame(
        self, rgb_frame: np.ndarray, bgr_canvas: Optional[np.ndarray] = None
    ) -> Tuple[GazeState, bool]:
        """
        Processes an RGB webcam frame and returns (GazeState, dwell_selection_triggered).
        Draws iris indicators onto bgr_canvas if provided.
        """
        now = time.time()
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            self._dwell_target = None
            return (
                GazeState(
                    x=int(self.smooth_x),
                    y=int(self.smooth_y),
                    target=None,
                    confidence=0.0,
                    dwell_progress=0.0,
                    timestamp=now,
                ),
                False,
            )

        face_lms = results.multi_face_landmarks[0].landmark
        h, w = rgb_frame.shape[:2]

        # Extract combined Iris + Head Pose feature vector (fx, fy) in [0.0, 1.0]
        fx, fy, conf = self._extract_gaze_features(face_lms)
        self.latest_raw_fx = fx
        self.latest_raw_fy = fy

        # Draw subtle cyan circles on both irises for the React Dashboard camera preview
        if bgr_canvas is not None:
            for idx in (self.LEFT_IRIS, self.RIGHT_IRIS):
                cx, cy = int(face_lms[idx].x * w), int(face_lms[idx].y * h)
                cv2.circle(bgr_canvas, (cx, cy), 4, (255, 255, 0), -1)

        # Map (fx, fy) to Screen Coordinates (X, Y)
        raw_screen_x, raw_screen_y = self._map_to_screen(fx, fy)

        # Smooth coordinates
        self.smooth_x = (self.smooth_alpha * raw_screen_x) + (
            (1.0 - self.smooth_alpha) * self.smooth_x
        )
        self.smooth_y = (self.smooth_alpha * raw_screen_y) + (
            (1.0 - self.smooth_alpha) * self.smooth_y
        )

        final_x = int(np.clip(self.smooth_x, 0, self.screen_w - 1))
        final_y = int(np.clip(self.smooth_y, 0, self.screen_h - 1))

        # Identify UI target under (final_x, final_y)
        target_name = screen_inspector.resolve_target_at(final_x, final_y)

        # Update Dwell Progress (Demo 3: Gaze Hold Selection)
        dwell_triggered = False
        if target_name == self._dwell_target and target_name is not None:
            elapsed_ms = (now - self._dwell_start_time) * 1000.0
            dwell_progress = min(1.0, elapsed_ms / float(self.dwell_time_ms))
            if dwell_progress >= 1.0 and not self._dwell_triggered_for_current:
                dwell_triggered = True
                self._dwell_triggered_for_current = True
        else:
            self._dwell_target = target_name
            self._dwell_start_time = now
            self._dwell_triggered_for_current = False
            dwell_progress = 0.0

        state = GazeState(
            x=final_x,
            y=final_y,
            target=target_name,
            confidence=round(conf, 3),
            dwell_progress=round(dwell_progress, 2),
            timestamp=now,
        )
        return state, dwell_triggered

    def _extract_gaze_features(self, lm) -> Tuple[float, float, float]:
        """
        Computes normalized horizontal (fx) and vertical (fy) gaze features
        combining relative iris position inside the eye socket + nose head-pose anchor.
        """
        # Left Eye Horizontal & Vertical Ratios
        l_width = max(abs(lm[self.LEFT_EYE_INNER].x - lm[self.LEFT_EYE_OUTER].x), 1e-4)
        l_height = max(abs(lm[self.LEFT_EYE_BOTTOM].y - lm[self.LEFT_EYE_TOP].y), 1e-4)
        l_rx = (lm[self.LEFT_IRIS].x - lm[self.LEFT_EYE_OUTER].x) / l_width
        l_ry = (lm[self.LEFT_IRIS].y - lm[self.LEFT_EYE_TOP].y) / l_height

        # Right Eye Horizontal & Vertical Ratios
        r_width = max(abs(lm[self.RIGHT_EYE_OUTER].x - lm[self.RIGHT_EYE_INNER].x), 1e-4)
        r_height = max(abs(lm[self.RIGHT_EYE_BOTTOM].y - lm[self.RIGHT_EYE_TOP].y), 1e-4)
        r_rx = (lm[self.RIGHT_IRIS].x - lm[self.RIGHT_EYE_INNER].x) / r_width
        r_ry = (lm[self.RIGHT_IRIS].y - lm[self.RIGHT_EYE_TOP].y) / r_height

        iris_x = (l_rx + r_rx) / 2.0
        iris_y = (l_ry + r_ry) / 2.0

        # Blend with Nose Tip position (head yaw/pitch compensation) for comfortable desktop tracking
        nose_x = lm[self.NOSE_TIP].x
        nose_y = lm[self.NOSE_TIP].y

        fx = (0.55 * iris_x) + (0.45 * nose_x)
        fy = (0.55 * iris_y) + (0.45 * nose_y)

        # Eye Aspect Ratio (EAR) check to drop confidence during blinks
        ear = (l_height / l_width + r_height / r_width) / 2.0
        if ear < 0.16:
            conf = 0.45  # Blinking or eyes partially closed
        else:
            conf = min(0.98, 0.86 + min(ear * 0.25, 0.12))

        return float(fx), float(fy), float(conf)

    def _map_to_screen(self, fx: float, fy: float) -> Tuple[float, float]:
        if self.is_calibrated and self.model_x is not None and self.model_y is not None:
            feat_poly = self.poly.transform([[fx, fy]])
            sx = float(self.model_x.predict(feat_poly)[0])
            sy = float(self.model_y.predict(feat_poly)[0])
            return sx, sy

        # Default uncalibrated sensitivity mapping (centers around natural head/eye neutral zone)
        norm_x = np.clip((fx - 0.36) / (0.64 - 0.36), 0.0, 1.0)
        norm_y = np.clip((fy - 0.38) / (0.62 - 0.38), 0.0, 1.0)
        return float(norm_x * self.screen_w), float(norm_y * self.screen_h)

    def calibrate_9_points(self, points: List[CalibrationPointInput]) -> bool:
        """
        Fits a 2nd-order Polynomial Ridge Regression mapping (feature_x, feature_y) -> (screen_x, screen_y)
        from the 9-Point Calibration Grid on the React Dashboard.
        """
        if len(points) < 4:
            return False

        X_raw = np.array([[p.feature_x, p.feature_y] for p in points], dtype=np.float64)
        y_screen_x = np.array([p.screen_x for p in points], dtype=np.float64)
        y_screen_y = np.array([p.screen_y for p in points], dtype=np.float64)

        X_poly = self.poly.fit_transform(X_raw)
        self.model_x = Ridge(alpha=1e-3).fit(X_poly, y_screen_x)
        self.model_y = Ridge(alpha=1e-3).fit(X_poly, y_screen_y)
        self.is_calibrated = True

        # Save calibration weights to JSON
        calib_data = {
            "is_calibrated": True,
            "intercept_x": float(self.model_x.intercept_),
            "coef_x": self.model_x.coef_.tolist(),
            "intercept_y": float(self.model_y.intercept_),
            "coef_y": self.model_y.coef_.tolist(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        settings.CALIBRATION_FILE.write_text(json.dumps(calib_data, indent=2))
        print("[SYNTRA GazeTracker] 9-Point Calibration Matrix saved successfully.")
        return True

    def _load_calibration(self) -> None:
        try:
            if not settings.CALIBRATION_FILE.exists():
                return
            data = json.loads(settings.CALIBRATION_FILE.read_text())
            if not data.get("is_calibrated") or not data.get("coef_x"):
                return

            self.poly.fit([[0.5, 0.5]])
            self.model_x = Ridge()
            self.model_x.intercept_ = np.float64(data["intercept_x"])
            self.model_x.coef_ = np.array(data["coef_x"], dtype=np.float64)

            self.model_y = Ridge()
            self.model_y.intercept_ = np.float64(data["intercept_y"])
            self.model_y.coef_ = np.array(data["coef_y"], dtype=np.float64)
            self.is_calibrated = True
        except Exception:
            self.is_calibrated = False


gaze_tracker = GazeTracker()