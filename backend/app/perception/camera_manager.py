import base64
import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np
from app.config import settings


class CameraManager:
    """
    Singleton threaded webcam producer.
    Captures 720p @ 30 FPS once and shares the latest RGB/BGR frame
    between the Gaze Tracking Module and Hand Gesture Module without I/O blocking.
    """

    def __init__(self) -> None:
        self.camera_index: int = settings.CAMERA_INDEX
        self.width: int = settings.CAMERA_WIDTH
        self.height: int = settings.CAMERA_HEIGHT
        self.fps: int = settings.CAMERA_FPS

        self._cap: Optional[cv2.VideoCapture] = None
        self._running: bool = False
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

        self._latest_bgr: Optional[np.ndarray] = None
        self._latest_rgb: Optional[np.ndarray] = None
        self._annotated_bgr: Optional[np.ndarray] = None
        self._frame_timestamp: float = 0.0

    @property
    def is_running(self) -> bool:
        return self._running and self._cap is not None and self._cap.isOpened()

    def start(self) -> bool:
        if self._running:
            return True

        # Use DirectShow on Windows for fast webcam initialization, default on Linux
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            print(f"[SYNTRA CameraManager] ERROR: Could not open webcam index {self.camera_index}")
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        self._cap = cap
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"[SYNTRA CameraManager] Webcam started ({self.width}x{self.height} @ {self.fps} FPS)")
        return True

    def _capture_loop(self) -> None:
        target_interval = 1.0 / max(self.fps, 1)
        while self._running and self._cap is not None:
            loop_start = time.time()
            ret, frame = self._cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Flip horizontally for natural mirror interaction
            flipped_bgr = cv2.flip(frame, 1)
            flipped_rgb = cv2.cvtColor(flipped_bgr, cv2.COLOR_BGR2RGB)
            now = time.time()

            with self._lock:
                self._latest_bgr = flipped_bgr
                self._latest_rgb = flipped_rgb
                self._frame_timestamp = now

            elapsed = time.time() - loop_start
            sleep_time = target_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """Returns (bgr_frame, rgb_frame, timestamp) safely across threads."""
        with self._lock:
            if self._latest_bgr is None or self._latest_rgb is None:
                return None, None, 0.0
            return self._latest_bgr.copy(), self._latest_rgb.copy(), self._frame_timestamp

    def update_annotated_frame(self, annotated_bgr: np.ndarray) -> None:
        """Stores overlay frame with landmarks drawn for the React Dashboard HUD."""
        with self._lock:
            self._annotated_bgr = annotated_bgr

    def get_jpeg_base64(self, width: int = 420) -> Optional[str]:
        """Encodes a lightweight JPEG base64 string for 15Hz WebSocket dashboard preview."""
        with self._lock:
            source = self._annotated_bgr if self._annotated_bgr is not None else self._latest_bgr
            if source is None:
                return None
            frame = source.copy()

        h, w = frame.shape[:2]
        scale = width / float(w)
        resized = cv2.resize(frame, (width, int(h * scale)), interpolation=cv2.INTER_AREA)
        success, buffer = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
        if not success:
            return None
        return base64.b64encode(buffer).decode("utf-8")

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        print("[SYNTRA CameraManager] Webcam stopped.")


camera_manager = CameraManager()