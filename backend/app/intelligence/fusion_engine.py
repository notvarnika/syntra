from bisect import bisect_right
from collections import deque
import logging
import threading
import time
from typing import Deque, Dict, List, Optional, Tuple

from app.config import settings
from app.intelligence.context_engine import context_engine
from app.intelligence.intent_parser import intent_parser
from app.schemas import GazeState, GestureState, StructuredIntent, VoiceState

logger = logging.getLogger(__name__)


class FusionEvent:
    __slots__ = ("modality", "timestamp", "state", "confidence")

    def __init__(self, modality: str, timestamp: float, state: object, confidence: float) -> None:
        self.modality = modality
        self.timestamp = float(timestamp)
        self.state = state
        self.confidence = max(0.0, min(1.0, float(confidence)))


class MultimodalFusionEngine:
    """
    Pure, thread-safe Temporal Multimodal Fusion Engine.
    Maintains a timestamp-ordered sliding window and computes normalized weighted confidence scores.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.window_seconds: float = settings.TEMPORAL_WINDOW_SECONDS
        self.w_gaze: float = settings.DEFAULT_GAZE_WEIGHT
        self.w_gesture: float = settings.DEFAULT_GESTURE_WEIGHT
        self.w_voice: float = settings.DEFAULT_VOICE_WEIGHT
        self.confidence_threshold: float = settings.DEFAULT_CONFIDENCE_THRESHOLD

        self.enable_gaze: bool = True
        self.enable_gesture: bool = True
        self.enable_voice: bool = True

        self._buffer: Deque[FusionEvent] = deque(maxlen=120)
        self._last_gesture_trigger_time: float = 0.0
        self._gesture_cooldown_s: float = 1.1

    def update_weights_and_settings(
        self,
        w_gaze: float,
        w_gesture: float,
        w_voice: float,
        threshold: float,
        enable_gaze: bool = True,
        enable_gesture: bool = True,
        enable_voice: bool = True,
    ) -> None:
        with self._lock:
            if any(w < 0.0 for w in (w_gaze, w_gesture, w_voice)):
                raise ValueError("Modality weights must be non-negative.")

            total = w_gaze + w_gesture + w_voice
            if total <= 0.0:
                raise ValueError("At least one modality weight must be > 0.")

            # Normalize weights so they always sum to 1.0
            self.w_gaze = round(w_gaze / total, 4)
            self.w_gesture = round(w_gesture / total, 4)
            self.w_voice = round(w_voice / total, 4)
            self.confidence_threshold = max(0.1, min(1.0, float(threshold)))

            self.enable_gaze = bool(enable_gaze)
            self.enable_gesture = bool(enable_gesture)
            self.enable_voice = bool(enable_voice)

    def _insert_ordered(self, event: FusionEvent) -> None:
        """Inserts event in true chronological order by timestamp."""
        if not self._buffer or event.timestamp >= self._buffer[-1].timestamp:
            self._buffer.append(event)
            return
        timestamps = [e.timestamp for e in self._buffer]
        idx = bisect_right(timestamps, event.timestamp)
        if len(self._buffer) == self._buffer.maxlen:
            self._buffer.popleft()
            idx = max(0, idx - 1)
        self._buffer.insert(idx, event)

    def _prune_old_events(self, now: Optional[float] = None) -> None:
        # Fix falsy-zero timestamp bug: explicitly check `now is None`
        current = time.time() if now is None else float(now)
        cutoff = current - self.window_seconds
        while self._buffer and self._buffer[0].timestamp < cutoff:
            self._buffer.popleft()

    def ingest_gaze(self, gaze: GazeState) -> None:
        with self._lock:
            if not self.enable_gaze or gaze.confidence <= 0.0:
                return
            context_engine.update_spatial_context(gaze)
            self._prune_old_events(gaze.timestamp)
            self._insert_ordered(FusionEvent("GAZE", gaze.timestamp, gaze, gaze.confidence))

    def ingest_gesture(self, gesture: GestureState) -> None:
        with self._lock:
            if not self.enable_gesture or gesture.gesture == "NONE" or gesture.confidence <= 0.0:
                return
            self._prune_old_events(gesture.timestamp)
            self._insert_ordered(FusionEvent("GESTURE", gesture.timestamp, gesture, gesture.confidence))

    def ingest_voice(self, voice: VoiceState) -> None:
        with self._lock:
            if not self.enable_voice or not voice.text.strip() or voice.confidence <= 0.0:
                return
            self._prune_old_events(voice.timestamp)
            self._insert_ordered(FusionEvent("VOICE", voice.timestamp, voice, voice.confidence))

    def compute_weighted_score(
        self, gaze_conf: float, gesture_conf: float, voice_conf: float
    ) -> Tuple[float, List[str]]:
        """
        Computes weighted confidence score while strictly excluding disabled modalities
        from both the score math and active_mods list.
        """
        with self._lock:
            g_c = max(0.0, min(1.0, float(gaze_conf))) if self.enable_gaze else 0.0
            h_c = max(0.0, min(1.0, float(gesture_conf))) if self.enable_gesture else 0.0
            v_c = max(0.0, min(1.0, float(voice_conf))) if self.enable_voice else 0.0

            active_mods: List[str] = []
            active_weight_sum = 0.0
            raw_weighted_sum = 0.0

            if g_c > 0.0 and self.w_gaze > 0.0:
                active_mods.append("GAZE")
                active_weight_sum += self.w_gaze
                raw_weighted_sum += self.w_gaze * g_c

            if h_c > 0.0 and self.w_gesture > 0.0:
                active_mods.append("GESTURE")
                active_weight_sum += self.w_gesture
                raw_weighted_sum += self.w_gesture * h_c

            if v_c > 0.0 and self.w_voice > 0.0:
                active_mods.append("VOICE")
                active_weight_sum += self.w_voice
                raw_weighted_sum += self.w_voice * v_c

            if active_weight_sum <= 0.0:
                return 0.0, []

            score = raw_weighted_sum / active_weight_sum
            return round(float(score), 4), active_mods

    def evaluate_voice_trigger(
        self, voice: VoiceState
    ) -> Optional[Tuple[StructuredIntent, float, List[str], Dict]]:
        with self._lock:
            if not self.enable_voice or not voice.text.strip() or voice.confidence <= 0.0:
                return None

            self.ingest_voice(voice)
            best_gaze = self._get_best_recent_gaze(now=voice.timestamp)
            best_gesture = self._get_best_recent_gesture(now=voice.timestamp)

            gaze_conf = best_gaze.confidence if (best_gaze and self.enable_gaze) else 0.0
            gaze_target = best_gaze.target if (best_gaze and self.enable_gaze) else None
            gesture_conf = best_gesture.confidence if (best_gesture and self.enable_gesture) else 0.0
            gesture_name = best_gesture.gesture if (best_gesture and self.enable_gesture) else None

            raw_intent = intent_parser.parse(
                voice.text, active_app=context_engine.active_application
            )
            if raw_intent.intent in ("NONE", "UNKNOWN"):
                return None

            # Detect conflicting modality signals (e.g., Voice says "Open Chrome" while Gesture is CANCEL/STOP)
            if gesture_name in ("THUMBS_DOWN", "PALM") and raw_intent.intent not in ("CANCEL", "STOP"):
                logger.warning(
                    "Conflicting intent detected: Voice='%s' vs Gesture='%s'. Blocking execution.",
                    raw_intent.intent,
                    gesture_name,
                )
                return None

            resolved_intent = context_engine.resolve_intent(
                raw_intent, window_gaze_target=gaze_target, gesture_name=gesture_name
            )

            # Flag ambiguous unresolved targets for risk-aware confirmation
            ambiguous_target = resolved_intent.target in ("IT_REF", "THIS_REF", None) and raw_intent.target in (
                "IT_REF",
                "THIS_REF",
            )

            final_score, active_mods = self.compute_weighted_score(
                gaze_conf, gesture_conf, voice.confidence
            )
            if final_score < self.confidence_threshold:
                return None

            telemetry_meta = {
                "gaze_target": gaze_target,
                "gaze_conf": gaze_conf,
                "gesture": gesture_name,
                "gesture_conf": gesture_conf,
                "voice_text": voice.text,
                "voice_conf": voice.confidence,
                "ambiguous_target": ambiguous_target,
            }
            return resolved_intent, final_score, active_mods, telemetry_meta

    def evaluate_gesture_trigger(
        self, gesture: GestureState
    ) -> Optional[Tuple[StructuredIntent, float, List[str], Dict]]:
        with self._lock:
            if not self.enable_gesture:
                return None
            self.ingest_gesture(gesture)
            if gesture.gesture in ("NONE", "POINT", "TWO_FINGERS") or gesture.confidence <= 0.0:
                return None

            now = gesture.timestamp if gesture.timestamp is not None else time.time()
            if (now - self._last_gesture_trigger_time) < self._gesture_cooldown_s:
                return None

            best_gaze = self._get_best_recent_gaze(now=now)
            gaze_conf = best_gaze.confidence if (best_gaze and self.enable_gaze) else 0.0
            gaze_target = best_gaze.target if (best_gaze and self.enable_gaze) else None

            intent = context_engine.resolve_gesture_intent(
                gesture.gesture, gesture.mapped_action, gaze_target
            )
            if intent is None:
                return None

            final_score, active_mods = self.compute_weighted_score(
                gaze_conf, gesture.confidence, 0.0
            )
            if final_score < self.confidence_threshold:
                return None

            self._last_gesture_trigger_time = now
            telemetry_meta = {
                "gaze_target": gaze_target,
                "gaze_conf": gaze_conf,
                "gesture": gesture.gesture,
                "gesture_conf": gesture.confidence,
                "voice_text": None,
                "voice_conf": 0.0,
                "ambiguous_target": False,
            }
            return intent, final_score, active_mods, telemetry_meta

    def evaluate_dwell_trigger(
        self, gaze: GazeState
    ) -> Optional[Tuple[StructuredIntent, float, List[str], Dict]]:
        with self._lock:
            if not self.enable_gaze:
                return None
            self.ingest_gaze(gaze)
            if not gaze.target or gaze.confidence < self.confidence_threshold:
                return None

            intent = StructuredIntent(
                intent="SELECT", target=gaze.target, raw_source="GAZE_DWELL"
            )
            final_score, active_mods = self.compute_weighted_score(gaze.confidence, 0.0, 0.0)
            if final_score < self.confidence_threshold:
                return None

            telemetry_meta = {
                "gaze_target": gaze.target,
                "gaze_conf": gaze.confidence,
                "gesture": "DWELL_HOLD",
                "gesture_conf": 0.0,
                "voice_text": None,
                "voice_conf": 0.0,
                "ambiguous_target": False,
            }
            return intent, final_score, active_mods, telemetry_meta

    def _get_best_recent_gaze(self, now: Optional[float] = None) -> Optional[GazeState]:
        self._prune_old_events(now)
        if not self.enable_gaze:
            return None
        for event in reversed(self._buffer):
            if event.modality == "GAZE" and getattr(event.state, "target", None):
                return event.state
        return None

    def _get_best_recent_gesture(self, now: Optional[float] = None) -> Optional[GestureState]:
        self._prune_old_events(now)
        if not self.enable_gesture:
            return None
        for event in reversed(self._buffer):
            if event.modality == "GESTURE" and getattr(event.state, "gesture", "NONE") != "NONE":
                return event.state
        return None


fusion_engine = MultimodalFusionEngine()