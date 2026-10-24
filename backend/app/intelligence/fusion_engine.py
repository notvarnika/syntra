from collections import deque
import time
from typing import Deque, Dict, List, Optional, Tuple
from app.config import settings
from app.intelligence.context_engine import context_engine
from app.intelligence.intent_parser import intent_parser
from app.schemas import GazeState, GestureState, StructuredIntent, VoiceState


class FusionEvent:
    def __init__(self, modality: str, timestamp: float, state: object, confidence: float) -> None:
        self.modality = modality
        self.timestamp = timestamp
        self.state = state
        self.confidence = confidence


class MultimodalFusionEngine:
    """
    Core component of SYNTRA.
    Maintains a 2-3 second Temporal Interaction Window and fuses:
      Gaze (WHERE) + Gesture (HOW) + Voice (WHAT) + Context
    using weighted confidence scoring:
      Final Score = w_gaze * C_gaze + w_gesture * C_gesture + w_voice * C_voice
    """

    def __init__(self) -> None:
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
        self.w_gaze = w_gaze
        self.w_gesture = w_gesture
        self.w_voice = w_voice
        self.confidence_threshold = threshold
        self.enable_gaze = enable_gaze
        self.enable_gesture = enable_gesture
        self.enable_voice = enable_voice

    def ingest_gaze(self, gaze: GazeState) -> None:
        if not self.enable_gaze or gaze.confidence <= 0.0:
            return
        context_engine.update_spatial_context(gaze)
        self._prune_old_events(gaze.timestamp)
        self._buffer.append(FusionEvent("GAZE", gaze.timestamp, gaze, gaze.confidence))

    def ingest_gesture(self, gesture: GestureState) -> None:
        if not self.enable_gesture or gesture.gesture == "NONE" or gesture.confidence <= 0.0:
            return
        self._prune_old_events(gesture.timestamp)
        self._buffer.append(
            FusionEvent("GESTURE", gesture.timestamp, gesture, gesture.confidence)
        )

    def ingest_voice(self, voice: VoiceState) -> None:
        if not self.enable_voice or not voice.text or voice.confidence <= 0.0:
            return
        self._prune_old_events(voice.timestamp)
        self._buffer.append(FusionEvent("VOICE", voice.timestamp, voice, voice.confidence))

    def _prune_old_events(self, now: Optional[float] = None) -> None:
        current = now or time.time()
        cutoff = current - self.window_seconds
        while self._buffer and self._buffer[0].timestamp < cutoff:
            self._buffer.popleft()

    def compute_weighted_score(
        self, gaze_conf: float, gesture_conf: float, voice_conf: float
    ) -> Tuple[float, List[str]]:
        """
        Computes the exact SYNTRA weighted confidence score:
          Final Score = 0.40 * Gaze + 0.30 * Gesture + 0.30 * Voice
        When all 3 modalities are present, applies the direct formula.
        When operating in Unimodal/Bimodal mode (e.g., Demo 1, Demo 2, Demo 3),
        normalizes active weights so single-modality accessibility modes also pass threshold.
        """
        active_mods: List[str] = []
        raw_weighted_sum = (
            (self.w_gaze * gaze_conf)
            + (self.w_gesture * gesture_conf)
            + (self.w_voice * voice_conf)
        )

        active_weight_sum = 0.0
        if gaze_conf > 0:
            active_mods.append("GAZE")
            active_weight_sum += self.w_gaze
        if gesture_conf > 0:
            active_mods.append("GESTURE")
            active_weight_sum += self.w_gesture
        if voice_conf > 0:
            active_mods.append("VOICE")
            active_weight_sum += self.w_voice

        if active_weight_sum <= 0.0:
            return 0.0, []

        # Full 3-modality interaction uses exact synopsis formula
        if len(active_mods) == 3:
            total_w = self.w_gaze + self.w_gesture + self.w_voice
            score = raw_weighted_sum / total_w if total_w > 0 else raw_weighted_sum
        else:
            # Normalize by active modality weights for Demo 1 / Demo 2 / Demo 3
            score = raw_weighted_sum / active_weight_sum

        return round(float(score), 4), active_mods

    def evaluate_voice_trigger(
        self, voice: VoiceState
    ) -> Optional[Tuple[StructuredIntent, float, List[str], Dict]]:
        """
        Triggered immediately when Faster-Whisper transcribes a spoken command.
        Looks back 2.5 seconds in the Temporal Ring Buffer for concurrent Gaze and Gesture signals.
        Covers Demo 2 (Voice Only), Demo 4 (Gaze + Pinch + 'Open it'), Demo 5 (Context), and Demo 6 (Security).
        """
        self.ingest_voice(voice)
        best_gaze = self._get_best_recent_gaze()
        best_gesture = self._get_best_recent_gesture()

        gaze_conf = best_gaze.confidence if best_gaze else 0.0
        gaze_target = best_gaze.target if best_gaze else None
        gesture_conf = best_gesture.confidence if best_gesture else 0.0
        gesture_name = best_gesture.gesture if best_gesture else None

        # Parse natural language into raw StructuredIntent
        raw_intent = intent_parser.parse(
            voice.text, active_app=context_engine.active_application
        )
        if raw_intent.intent in ("NONE", "UNKNOWN"):
            return None

        # Resolve pronouns ("it", "this", "first result") via Context Engine + Temporal Gaze Target
        resolved_intent = context_engine.resolve_intent(
            raw_intent, window_gaze_target=gaze_target
        )

        # If user pinched AND said "Open it" while looking at an app (Demo 4), ensure OPEN_APPLICATION
        if (
            gesture_name == "PINCH"
            and resolved_intent.intent in ("CLICK", "SELECT")
            and resolved_intent.target in ("Chrome", "YouTube", "VS Code", "Spotify")
        ):
            resolved_intent.intent = "OPEN_APPLICATION"

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
        }
        return resolved_intent, final_score, active_mods, telemetry_meta

    def evaluate_gesture_trigger(
        self, gesture: GestureState
    ) -> Optional[Tuple[StructuredIntent, float, List[str], Dict]]:
        """
        Evaluates discrete actionable gestures (PINCH, PALM, THUMBS_UP, THUMBS_DOWN)
        fused with Gaze Target within the temporal window (Demo 1: Point/Gaze + Pinch -> Open Chrome).
        """
        self.ingest_gesture(gesture)
        if gesture.gesture in ("NONE", "POINT", "TWO_FINGERS"):
            return None

        now = time.time()
        if (now - self._last_gesture_trigger_time) < self._gesture_cooldown_s:
            return None

        best_gaze = self._get_best_recent_gaze()
        gaze_conf = best_gaze.confidence if best_gaze else 0.0
        gaze_target = best_gaze.target if best_gaze else None

        # Map gesture to StructuredIntent
        if gesture.gesture == "THUMBS_UP":
            intent = StructuredIntent(intent="CONFIRM", raw_source="GESTURE")
        elif gesture.gesture == "THUMBS_DOWN":
            intent = StructuredIntent(intent="CANCEL", raw_source="GESTURE")
        elif gesture.gesture == "PALM":
            intent = StructuredIntent(
                intent=gesture.mapped_action if gesture.mapped_action != "NONE" else "STOP",
                raw_source="GESTURE",
            )
        elif gesture.gesture == "PINCH":
            # If looking at or pointing at an application dock target, open/select it
            if gaze_target in ("Chrome", "YouTube", "VS Code", "Spotify"):
                intent = StructuredIntent(
                    intent="OPEN_APPLICATION", target=gaze_target, raw_source="FUSION"
                )
            else:
                intent = StructuredIntent(
                    intent=gesture.mapped_action if gesture.mapped_action != "NONE" else "CLICK",
                    target=gaze_target,
                    raw_source="GESTURE",
                )
        elif gesture.gesture == "FIST":
            intent = StructuredIntent(intent="GRAB", target=gaze_target, raw_source="GESTURE")
        else:
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
        }
        return intent, final_score, active_mods, telemetry_meta

    def evaluate_dwell_trigger(
        self, gaze: GazeState
    ) -> Optional[Tuple[StructuredIntent, float, List[str], Dict]]:
        """
        Demo 3 Support: Gaze-Only Dwell Selection when user holds gaze on a target.
        """
        self.ingest_gaze(gaze)
        if not gaze.target or gaze.confidence < self.confidence_threshold:
            return None

        intent = StructuredIntent(
            intent="SELECT", target=gaze.target, raw_source="GAZE_DWELL"
        )
        final_score, active_mods = self.compute_weighted_score(gaze.confidence, 0.0, 0.0)
        telemetry_meta = {
            "gaze_target": gaze.target,
            "gaze_conf": gaze.confidence,
            "gesture": "DWELL_HOLD",
            "gesture_conf": 0.0,
            "voice_text": None,
            "voice_conf": 0.0,
        }
        return intent, final_score, active_mods, telemetry_meta

    def _get_best_recent_gaze(self) -> Optional[GazeState]:
        self._prune_old_events()
        gaze_events = [
            e.state for e in reversed(self._buffer) if e.modality == "GAZE" and e.state.target
        ]
        return gaze_events[0] if gaze_events else None

    def _get_best_recent_gesture(self) -> Optional[GestureState]:
        self._prune_old_events()
        gesture_events = [
            e.state
            for e in reversed(self._buffer)
            if e.modality == "GESTURE" and e.state.gesture != "NONE"
        ]
        return gesture_events[0] if gesture_events else None


fusion_engine = MultimodalFusionEngine()