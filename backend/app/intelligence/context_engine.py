from collections import deque
import threading
from typing import Deque, List, Optional
from app.schemas import ContextSnapshot, GazeState, StructuredIntent


class ContextEngine:
    """
    Maintains spatial (gaze) and conversational short-term memory.
    Resolves deictic references ('it', 'this', 'that', 'first result') and promotes
    gestural selection on known application targets into OPEN_APPLICATION intents.
    """

    KNOWN_APPLICATIONS = {"chrome", "youtube", "vs code", "vscode", "spotify", "notepad", "calculator", "terminal"}
    CANONICAL_APP_NAMES = {
        "chrome": "Chrome",
        "youtube": "YouTube",
        "vs code": "VS Code",
        "vscode": "VS Code",
        "spotify": "Spotify",
        "notepad": "Notepad",
        "calculator": "Calculator",
        "terminal": "Terminal",
    }

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.active_application: str = "Desktop"
        self._recent_targets: Deque[str] = deque(maxlen=8)
        self._last_intents: Deque[str] = deque(maxlen=8)

    @classmethod
    def is_launchable_app(cls, target: Optional[str]) -> bool:
        if not target:
            return False
        return target.strip().lower() in cls.KNOWN_APPLICATIONS

    @classmethod
    def normalize_app_name(cls, target: str) -> str:
        key = target.strip().lower()
        return cls.CANONICAL_APP_NAMES.get(key, target)

    def update_spatial_context(self, gaze: GazeState) -> None:
        with self._lock:
            if gaze.target and gaze.target != "None":
                if not self._recent_targets or self._recent_targets[0] != gaze.target:
                    self._recent_targets.appendleft(gaze.target)

    def record_executed_intent(self, intent: StructuredIntent) -> None:
        with self._lock:
            summary = f"{intent.intent}({intent.target or ''})"
            self._last_intents.appendleft(summary)
            if intent.intent == "OPEN_APPLICATION" and intent.target:
                self.active_application = self.normalize_app_name(intent.target)

    def resolve_intent(
        self,
        raw_intent: StructuredIntent,
        window_gaze_target: Optional[str] = None,
        gesture_name: Optional[str] = None,
    ) -> StructuredIntent:
        with self._lock:
            resolved = raw_intent.model_copy(deep=True)
            fallback_target = window_gaze_target or (
                self._recent_targets[0] if self._recent_targets else None
            )

            # Bind deictic pronouns ("Open it", "Delete this file") to gaze/context target
            if resolved.target in ("IT_REF", "THIS_REF", None) and fallback_target:
                if resolved.intent in ("OPEN_APPLICATION", "SELECT", "CLICK", "DELETE_FILE", "CLOSE_APPLICATION"):
                    resolved.target = fallback_target

            # Promote CLICK/SELECT + PINCH on a known application target to OPEN_APPLICATION
            if (
                gesture_name == "PINCH"
                and resolved.intent in ("CLICK", "SELECT")
                and self.is_launchable_app(resolved.target)
            ):
                resolved.intent = "OPEN_APPLICATION"
                resolved.target = self.normalize_app_name(resolved.target)
            elif resolved.intent == "OPEN_APPLICATION" and resolved.target:
                resolved.target = self.normalize_app_name(resolved.target)

            return resolved

    def resolve_gesture_intent(
        self, gesture: str, mapped_action: str, gaze_target: Optional[str]
    ) -> Optional[StructuredIntent]:
        """Maps discrete hand gestures and spatial context to a StructuredIntent."""
        if gesture == "THUMBS_UP":
            return StructuredIntent(intent="CONFIRM", raw_source="GESTURE")
        if gesture == "THUMBS_DOWN":
            return StructuredIntent(intent="CANCEL", raw_source="GESTURE")
        if gesture == "PALM":
            action = mapped_action if mapped_action and mapped_action != "NONE" else "STOP"
            return StructuredIntent(intent=action, raw_source="GESTURE")
        if gesture == "PINCH":
            if self.is_launchable_app(gaze_target):
                return StructuredIntent(
                    intent="OPEN_APPLICATION",
                    target=self.normalize_app_name(gaze_target),
                    raw_source="FUSION",
                )
            action = mapped_action if mapped_action and mapped_action != "NONE" else "CLICK"
            return StructuredIntent(intent=action, target=gaze_target, raw_source="GESTURE")
        if gesture == "FIST":
            return StructuredIntent(intent="GRAB", target=gaze_target, raw_source="GESTURE")
        return None

    def get_snapshot(self) -> ContextSnapshot:
        with self._lock:
            return ContextSnapshot(
                active_application=self.active_application,
                recent_targets=list(self._recent_targets),
                last_intents=list(self._last_intents),
            )


context_engine = ContextEngine()