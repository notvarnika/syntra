from collections import deque
from typing import Deque, Optional
from app.perception.screen_inspector import screen_inspector
from app.schemas import ContextSnapshot, GazeState, StructuredIntent


class ContextEngine:
    """
    Maintains conversational and spatial state history across interactions.
    Resolves ambiguous references ('it', 'this', 'that', 'first result', 'previous result')
    using recent Gaze Targets, active OS windows, and prior executed commands.
    """

    def __init__(self) -> None:
        self.active_application: str = "Desktop"
        self.last_gaze_target: Optional[str] = None
        self.last_executed_intent: Optional[str] = None
        self.last_search_query: Optional[str] = None
        self.recent_targets: Deque[str] = deque(maxlen=6)

    def update_spatial_context(self, gaze: GazeState) -> None:
        """Updates the context memory with the latest high-confidence gaze target and OS window."""
        os_win = screen_inspector.get_active_window_title()
        if os_win and os_win != "Desktop" and not os_win.startswith("c \"import"):
            self.active_application = os_win

        if gaze.target and gaze.confidence >= 0.60:
            clean_target = gaze.target
            if clean_target.startswith("c \"import"):
                clean_target = "Terminal"
            self.last_gaze_target = clean_target
            if not self.recent_targets or self.recent_targets[0] != clean_target:
                self.recent_targets.appendleft(clean_target)

    def resolve_intent(
        self, intent: StructuredIntent, window_gaze_target: Optional[str] = None
    ) -> StructuredIntent:
        """
        Resolves deictic references (IT_REF, FIRST_RESULT, missing application context)
        by combining the current temporal window's Gaze Target with historical context.
        """
        resolved = intent.model_copy(deep=True)

        # Best available spatial target: temporal window gaze target -> recent gaze target -> active app
        spatial_target = window_gaze_target or self.last_gaze_target or self.active_application
        if spatial_target and spatial_target.startswith("c \"import"):
            spatial_target = "Chrome"

        # 1. Resolve Pronoun Reference ("Open it", "Play this", "Close that", "Delete this file")
        if resolved.target in ("IT_REF", None, ""):
            if resolved.intent in (
                "OPEN_APPLICATION",
                "CLICK",
                "SELECT",
                "PLAY",
                "CLOSE_APPLICATION",
                "DELETE_FILE",
            ):
                resolved.target = spatial_target

        # 2. Demo 5 Support: "Search for Python" while Chrome is open -> bind application = Chrome
        if resolved.intent == "WEB_SEARCH" and not resolved.application:
            resolved.application = (
                self.active_application
                if self.active_application in ("Chrome", "YouTube")
                else "Chrome"
            )

        # 3. Demo 5 Support: "Open the first result" after a web search
        if resolved.target == "FIRST_RESULT":
            resolved.application = self.active_application or "Chrome"

        return resolved

    def record_executed_intent(self, intent: StructuredIntent) -> None:
        """Updates conversation state after an intent is successfully executed."""
        self.last_executed_intent = intent.intent
        if intent.intent == "OPEN_APPLICATION" and intent.target:
            self.active_application = intent.target.title()
            if not self.recent_targets or self.recent_targets[0] != self.active_application:
                self.recent_targets.appendleft(self.active_application)
        elif intent.intent == "WEB_SEARCH":
            self.active_application = "Chrome"
            self.last_search_query = intent.query or intent.target

    def get_snapshot(self) -> ContextSnapshot:
        return ContextSnapshot(
            active_application=self.active_application,
            last_target=self.last_gaze_target,
            last_intent=self.last_executed_intent,
            last_query=self.last_search_query,
            recent_targets=list(self.recent_targets),
        )


context_engine = ContextEngine()