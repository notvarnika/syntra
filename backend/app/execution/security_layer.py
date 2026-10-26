import logging
import threading
import time
from typing import Optional, Tuple
from app.schemas import ExecutionStatus, RiskLevel, SecurityPrompt, StructuredIntent

logger = logging.getLogger(__name__)


class SecurityLayer:
    """
    Thread-safe Safety & Risk-Tiered Confirmation Controller.
    Blocks or requires explicit confirmation for destructive and ambiguous actions.
    """

    HIGH_RISK_INTENTS = {"DELETE_FILE", "SYSTEM_SHUTDOWN", "FORMAT_DISK"}
    MEDIUM_RISK_INTENTS = {"CLOSE_APPLICATION"}

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._pending_intent: Optional[StructuredIntent] = None
        self._pending_meta: Optional[dict] = None
        self._prompt_expires_at: float = 0.0
        self.confirmation_timeout_s: float = 10.0

    def classify_risk(self, intent: StructuredIntent, meta: Optional[dict] = None) -> RiskLevel:
        action = intent.intent.upper()
        if action in self.HIGH_RISK_INTENTS:
            return RiskLevel.HIGH
        # Elevate ambiguous targets on state-modifying actions to HIGH risk
        if meta and meta.get("ambiguous_target") and action in ("CLOSE_APPLICATION", "OPEN_APPLICATION"):
            return RiskLevel.HIGH
        if action in self.MEDIUM_RISK_INTENTS:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def evaluate(
        self, intent: StructuredIntent, fusion_score: float, meta: Optional[dict] = None
    ) -> Tuple[ExecutionStatus, RiskLevel, Optional[StructuredIntent], str]:
        with self._lock:
            now = time.time()
            action = intent.intent.upper()

            if self._pending_intent and now > self._prompt_expires_at:
                self._pending_intent = None
                self._pending_meta = None

            # 1. Process CONFIRM / CANCEL when a prompt is active
            if self._pending_intent is not None:
                if action == "CONFIRM":
                    approved = self._pending_intent
                    self._pending_intent = None
                    self._pending_meta = None
                    return (
                        ExecutionStatus.SUCCESS,
                        RiskLevel.HIGH,
                        approved,
                        f"High-risk action '{approved.intent}' confirmed by user.",
                    )
                if action in ("CANCEL", "STOP"):
                    cancelled = self._pending_intent
                    self._pending_intent = None
                    self._pending_meta = None
                    return (
                        ExecutionStatus.CANCELLED,
                        RiskLevel.HIGH,
                        cancelled,
                        f"High-risk action '{cancelled.intent}' cancelled by user.",
                    )

            if action in ("CONFIRM", "CANCEL"):
                return (
                    ExecutionStatus.BLOCKED,
                    RiskLevel.LOW,
                    None,
                    "No pending security prompt to confirm or cancel.",
                )

            # Block destructive operations if target is completely unresolved/ambiguous
            if action in self.HIGH_RISK_INTENTS and (
                not intent.target or intent.target in ("IT_REF", "THIS_REF", "None")
            ):
                return (
                    ExecutionStatus.BLOCKED,
                    RiskLevel.HIGH,
                    intent,
                    f"Blocked destructive action '{action}': exact target path or window is unresolved.",
                )

            risk = self.classify_risk(intent, meta)

            # 2. HIGH RISK or Ambiguous Action -> Require Explicit Confirmation
            if risk == RiskLevel.HIGH:
                self._pending_intent = intent
                self._pending_meta = meta
                self._prompt_expires_at = now + self.confirmation_timeout_s
                target_label = intent.target or "ambiguous target"
                msg = (
                    f"Confirmation required for {intent.intent} ({target_label}). "
                    "Show THUMBS UP or say 'Confirm' to proceed, or THUMBS DOWN / 'Cancel' to abort."
                )
                logger.warning(msg)
                return (
                    ExecutionStatus.AWAITING_CONFIRMATION,
                    RiskLevel.HIGH,
                    intent,
                    msg,
                )

            # 3. MEDIUM RISK -> Elevated confidence threshold
            if risk == RiskLevel.MEDIUM and fusion_score < 0.75:
                return (
                    ExecutionStatus.BLOCKED,
                    RiskLevel.MEDIUM,
                    intent,
                    f"Medium-risk action '{intent.intent}' blocked (score {fusion_score:.2f} < 0.75).",
                )

            return (
                ExecutionStatus.SUCCESS,
                risk,
                intent,
                f"Permission Allowed ({risk.value} Risk)",
            )

    def get_active_prompt(self) -> SecurityPrompt:
        with self._lock:
            now = time.time()
            if self._pending_intent is None or now > self._prompt_expires_at:
                self._pending_intent = None
                return SecurityPrompt(active=False)

            target_str = self._pending_intent.target or "Selected Resource"
            return SecurityPrompt(
                active=True,
                intent_summary=f"{self._pending_intent.intent} ({target_str})",
                risk_level=RiskLevel.HIGH,
                message="Show Thumbs Up or say 'Confirm' to execute, or Thumbs Down / 'Cancel' to abort.",
                expires_at=self._prompt_expires_at,
            )


security_layer = SecurityLayer()