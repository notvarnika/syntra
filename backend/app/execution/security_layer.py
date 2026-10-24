import time
from typing import Optional, Tuple
from app.schemas import ExecutionStatus, RiskLevel, SecurityPrompt, StructuredIntent


class SecurityLayer:
    """
    Safety-Aware Automation Controller (Synopsis Module 7 & Demo 6).
    Classifies every StructuredIntent into LOW, MEDIUM, or HIGH risk.
    High-risk destructive actions (e.g., DELETE_FILE, SYSTEM_SHUTDOWN) are held in
    an AWAITING_CONFIRMATION state until the user explicitly confirms via
    Thumbs Up gesture or vocal 'Confirm' within 10 seconds.
    """

    HIGH_RISK_INTENTS = {"DELETE_FILE", "SYSTEM_SHUTDOWN", "FORMAT_DISK"}
    MEDIUM_RISK_INTENTS = {"CLOSE_APPLICATION"}

    def __init__(self) -> None:
        self._pending_intent: Optional[StructuredIntent] = None
        self._pending_meta: Optional[dict] = None
        self._prompt_expires_at: float = 0.0
        self.confirmation_timeout_s: float = 10.0

    def classify_risk(self, intent: StructuredIntent) -> RiskLevel:
        action = intent.intent.upper()
        if action in self.HIGH_RISK_INTENTS:
            return RiskLevel.HIGH
        if action in self.MEDIUM_RISK_INTENTS:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def evaluate(
        self, intent: StructuredIntent, fusion_score: float, meta: Optional[dict] = None
    ) -> Tuple[ExecutionStatus, RiskLevel, Optional[StructuredIntent], str]:
        """
        Inspects an incoming intent:
        - If a high-risk prompt is currently awaiting confirmation, checks for CONFIRM / CANCEL.
        - Otherwise, evaluates risk tier and either allows immediate execution or triggers confirmation.
        """
        now = time.time()
        action = intent.intent.upper()

        # Expire stale confirmation prompts
        if self._pending_intent and now > self._prompt_expires_at:
            self._pending_intent = None
            self._pending_meta = None

        # 1. Handle Confirmation / Cancellation for a Pending High-Risk Action (Demo 6)
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

        # Ignore stray CONFIRM/CANCEL when no prompt is active
        if action in ("CONFIRM", "CANCEL"):
            return (
                ExecutionStatus.BLOCKED,
                RiskLevel.LOW,
                None,
                "No pending security prompt to confirm or cancel.",
            )

        risk = self.classify_risk(intent)

        # 2. HIGH RISK -> Intercept and request explicit user confirmation (Demo 6)
        if risk == RiskLevel.HIGH:
            self._pending_intent = intent
            self._pending_meta = meta
            self._prompt_expires_at = now + self.confirmation_timeout_s
            target_label = intent.target or "selected item"
            msg = (
                f"Dangerous action detected: {intent.intent} ({target_label}). "
                "Show THUMBS UP or say 'Confirm' to proceed, or THUMBS DOWN / 'Cancel' to abort."
            )
            print(f"[SYNTRA SecurityLayer] WARNING: {msg}")
            return (
                ExecutionStatus.AWAITING_CONFIRMATION,
                RiskLevel.HIGH,
                intent,
                msg,
            )

        # 3. MEDIUM RISK -> Require higher confidence threshold (>= 0.75)
        if risk == RiskLevel.MEDIUM and fusion_score < 0.75:
            return (
                ExecutionStatus.BLOCKED,
                RiskLevel.MEDIUM,
                intent,
                f"Medium-risk action '{intent.intent}' blocked due to low confidence ({fusion_score:.2f} < 0.75).",
            )

        # 4. LOW RISK -> Allowed immediately
        return (
            ExecutionStatus.SUCCESS,
            risk,
            intent,
            f"Permission Allowed ({risk.value} Risk)",
        )

    def get_active_prompt(self) -> SecurityPrompt:
        now = time.time()
        if self._pending_intent is None or now > self._prompt_expires_at:
            self._pending_intent = None
            return SecurityPrompt(active=False)

        target_str = self._pending_intent.target or "Selected Resource"
        return SecurityPrompt(
            active=True,
            intent_summary=f"{self._pending_intent.intent} ({target_str})",
            risk_level=RiskLevel.HIGH,
            message="Show Thumbs Up 👍 or say 'Confirm' to execute, or Thumbs Down 👎 / 'Cancel' to abort.",
            expires_at=self._prompt_expires_at,
        )


security_layer = SecurityLayer()