from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ModalityType(str, Enum):
    GAZE = "GAZE"
    GESTURE = "GESTURE"
    VOICE = "VOICE"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ExecutionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


# --- Real-Time Perception Outputs ---
class GazeState(BaseModel):
    x: int = 0
    y: int = 0
    target: Optional[str] = None
    confidence: float = 0.0
    dwell_progress: float = 0.0  # 0.0 to 1.0 for Demo 3 Gaze Dwell Selection
    timestamp: float = 0.0


class GestureState(BaseModel):
    gesture: str = "NONE"  # NONE, POINT, PINCH, SCROLL, PALM, FIST, THUMBS_UP, THUMBS_DOWN
    mapped_action: str = "NONE"
    cursor_x: int = 0
    cursor_y: int = 0
    confidence: float = 0.0
    timestamp: float = 0.0


class VoiceState(BaseModel):
    text: str = ""
    is_listening: bool = False
    is_processing: bool = False
    confidence: float = 0.0
    timestamp: float = 0.0


# --- Intent, Fusion & Security ---
class StructuredIntent(BaseModel):
    intent: str = "NONE"          # e.g., OPEN_APPLICATION, WEB_SEARCH, CLICK, SCROLL, DELETE_FILE
    target: Optional[str] = None  # e.g., Chrome, YouTube, IT_REF, FIRST_RESULT
    application: Optional[str] = None
    query: Optional[str] = None
    direction: Optional[str] = None
    amount: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    raw_source: str = "RULE"      # RULE or LLM


class SecurityPrompt(BaseModel):
    active: bool = False
    intent_summary: str = ""
    risk_level: RiskLevel = RiskLevel.LOW
    message: str = ""
    expires_at: float = 0.0


class PrivacyStatus(BaseModel):
    camera_on: bool = False
    microphone_on: bool = False
    voice_processing_active: bool = False


class ContextSnapshot(BaseModel):
    active_application: str = "Desktop"
    last_target: Optional[str] = None
    last_intent: Optional[str] = None
    last_query: Optional[str] = None
    recent_targets: List[str] = Field(default_factory=list)


# --- Live WebSocket Broadcast Packet (Sent at 15Hz to React Dashboard) ---
class TelemetryFrame(BaseModel):
    privacy: PrivacyStatus
    gaze: GazeState
    gesture: GestureState
    voice: VoiceState
    fusion_score: float = 0.0
    active_modalities: List[str] = Field(default_factory=list)
    resolved_intent: Optional[StructuredIntent] = None
    context: ContextSnapshot
    security_prompt: SecurityPrompt
    camera_Preview_b64: Optional[str] = None


# --- REST API Payloads for User Personalization & Calibration ---
class GestureMappingSchema(BaseModel):
    gesture_name: str
    mapped_action: str

    class Config:
        from_attributes = True


class UserProfileUpdate(BaseModel):
    active_profile_name: Optional[str] = None
    dwell_time_ms: Optional[int] = None
    gaze_weight: Optional[float] = None
    gesture_weight: Optional[float] = None
    voice_weight: Optional[float] = None
    confidence_threshold: Optional[float] = None
    enable_gaze: Optional[bool] = None
    enable_gesture: Optional[bool] = None
    enable_voice: Optional[bool] = None
    gesture_mappings: Optional[List[GestureMappingSchema]] = None


class UserProfileResponse(BaseModel):
    id: int
    profile_name: str
    is_active: bool
    dwell_time_ms: int
    gaze_weight: float
    gesture_weight: float
    voice_weight: float
    confidence_threshold: float
    enable_gaze: bool
    enable_gesture: bool
    enable_voice: bool
    gesture_mappings: List[GestureMappingSchema]

    class Config:
        from_attributes = True


class CalibrationPointInput(BaseModel):
    screen_x: int
    screen_y: int
    feature_x: float
    feature_y: float


class CalibrationRequest(BaseModel):
    points: List[CalibrationPointInput]


class ActionLogResponse(BaseModel):
    id: int
    timestamp: datetime
    gaze_target: Optional[str]
    gesture_detected: Optional[str]
    voice_command: Optional[str]
    resolved_intent: str
    target_entity: Optional[str]
    fusion_score: float
    risk_level: str
    status: str
    latency_ms: float

    class Config:
        from_attributes = True