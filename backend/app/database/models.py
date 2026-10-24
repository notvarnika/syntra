from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from app.database.db import Base


class UserProfile(Base):
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, index=True)
    profile_name = Column(String(64), unique=True, index=True, nullable=False)
    is_active = Column(Boolean, default=False)

    # Personalization Settings (Synopsis Section: Personalization)
    dwell_time_ms = Column(Integer, default=1200)
    gaze_weight = Column(Float, default=0.40)
    gesture_weight = Column(Float, default=0.30)
    voice_weight = Column(Float, default=0.30)
    confidence_threshold = Column(Float, default=0.70)

    # Accessibility Modality Toggles
    enable_gaze = Column(Boolean, default=True)
    enable_gesture = Column(Boolean, default=True)
    enable_voice = Column(Boolean, default=True)

    gesture_mappings = relationship(
        "GestureMapping", back_populates="profile", cascade="all, delete-orphan"
    )


class GestureMapping(Base):
    __tablename__ = "gesture_mappings"

    id = Column(Integer, primary_key=True, index=True)
    profile_id = Column(Integer, ForeignKey("user_profiles.id"), nullable=False)
    gesture_name = Column(String(32), nullable=False)   # e.g., PINCH, THUMBS_UP, PALM
    mapped_action = Column(String(64), nullable=False)  # e.g., CLICK, CONFIRM, STOP

    profile = relationship("UserProfile", back_populates="gesture_mappings")


class ActionLog(Base):
    __tablename__ = "action_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    # Multimodal Inputs
    gaze_target = Column(String(128), nullable=True)
    gaze_confidence = Column(Float, default=0.0)
    gesture_detected = Column(String(64), nullable=True)
    gesture_confidence = Column(Float, default=0.0)
    voice_command = Column(Text, nullable=True)
    voice_confidence = Column(Float, default=0.0)

    # Fusion & Execution Output
    resolved_intent = Column(String(64), nullable=False)
    target_entity = Column(String(256), nullable=True)
    fusion_score = Column(Float, default=0.0)
    risk_level = Column(String(16), default="LOW")
    status = Column(String(32), default="SUCCESS")  # SUCCESS, BLOCKED, CANCELLED
    latency_ms = Column(Float, default=0.0)


class BenchmarkRecord(Base):
    __tablename__ = "benchmark_records"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    mode = Column(String(32), nullable=False)  # GAZE_ONLY, GESTURE_ONLY, VOICE_ONLY, MULTIMODAL
    expected_intent = Column(String(64), nullable=False)
    actual_intent = Column(String(64), nullable=False)
    is_correct = Column(Boolean, default=True)
    false_activation = Column(Boolean, default=False)
    latency_ms = Column(Float, default=0.0)