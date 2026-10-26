from datetime import datetime, timezone
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from app.database.db import Base


class UserProfile(Base):
    __tablename__ = "user_profiles"
    __table_args__ = (
        CheckConstraint("dwell_time_ms >= 300 AND dwell_time_ms <= 5000", name="ck_profile_dwell_bounds"),
        CheckConstraint("gaze_weight >= 0.0 AND gaze_weight <= 1.0", name="ck_profile_gaze_weight"),
        CheckConstraint("gesture_weight >= 0.0 AND gesture_weight <= 1.0", name="ck_profile_gesture_weight"),
        CheckConstraint("voice_weight >= 0.0 AND voice_weight <= 1.0", name="ck_profile_voice_weight"),
        CheckConstraint(
            "confidence_threshold >= 0.1 AND confidence_threshold <= 1.0",
            name="ck_profile_conf_threshold",
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    profile_name = Column(String(64), unique=True, index=True, nullable=False)
    is_active = Column(Boolean, default=False, nullable=False)

    dwell_time_ms = Column(Integer, default=1200, nullable=False)
    gaze_weight = Column(Float, default=0.40, nullable=False)
    gesture_weight = Column(Float, default=0.30, nullable=False)
    voice_weight = Column(Float, default=0.30, nullable=False)
    confidence_threshold = Column(Float, default=0.70, nullable=False)

    enable_gaze = Column(Boolean, default=True, nullable=False)
    enable_gesture = Column(Boolean, default=True, nullable=False)
    enable_voice = Column(Boolean, default=True, nullable=False)

    gesture_mappings = relationship(
        "GestureMapping", back_populates="profile", cascade="all, delete-orphan"
    )


class GestureMapping(Base):
    __tablename__ = "gesture_mappings"
    __table_args__ = (
        UniqueConstraint("profile_id", "gesture_name", name="uq_profile_gesture_name"),
    )

    id = Column(Integer, primary_key=True, index=True)
    profile_id = Column(Integer, ForeignKey("user_profiles.id", ondelete="CASCADE"), nullable=False)
    gesture_name = Column(String(32), nullable=False)
    mapped_action = Column(String(64), nullable=False)

    profile = relationship("UserProfile", back_populates="gesture_mappings")


class ActionLog(Base):
    __tablename__ = "action_logs"
    __table_args__ = (
        CheckConstraint("gaze_confidence >= 0.0 AND gaze_confidence <= 1.0", name="ck_log_gaze_conf"),
        CheckConstraint("gesture_confidence >= 0.0 AND gesture_confidence <= 1.0", name="ck_log_gesture_conf"),
        CheckConstraint("voice_confidence >= 0.0 AND voice_confidence <= 1.0", name="ck_log_voice_conf"),
        CheckConstraint("fusion_score >= 0.0 AND fusion_score <= 1.0", name="ck_log_fusion_score"),
        CheckConstraint("risk_level IN ('LOW', 'MEDIUM', 'HIGH')", name="ck_log_risk_enum"),
        Index("ix_action_logs_intent_risk_status", "resolved_intent", "risk_level", "status"),
    )

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    gaze_target = Column(String(128), nullable=True)
    gaze_confidence = Column(Float, default=0.0, nullable=False)
    gesture_detected = Column(String(64), nullable=True)
    gesture_confidence = Column(Float, default=0.0, nullable=False)
    voice_command = Column(Text, nullable=True)
    voice_confidence = Column(Float, default=0.0, nullable=False)

    resolved_intent = Column(String(64), nullable=False, index=True)
    target_entity = Column(String(256), nullable=True)
    fusion_score = Column(Float, default=0.0, nullable=False, index=True)
    risk_level = Column(String(16), default="LOW", nullable=False, index=True)
    status = Column(String(32), default="SUCCESS", nullable=False, index=True)
    latency_ms = Column(Float, default=0.0, nullable=False)


class BenchmarkRecord(Base):
    __tablename__ = "benchmark_records"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    mode = Column(String(32), nullable=False, index=True)
    expected_intent = Column(String(64), nullable=False)
    actual_intent = Column(String(64), nullable=False)
    is_correct = Column(Boolean, default=True, nullable=False)
    false_activation = Column(Boolean, default=False, nullable=False)
    latency_ms = Column(Float, default=0.0, nullable=False)