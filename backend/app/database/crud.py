from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.database.models import ActionLog, BenchmarkRecord, GestureMapping, UserProfile
from app.schemas import UserProfileUpdate


DEFAULT_GESTURES_USER_A = {
    "POINT": "MOVE_CURSOR",
    "PINCH": "CLICK",
    "TWO_FINGERS": "SCROLL",
    "PALM": "STOP",
    "FIST": "GRAB",
    "THUMBS_UP": "CONFIRM",
    "THUMBS_DOWN": "CANCEL",
}

DEFAULT_GESTURES_USER_B = {
    "POINT": "MOVE_CURSOR",
    "PINCH": "SELECT",
    "TWO_FINGERS": "SCROLL",
    "PALM": "PAUSE_MEDIA",
    "FIST": "GRAB",
    "THUMBS_UP": "PLAY_MEDIA",
    "THUMBS_DOWN": "CANCEL",
}


def init_db_and_seed(db: Session) -> None:
    """Seeds User A and User B profiles from the SYNTRA Synopsis if DB is empty."""
    existing = db.query(UserProfile).first()
    if existing:
        return

    # Create User A (Default Active Profile)
    user_a = UserProfile(
        profile_name="User A (Desktop Control)",
        is_active=True,
        dwell_time_ms=1200,
        gaze_weight=0.40,
        gesture_weight=0.30,
        voice_weight=0.30,
        confidence_threshold=0.70,
        enable_gaze=True,
        enable_gesture=True,
        enable_voice=True,
    )
    db.add(user_a)
    db.flush()

    for gesture, action in DEFAULT_GESTURES_USER_A.items():
        db.add(GestureMapping(profile_id=user_a.id, gesture_name=gesture, mapped_action=action))

    # Create User B (Media & Accessibility Profile)
    user_b = UserProfile(
        profile_name="User B (Media & Custom)",
        is_active=False,
        dwell_time_ms=1000,
        gaze_weight=0.45,
        gesture_weight=0.30,
        voice_weight=0.25,
        confidence_threshold=0.65,
        enable_gaze=True,
        enable_gesture=True,
        enable_voice=True,
    )
    db.add(user_b)
    db.flush()

    for gesture, action in DEFAULT_GESTURES_USER_B.items():
        db.add(GestureMapping(profile_id=user_b.id, gesture_name=gesture, mapped_action=action))

    db.commit()


def get_active_profile(db: Session) -> UserProfile:
    profile = db.query(UserProfile).filter(UserProfile.is_active == True).first()
    if not profile:
        init_db_and_seed(db)
        profile = db.query(UserProfile).filter(UserProfile.is_active == True).first()
    return profile


def get_all_profiles(db: Session) -> List[UserProfile]:
    return db.query(UserProfile).all()


def switch_active_profile(db: Session, profile_name: str) -> Optional[UserProfile]:
    target = db.query(UserProfile).filter(UserProfile.profile_name == profile_name).first()
    if not target:
        return None
    db.query(UserProfile).update({UserProfile.is_active: False})
    target.is_active = True
    db.commit()
    db.refresh(target)
    return target


def update_active_profile(db: Session, payload: UserProfileUpdate) -> UserProfile:
    if payload.active_profile_name:
        switched = switch_active_profile(db, payload.active_profile_name)
        if switched:
            profile = switched
        else:
            profile = get_active_profile(db)
    else:
        profile = get_active_profile(db)

    if payload.dwell_time_ms is not None:
        profile.dwell_time_ms = payload.dwell_time_ms
    if payload.gaze_weight is not None:
        profile.gaze_weight = payload.gaze_weight
    if payload.gesture_weight is not None:
        profile.gesture_weight = payload.gesture_weight
    if payload.voice_weight is not None:
        profile.voice_weight = payload.voice_weight
    if payload.confidence_threshold is not None:
        profile.confidence_threshold = payload.confidence_threshold
    if payload.enable_gaze is not None:
        profile.enable_gaze = payload.enable_gaze
    if payload.enable_gesture is not None:
        profile.enable_gesture = payload.enable_gesture
    if payload.enable_voice is not None:
        profile.enable_voice = payload.enable_voice

    if payload.gesture_mappings is not None:
        db.query(GestureMapping).filter(GestureMapping.profile_id == profile.id).delete()
        for item in payload.gesture_mappings:
            db.add(
                GestureMapping(
                    profile_id=profile.id,
                    gesture_name=item.gesture_name,
                    mapped_action=item.mapped_action,
                )
            )

    db.commit()
    db.refresh(profile)
    return profile


def create_action_log(
    db: Session,
    resolved_intent: str,
    target_entity: Optional[str] = None,
    gaze_target: Optional[str] = None,
    gaze_conf: float = 0.0,
    gesture: Optional[str] = None,
    gesture_conf: float = 0.0,
    voice_text: Optional[str] = None,
    voice_conf: float = 0.0,
    fusion_score: float = 0.0,
    risk_level: str = "LOW",
    status: str = "SUCCESS",
    latency_ms: float = 0.0,
) -> ActionLog:
    log = ActionLog(
        gaze_target=gaze_target,
        gaze_confidence=round(gaze_conf, 3),
        gesture_detected=gesture,
        gesture_confidence=round(gesture_conf, 3),
        voice_command=voice_text,
        voice_confidence=round(voice_conf, 3),
        resolved_intent=resolved_intent,
        target_entity=target_entity,
        fusion_score=round(fusion_score, 3),
        risk_level=risk_level,
        status=status,
        latency_ms=round(latency_ms, 2),
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def get_recent_logs(db: Session, limit: int = 50) -> List[ActionLog]:
    return db.query(ActionLog).order_by(ActionLog.timestamp.desc()).limit(limit).all()


def get_analytics_summary(db: Session) -> Dict:
    total = db.query(func.count(ActionLog.id)).scalar() or 0
    success_count = (
        db.query(func.count(ActionLog.id)).filter(ActionLog.status == "SUCCESS").scalar() or 0
    )
    blocked_count = (
        db.query(func.count(ActionLog.id))
        .filter(ActionLog.status.in_(["BLOCKED", "CANCELLED"]))
        .scalar()
        or 0
    )
    avg_latency = db.query(func.avg(ActionLog.latency_ms)).scalar() or 0.0
    avg_confidence = db.query(func.avg(ActionLog.fusion_score)).scalar() or 0.0

    return {
        "total_actions": total,
        "success_rate": round((success_count / total) * 100, 1) if total > 0 else 100.0,
        "security_interceptions": blocked_count,
        "avg_latency_ms": round(avg_latency, 1),
        "avg_fusion_confidence": round(avg_confidence * 100, 1),
    }