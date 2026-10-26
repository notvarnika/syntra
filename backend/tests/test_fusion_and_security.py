import pytest
from app.intelligence.fusion_engine import MultimodalFusionEngine
from app.execution.security_layer import SecurityLayer
from app.schemas import ExecutionStatus, GazeState, GestureState, RiskLevel, StructuredIntent, VoiceState


def test_unimodal_bimodal_trimodal_scoring():
    engine = MultimodalFusionEngine()
    # Unimodal Gaze
    score_u, mods_u = engine.compute_weighted_score(0.90, 0.0, 0.0)
    assert score_u == pytest.approx(0.90, rel=1e-3)
    assert mods_u == ["GAZE"]

    # Bimodal Gaze + Gesture
    score_b, mods_b = engine.compute_weighted_score(0.92, 0.96, 0.0)
    expected_b = (0.40 * 0.92 + 0.30 * 0.96) / 0.70
    assert score_b == pytest.approx(expected_b, rel=1e-3)
    assert mods_b == ["GAZE", "GESTURE"]

    # Trimodal Gaze + Gesture + Voice (Synopsis formula: 0.40*0.92 + 0.30*0.96 + 0.30*0.87 = 0.917)
    score_t, mods_t = engine.compute_weighted_score(0.92, 0.96, 0.87)
    assert score_t == pytest.approx(0.917, rel=1e-3)
    assert mods_t == ["GAZE", "GESTURE", "VOICE"]


def test_disabled_modality_excluded_from_score_and_active_mods():
    engine = MultimodalFusionEngine()
    engine.update_weights_and_settings(
        w_gaze=0.40, w_gesture=0.30, w_voice=0.30, threshold=0.70,
        enable_gaze=False, enable_gesture=True, enable_voice=True,
    )
    score, mods = engine.compute_weighted_score(0.99, 0.80, 0.90)
    assert "GAZE" not in mods
    assert mods == ["GESTURE", "VOICE"]
    assert score == pytest.approx(0.85, rel=1e-3)


def test_falsy_zero_timestamp_and_chronological_ordering():
    engine = MultimodalFusionEngine()
    # Ingest out-of-order timestamps including t=0.0
    engine.ingest_gaze(GazeState(x=100, y=100, target="Chrome", confidence=0.9, timestamp=1.5))
    engine.ingest_gaze(GazeState(x=200, y=200, target="YouTube", confidence=0.9, timestamp=0.5))
    timestamps = [e.timestamp for e in engine._buffer]
    assert timestamps == [0.5, 1.5]

    # Pruning at now=0.0 should NOT fall back to time.time() and wipe future events
    engine._prune_old_events(now=0.0)
    assert len(engine._buffer) == 2


def test_stale_events_pruned():
    engine = MultimodalFusionEngine()
    engine.ingest_gaze(GazeState(x=100, y=100, target="Chrome", confidence=0.95, timestamp=10.0))
    # 3.0s later (> 2.5s window), Chrome should be pruned
    best = engine._get_best_recent_gaze(now=13.0)
    assert best is None


def test_conflicting_intent_blocked():
    engine = MultimodalFusionEngine()
    engine.ingest_gesture(
        GestureState(gesture="THUMBS_DOWN", mapped_action="CANCEL", confidence=0.95, timestamp=100.0)
    )
    res = engine.evaluate_voice_trigger(
        VoiceState(text="Open Chrome", confidence=0.92, timestamp=100.2)
    )
    assert res is None


def test_security_layer_blocks_unresolved_destructive_target():
    sec = SecurityLayer()
    status, risk, _, _ = sec.evaluate(
        StructuredIntent(intent="DELETE_FILE", target="IT_REF"), fusion_score=0.95
    )
    assert status == ExecutionStatus.BLOCKED
    assert risk == RiskLevel.HIGH