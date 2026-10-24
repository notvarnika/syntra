import asyncio
from contextlib import asynccontextmanager
import time
from typing import List, Optional
import cv2
from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.config import settings
from app.database import crud
from app.database.db import Base, SessionLocal, engine, get_db
from app.evaluation.benchmark import benchmark_evaluator
from app.execution.action_executor import action_executor
from app.execution.security_layer import security_layer
from app.intelligence.context_engine import context_engine
from app.intelligence.fusion_engine import fusion_engine
from app.perception.camera_manager import camera_manager
from app.perception.gaze_tracker import gaze_tracker
from app.perception.gesture_tracker import gesture_tracker
from app.perception.voice_engine import voice_engine
from app.schemas import (
    ActionLogResponse,
    CalibrationRequest,
    ExecutionStatus,
    GazeState,
    GestureState,
    PrivacyStatus,
    StructuredIntent,
    TelemetryFrame,
    UserProfileResponse,
    UserProfileUpdate,
    VoiceState,
)


class SyntraRuntimeController:
    """
    Master runtime orchestrator connecting Perception -> Fusion -> Context -> Security -> Execution
    and broadcasting 15Hz live telemetry to connected React Dashboard WebSocket clients.
    """

    def __init__(self) -> None:
        self.running: bool = False
        self.os_control_enabled: bool = True  # Toggle to pause physical OS mouse movement while configuring UI
        self.clients: List[WebSocket] = []

        self.latest_gaze = GazeState()
        self.latest_gesture = GestureState()
        self.latest_voice = VoiceState()
        self.latest_fusion_score: float = 0.0
        self.latest_active_mods: List[str] = []
        self.latest_resolved_intent: Optional[StructuredIntent] = None
        
        # [NEW] Global cooldown tracker to prevent runaway executions
        self.last_exec_time: float = 0.0

    def sync_profile_from_db(self) -> None:
        db = SessionLocal()
        try:
            crud.init_db_and_seed(db)
            profile = crud.get_active_profile(db)
            fusion_engine.update_weights_and_settings(
                w_gaze=profile.gaze_weight,
                w_gesture=profile.gesture_weight,
                w_voice=profile.voice_weight,
                threshold=profile.confidence_threshold,
                enable_gaze=profile.enable_gaze,
                enable_gesture=profile.enable_gesture,
                enable_voice=profile.enable_voice,
            )
            gaze_tracker.set_dwell_time(profile.dwell_time_ms)
            mapping_dict = {
                m.gesture_name: m.mapped_action for m in profile.gesture_mappings
            }
            gesture_tracker.update_gesture_mappings(mapping_dict)
        finally:
            db.close()

    def handle_voice_utterance(self, voice_state: VoiceState) -> None:
        """Callback invoked immediately by VoiceEngine when an utterance is transcribed."""
        self.latest_voice = voice_state
        start_t = time.perf_counter()

        eval_result = fusion_engine.evaluate_voice_trigger(voice_state)
        if eval_result is None:
            return

        resolved_intent, score, active_mods, meta = eval_result
        self._process_and_execute(resolved_intent, score, active_mods, meta, start_t)

    def _process_and_execute(
        self,
        intent: StructuredIntent,
        score: float,
        active_mods: List[str],
        meta: dict,
        start_perf: float,
    ) -> None:
        
        # [NEW] Enforce 3.0 second cooldown between automated commands
        now = time.time()
        if (now - self.last_exec_time) < 3.0:
            return
            
        self.latest_fusion_score = score
        self.latest_active_mods = active_mods
        self.latest_resolved_intent = intent

        # [NEW] Print the exact Trigger context directly to your terminal
        print(f"\n==================================================")
        print(f"[SYNTRA TRIGGERED] Modalities: {active_mods}")
        print(f"[SYNTRA INTENT] Action: {intent.intent} | Target: {intent.target}")

        # Pass through Security Layer (Risk classification + Confirmation state machine)
        status, risk, approved_intent, msg = security_layer.evaluate(
            intent, score, meta
        )

        if status == ExecutionStatus.SUCCESS and approved_intent is not None:
            gaze_xy = (
                (self.latest_gaze.x, self.latest_gaze.y)
                if self.latest_gaze.confidence > 0
                else None
            )
            cursor_xy = (
                (self.latest_gesture.cursor_x, self.latest_gesture.cursor_y)
                if self.latest_gesture.confidence > 0
                else None
            )
            ok, exec_msg = action_executor.execute(
                approved_intent, gaze_xy=gaze_xy, cursor_xy=cursor_xy
            )
            if ok:
                context_engine.record_executed_intent(approved_intent)
            final_status = "SUCCESS" if ok else "FAILED"
            
            # [NEW] Log the exact execution result status
            print(f"[SYNTRA EXECUTED] Status: {final_status} | {exec_msg}")
            
        elif status == ExecutionStatus.AWAITING_CONFIRMATION:
            final_status = "AWAITING_CONFIRMATION"
            print(f"[SYNTRA EXECUTED] Status: {final_status} | {msg}")
        elif status == ExecutionStatus.CANCELLED:
            final_status = "CANCELLED"
            print(f"[SYNTRA EXECUTED] Status: {final_status} | {msg}")
        else:
            final_status = "BLOCKED"
            print(f"[SYNTRA EXECUTED] Status: {final_status} | {msg}")

        print(f"==================================================\n")
        
        # [NEW] Update the cooldown timer so it waits 3 seconds before next trigger
        self.last_exec_time = time.time()

        latency_ms = (time.perf_counter() - start_perf) * 1000.0

        # Persist to SQLite / PostgreSQL ActionLog
        db = SessionLocal()
        try:
            crud.create_action_log(
                db=db,
                resolved_intent=(
                    approved_intent.intent if approved_intent else intent.intent
                ),
                target_entity=(
                    approved_intent.target if approved_intent else intent.target
                ),
                gaze_target=meta.get("gaze_target"),
                gaze_conf=meta.get("gaze_conf", 0.0),
                gesture=meta.get("gesture"),
                gesture_conf=meta.get("gesture_conf", 0.0),
                voice_text=meta.get("voice_text"),
                voice_conf=meta.get("voice_conf", 0.0),
                fusion_score=score,
                risk_level=risk.value,
                status=final_status,
                latency_ms=latency_ms,
            )
        finally:
            db.close()

    async def perception_and_telemetry_loop(self) -> None:
        """
        Main asynchronous background loop running at ~20Hz:
        1. Grabs shared webcam frame.
        2. Runs Gaze Tracking & Hand Gesture Recognition.
        3. Evaluates Multimodal Fusion & executes approved OS actions.
        4. Broadcasts TelemetryFrame over WebSockets to the React Dashboard.
        """
        self.running = True
        while self.running:
            loop_start = time.perf_counter()
            bgr_frame, rgb_frame, _ = camera_manager.get_latest_frame()

            if rgb_frame is not None and bgr_frame is not None:
                annotated = bgr_frame.copy()

                # 1. Run Gaze Tracking Module (WHERE)
                if fusion_engine.enable_gaze:
                    gaze_state, dwell_fired = gaze_tracker.process_frame(
                        rgb_frame, bgr_canvas=annotated
                    )
                    self.latest_gaze = gaze_state
                    fusion_engine.ingest_gaze(gaze_state)

                    # Demo 3: Gaze Dwell Selection trigger
                    # if dwell_fired and self.os_control_enabled:
                        # dwell_res = fusion_engine.evaluate_dwell_trigger(gaze_state)
                        # if dwell_res:
                        #     self._process_and_execute(*dwell_res, loop_start)

                # 2. Run Hand Gesture Module (HOW)
                if fusion_engine.enable_gesture:
                    gesture_state, scroll_delta = gesture_tracker.process_frame(
                        rgb_frame, bgr_canvas=annotated
                    )
                    self.latest_gesture = gesture_state

                    if self.os_control_enabled:
                        if gesture_state.gesture == "POINT":
                            action_executor.move_cursor(
                                gesture_state.cursor_x, gesture_state.cursor_y
                            )
                        elif gesture_state.gesture == "TWO_FINGERS":
                            action_executor.scroll_direct(scroll_delta)
                        elif gesture_state.gesture not in ("NONE", "POINT", "TWO_FINGERS"):
                            gest_res = fusion_engine.evaluate_gesture_trigger(
                                gesture_state
                            )
                            if gest_res:
                                self._process_and_execute(*gest_res, loop_start)

                # Draw target HUD label on camera preview
                if self.latest_gaze.target:
                    cv2.putText(
                        annotated,
                        f"Gaze Target: {self.latest_gaze.target} ({int(self.latest_gaze.confidence*100)}%)",
                        (16, 34),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 255),
                        2,
                    )
                if self.latest_gesture.gesture != "NONE":
                    cv2.putText(
                        annotated,
                        f"Gesture: {self.latest_gesture.gesture} -> {self.latest_gesture.mapped_action}",
                        (16, 68),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 128),
                        2,
                    )
                camera_manager.update_annotated_frame(annotated)

            # Update continuous fusion meter for live display
            live_score, live_mods = fusion_engine.compute_weighted_score(
                self.latest_gaze.confidence if fusion_engine.enable_gaze else 0.0,
                self.latest_gesture.confidence
                if (fusion_engine.enable_gesture and self.latest_gesture.gesture != "NONE")
                else 0.0,
                self.latest_voice.confidence
                if (
                    fusion_engine.enable_voice
                    and (time.time() - self.latest_voice.timestamp) < 3.0
                )
                else 0.0,
            )
            if live_score > 0:
                self.latest_fusion_score = live_score
                self.latest_active_mods = live_mods

            # Broadcast WebSocket Telemetry to React Dashboard
            if self.clients:
                await self._broadcast_telemetry()

            elapsed = time.perf_counter() - loop_start
            await asyncio.sleep(max(0.015, 0.05 - elapsed))

    async def _broadcast_telemetry(self) -> None:
        voice_live = voice_engine.get_latest_state()
        frame_packet = TelemetryFrame(
            privacy=PrivacyStatus(
                camera_on=camera_manager.is_running,
                microphone_on=voice_engine.is_running,
                voice_processing_active=voice_live.is_processing,
            ),
            gaze=self.latest_gaze,
            gesture=self.latest_gesture,
            voice=voice_live,
            fusion_score=round(self.latest_fusion_score, 3),
            active_modalities=self.latest_active_mods,
            resolved_intent=self.latest_resolved_intent,
            context=context_engine.get_snapshot(),
            security_prompt=security_layer.get_active_prompt(),
            camera_Preview_b64=camera_manager.get_jpeg_base64(width=420),
        )
        payload_json = frame_packet.model_dump_json()
        stale_clients = []
        for ws in self.clients: 
            try:
                await ws.send_text(payload_json)
            except Exception:
                stale_clients.append(ws)
        for ws in stale_clients:
            if ws in self.clients:
                self.clients.remove(ws)


runtime = SyntraRuntimeController()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize SQLite/PostgreSQL tables & seed User A / User B
    Base.metadata.create_all(bind=engine)
    runtime.sync_profile_from_db()

    # Start Camera & Voice Perception Engines
    camera_manager.start()
    voice_engine.set_callback(runtime.handle_voice_utterance)
    voice_engine.start()

    # Launch async perception & WebSocket telemetry task
    loop_task = asyncio.create_task(runtime.perception_and_telemetry_loop())
    yield

    # Clean shutdown
    runtime.running = False
    loop_task.cancel()
    voice_engine.stop()
    camera_manager.stop()


app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# WEBSOCKET ENDPOINT (15-20Hz Real-Time Telemetry Stream)
# ---------------------------------------------------------------------------
@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    runtime.clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in runtime.clients:
            runtime.clients.remove(websocket)


# ---------------------------------------------------------------------------
# REST API ENDPOINTS FOR DASHBOARD, PERSONALIZATION, CALIBRATION & DEMOS
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "project": settings.PROJECT_NAME,
        "camera_on": camera_manager.is_running,
        "mic_on": voice_engine.is_running,
        "os_control_enabled": runtime.os_control_enabled,
    }


@app.post("/api/control/toggle")
def toggle_os_control(enable: Optional[bool] = None):
    if enable is None:
        runtime.os_control_enabled = not runtime.os_control_enabled
    else:
        runtime.os_control_enabled = enable
    return {"os_control_enabled": runtime.os_control_enabled}


@app.get("/api/profiles", response_model=List[UserProfileResponse])
def list_profiles(db: Session = Depends(get_db)):
    crud.init_db_and_seed(db)
    return crud.get_all_profiles(db)


@app.get("/api/settings", response_model=UserProfileResponse)
def get_active_settings(db: Session = Depends(get_db)):
    return crud.get_active_profile(db)


@app.put("/api/settings", response_model=UserProfileResponse)
def update_settings(payload: UserProfileUpdate, db: Session = Depends(get_db)):
    updated = crud.update_active_profile(db, payload)
    runtime.sync_profile_from_db()
    return updated


@app.get("/api/calibration/current-feature")
def get_current_gaze_feature():
    """Returns current raw iris/head feature (fx, fy) when user clicks a calibration point."""
    return {
        "feature_x": gaze_tracker.latest_raw_fx,
        "feature_y": gaze_tracker.latest_raw_fy,
        "is_calibrated": gaze_tracker.is_calibrated,
    }


@app.post("/api/calibration/save")
def save_calibration(req: CalibrationRequest):
    ok = gaze_tracker.calibrate_9_points(req.points)
    return {"success": ok, "is_calibrated": gaze_tracker.is_calibrated}


@app.post("/api/security/respond")
def respond_to_security_prompt(decision: str):
    """Allows confirming or cancelling a High-Risk prompt from the Dashboard UI as well as Gesture/Voice."""
    cmd = "CONFIRM" if decision.upper() == "CONFIRM" else "CANCEL"
    runtime.handle_voice_utterance(
        VoiceState(text=cmd, confidence=0.99, timestamp=time.time())
    )
    return {"status": "processed", "decision": cmd}


@app.post("/api/simulate/demo/{demo_id}")
def trigger_synopsis_demo(demo_id: int):
    """
    One-click simulator for the 6 Expected Final Demonstrations in the SYNTRA Synopsis
    (Also works alongside live physical Gaze, Gesture, and Voice inputs).
    """
    now = time.time()
    if demo_id == 1:
        fusion_engine.ingest_gaze(
            GazeState(x=220, y=120, target="Chrome", confidence=0.92, timestamp=now)
        )
        res = fusion_engine.evaluate_gesture_trigger(
            GestureState(
                gesture="PINCH",
                mapped_action="CLICK",
                cursor_x=220,
                cursor_y=120,
                confidence=0.96,
                timestamp=now + 0.1,
            )
        )
        if res:
            runtime._process_and_execute(*res, time.perf_counter())
        return {"demo": 1, "description": "Gesture Control: Point at Chrome + Pinch -> Open Chrome"}

    if demo_id == 2:
        runtime.handle_voice_utterance(
            VoiceState(text="Open YouTube", confidence=0.91, timestamp=now)
        )
        return {"demo": 2, "description": "Voice Control: 'Open YouTube' -> YouTube Opens"}

    if demo_id == 3:
        gaze = GazeState(
            x=640, y=140, target="YouTube", confidence=0.94, dwell_progress=1.0, timestamp=now
        )
        res = fusion_engine.evaluate_dwell_trigger(gaze)
        if res:
            runtime._process_and_execute(*res, time.perf_counter())
        return {"demo": 3, "description": "Gaze Dwell Selection on Target"}

    if demo_id == 4:
        fusion_engine.ingest_gaze(
            GazeState(x=200, y=120, target="Chrome", confidence=0.92, timestamp=now)
        )
        fusion_engine.ingest_gesture(
            GestureState(
                gesture="PINCH",
                mapped_action="CLICK",
                cursor_x=200,
                cursor_y=120,
                confidence=0.96,
                timestamp=now + 0.4,
            )
        )
        runtime.handle_voice_utterance(
            VoiceState(text="Open it", confidence=0.87, timestamp=now + 0.8)
        )
        return {"demo": 4, "description": "Multimodal Fusion: Gaze(Chrome) + Pinch + 'Open it'"}

    if demo_id == 5:
        runtime.handle_voice_utterance(
            VoiceState(text="Open Chrome", confidence=0.93, timestamp=now)
        )
        runtime.handle_voice_utterance(
            VoiceState(text="Search for Python", confidence=0.90, timestamp=now + 0.5)
        )
        return {
            "demo": 5,
            "description": "Context Chain executed: 'Open Chrome' -> 'Search for Python' (Say 'Open the first result' next!)",
        }

    if demo_id == 6:
        fusion_engine.ingest_gaze(
            GazeState(x=400, y=400, target="important_report.docx", confidence=0.91, timestamp=now)
        )
        runtime.handle_voice_utterance(
            VoiceState(text="Delete this file", confidence=0.89, timestamp=now + 0.2)
        )
        return {
            "demo": 6,
            "description": "Security Layer triggered: High-Risk DELETE_FILE awaiting Thumbs Up or 'Confirm'",
        }

    return {"error": "Invalid demo_id (1-6)"}


@app.get("/api/logs", response_model=List[ActionLogResponse])
def get_action_logs(limit: int = 40, db: Session = Depends(get_db)):
    return crud.get_recent_logs(db, limit=limit)


@app.get("/api/analytics")
def get_analytics_and_benchmarks(db: Session = Depends(get_db)):
    return {
        "summary": crud.get_analytics_summary(db),
        "benchmarks": benchmark_evaluator.get_comparative_metrics(db),
    }