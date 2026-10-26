
export type Telemetry = {
  privacy: {
    camera_on: boolean;
    microphone_on: boolean;
    voice_processing_active: boolean;
  };

  gaze: {
    x: number;
    y: number;
    target: string | null;
    confidence: number;
    dwell_progress: number;
  };

  gesture: {
    gesture: string;
    mapped_action: string;
    confidence: number;
    cursor_x: number;
    cursor_y: number;
  };

  voice: {
    text: string;
    is_listening: boolean;
    is_processing: boolean;
    confidence: number;
  };

  fusion_score: number;

  active_modalities: string[];

  resolved_intent: {
    intent: string;
    target?: string | null;
  } | null;

  context: {
    active_application: string;
    last_target: string | null;
    last_intent: string | null;
    recent_targets: string[];
  };

  security_prompt: {
    active: boolean;
    intent_summary: string;
    risk_level: string;
    message: string;
  };

  camera_Preview_b64?: string | null;
};

export type Profile = {
  id: number;
  profile_name: string;
  dwell_time_ms: number;
  gaze_weight: number;
  gesture_weight: number;
  voice_weight: number;
  confidence_threshold: number;
  enable_gaze: boolean;
  enable_gesture: boolean;
  enable_voice: boolean;

  gesture_mappings: {
    gesture_name: string;
    mapped_action: string;
  }[];
};

export type ActionLog = {
  id: number;
  timestamp: string;
  resolved_intent: string;
  target_entity: string | null;
  fusion_score: number;
  risk_level: string;
  status: string;
  latency_ms: number;
  voice_command: string | null;
};
