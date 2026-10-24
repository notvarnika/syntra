import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env")


class Settings:
    PROJECT_NAME: str = "SYNTRA - Multimodal HCI Framework"
    VERSION: str = "1.0.0"
    ENV: str = os.getenv("SYNTRA_ENV", "development")

    # Database (SQLite for Dev, PostgreSQL ready for Prod)
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", f"sqlite:///{DATA_DIR / 'syntra_dev.db'}"
    )

    # Persistent Data Paths
    CALIBRATION_FILE: Path = DATA_DIR / "calibration_matrix.json"
    BENCHMARK_FILE: Path = DATA_DIR / "benchmarks.csv"

    # Camera & Microphone Hardware Settings
    CAMERA_INDEX: int = int(os.getenv("CAMERA_INDEX", 0))
    CAMERA_WIDTH: int = int(os.getenv("CAMERA_WIDTH", 1280))
    CAMERA_HEIGHT: int = int(os.getenv("CAMERA_HEIGHT", 720))
    CAMERA_FPS: int = int(os.getenv("CAMERA_FPS", 30))
    MIC_SAMPLE_RATE: int = int(os.getenv("MIC_SAMPLE_RATE", 16000))

    # Multimodal Fusion Defaults (Synopsis Formula: 0.40 Gaze + 0.30 Gesture + 0.30 Voice)
    DEFAULT_GAZE_WEIGHT: float = 0.40
    DEFAULT_GESTURE_WEIGHT: float = 0.30
    DEFAULT_VOICE_WEIGHT: float = 0.30
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.85
    TEMPORAL_WINDOW_SECONDS: float = 2.5  # 2-3 second interaction window
    DEFAULT_DWELL_TIME_MS: int = 3000     # Dwell selection time for Demo 3

    # ASR & LLM Configuration
    WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "base.en")
    WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")


settings = Settings()