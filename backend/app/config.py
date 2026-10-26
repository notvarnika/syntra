import logging
from pathlib import Path
from typing import List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    PROJECT_NAME: str = "SYNTRA - Multimodal HCI Framework"
    VERSION: str = "1.0.0"
    SYNTRA_ENV: str = Field(default="development")

    # Network & API Security (Defaults to localhost & explicit CORS origins)
    HOST: str = Field(default="127.0.0.1")
    PORT: int = Field(default=8000, ge=1024, le=65535)
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173"]
    )
    API_AUTH_TOKEN: str = Field(default="")  # If set, required via X-Syntra-Token header

    # Database & Persistent Paths
    DATABASE_URL: str = Field(default=f"sqlite:///{DATA_DIR / 'syntra_dev.db'}")
    CALIBRATION_FILE: Path = DATA_DIR / "calibration_matrix.json"
    BENCHMARK_FILE: Path = DATA_DIR / "benchmarks.csv"

    # Hardware Settings
    CAMERA_INDEX: int = Field(default=0, ge=0, le=10)
    CAMERA_WIDTH: int = Field(default=1280, ge=320, le=3840)
    CAMERA_HEIGHT: int = Field(default=720, ge=240, le=2160)
    CAMERA_FPS: int = Field(default=30, ge=5, le=120)
    MIC_SAMPLE_RATE: int = Field(default=16000, ge=8000, le=48000)

    # Multimodal Fusion Defaults
    DEFAULT_GAZE_WEIGHT: float = Field(default=0.40, ge=0.0, le=1.0)
    DEFAULT_GESTURE_WEIGHT: float = Field(default=0.30, ge=0.0, le=1.0)
    DEFAULT_VOICE_WEIGHT: float = Field(default=0.30, ge=0.0, le=1.0)
    DEFAULT_CONFIDENCE_THRESHOLD: float = Field(default=0.70, ge=0.1, le=1.0)
    TEMPORAL_WINDOW_SECONDS: float = Field(default=2.5, ge=0.5, le=10.0)
    DEFAULT_DWELL_TIME_MS: int = Field(default=1200, ge=300, le=5000)

    # ASR & LLM Settings
    WHISPER_MODEL_SIZE: str = Field(default="base.en")
    WHISPER_COMPUTE_TYPE: str = Field(default="int8")
    LLM_PROVIDER: str = Field(default="ollama")
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434")
    OLLAMA_MODEL: str = Field(default="llama3.2:3b")
    GEMINI_API_KEY: str = Field(default="")
    OPENAI_API_KEY: str = Field(default="")

    @field_validator("WHISPER_MODEL_SIZE")
    @classmethod
    def validate_whisper_model(cls, v: str) -> str:
        allowed = {"tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en"}
        if v not in allowed:
            raise ValueError(f"WHISPER_MODEL_SIZE must be one of {allowed}")
        return v


settings = Settings()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)