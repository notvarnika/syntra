import math
import queue
import threading
import time
from typing import Callable, Optional
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from app.config import settings
from app.schemas import VoiceState


class VoiceEngine:
    """
    Real-time Speech Recognition Module using sounddevice + Faster-Whisper.
    Captures 16kHz mono audio, segments speech via RMS Voice Activity Detection (VAD),
    and transcribes utterances locally with confidence scoring.
    """

    def __init__(self) -> None:
        self.sample_rate: int = settings.MIC_SAMPLE_RATE
        self.model_size: str = settings.WHISPER_MODEL_SIZE
        self.compute_type: str = settings.WHISPER_COMPUTE_TYPE

        self._model: Optional[WhisperModel] = None
        self._running: bool = False
        self._is_listening: bool = False
        self._is_processing: bool = False

        self._audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._stream: Optional[sd.InputStream] = None

        # VAD Parameters (Tuned for desktop microphones)
        self.energy_threshold: float = 0.035     # RMS amplitude to trigger speech start
        self.silence_duration_s: float = 0.75    # Seconds of silence to mark end of phrase
        self.min_phrase_duration_s: float = 0.35 # Ignore brief clicks/coughs
        self.max_phrase_duration_s: float = 8.0  # Cap single utterance length

        self._latest_state = VoiceState(
            text="", is_listening=False, is_processing=False, confidence=0.0, timestamp=0.0
        )
        self._on_utterance_callback: Optional[Callable[[VoiceState], None]] = None

    @property
    def is_running(self) -> bool:
        return self._running

    def set_callback(self, callback: Callable[[VoiceState], None]) -> None:
        """Registers a callback triggered immediately whenever a phrase is transcribed."""
        self._on_utterance_callback = callback

    def get_latest_state(self) -> VoiceState:
        self._latest_state.is_listening = self._is_listening
        self._latest_state.is_processing = self._is_processing
        return self._latest_state

    def load_model(self) -> None:
        if self._model is None:
            print(
                f"[SYNTRA VoiceEngine] Loading Faster-Whisper ({self.model_size}, {self.compute_type})..."
            )
            self._model = WhisperModel(
                self.model_size, device="cpu", compute_type=self.compute_type
            )
            print("[SYNTRA VoiceEngine] Faster-Whisper loaded successfully.")

    def start(self) -> bool:
        if self._running:
            return True

        try:
            self.load_model()
            self._running = True
            self._is_listening = True

            # Start 16kHz mono microphone stream (100ms blocks)
            block_size = int(self.sample_rate * 0.1)
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=block_size,
                callback=self._audio_callback,
            )
            self._stream.start()

            self._worker_thread = threading.Thread(target=self._vad_and_transcribe_loop, daemon=True)
            self._worker_thread.start()
            print("[SYNTRA VoiceEngine] Microphone stream active (16kHz Mono).")
            return True
        except Exception as exc:
            print(f"[SYNTRA VoiceEngine] ERROR starting microphone: {exc}")
            self._running = False
            self._is_listening = False
            return False

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if self._running:
            self._audio_queue.put(indata[:, 0].copy())

    def _vad_and_transcribe_loop(self) -> None:
        speech_buffer = []
        speaking = False
        silence_Start: Optional[float] = None
        speech_start_time: float = 0.0

        while self._running:
            try:
                chunk = self._audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            rms = float(np.sqrt(np.mean(np.square(chunk))))
            now = time.time()

            if rms >= self.energy_threshold:
                if not speaking:
                    speaking = True
                    speech_start_time = now
                    speech_buffer = []
                speech_buffer.append(chunk)
                silence_Start = None

                # Force transcription if user speaks continuously past max duration
                if (now - speech_start_time) >= self.max_phrase_duration_s:
                    self._process_speech_buffer(speech_buffer)
                    speaking = False
                    speech_buffer = []
            else:
                if speaking:
                    speech_buffer.append(chunk)
                    if silence_Start is None:
                        silence_Start = now
                    elif (now - silence_Start) >= self.silence_duration_s:
                        duration = now - speech_start_time - self.silence_duration_s
                        if duration >= self.min_phrase_duration_s:
                            self._process_speech_buffer(speech_buffer)
                        speaking = False
                        speech_buffer = []
                        silence_Start = None

    def _process_speech_buffer(self, chunks: list) -> None:
        if not chunks or self._model is None:
            return

        self._is_processing = True
        try:
            audio_np = np.concatenate(chunks, axis=0).astype(np.float32)

            segments, _ = self._model.transcribe(
                audio_np,
                language="en",
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False,
            )

            texts = []
            probs = []
            for seg in segments:
                cleaned = seg.text.strip()
                if cleaned:
                    texts.append(cleaned)
                    # Convert average log probability to [0.0, 1.0] confidence score
                    probs.append(math.exp(max(seg.avg_logprob, -2.5)))

            full_text = " ".join(texts).strip()
            # Filter out common Whisper silence hallucinations
            if full_text and full_text.lower() not in (
                "you",
                "thank you.",
                "thanks for watching!",
                ".",
            ):
                avg_conf = round(float(np.mean(probs)), 3) if probs else 0.85
                avg_conf = min(max(avg_conf, 0.50), 0.99)

                state = VoiceState(
                    text=full_text,
                    is_listening=True,
                    is_processing=False,
                    confidence=avg_conf,
                    timestamp=time.time(),
                )
                self._latest_state = state
                print(f"[SYNTRA VoiceEngine] Heard: '{full_text}' (Confidence: {int(avg_conf*100)}%)")

                if self._on_utterance_callback:
                    self._on_utterance_callback(state)
        except Exception as exc:
            print(f"[SYNTRA VoiceEngine] Transcription error: {exc}")
        finally:
            self._is_processing = False

    def stop(self) -> None:
        self._running = False
        self._is_listening = False
        self._is_processing = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.5)
        print("[SYNTRA VoiceEngine] Microphone stopped.")


voice_engine = VoiceEngine()