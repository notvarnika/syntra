# SYNTRA — Multimodal Human-Computer Interaction Framework

SYNTRA lets you control your desktop by combining **where you look** (gaze), **how you move your hand** (gestures) and **what you say** (voice). Each input alone is ambiguous; fused together they become a reliable command. Looking at Chrome, pinching and saying "open it" opens Chrome. Risky actions such as deleting a file need an explicit confirmation first.

> **Status:** the Python backend is functional. The React dashboard is under development.

---

## Features

- **Gaze tracking (WHERE):** MediaPipe FaceMesh iris and head-pose features, mapped to screen coordinates, with optional 9-point polynomial calibration and gaze-dwell selection.
- **Hand gestures (HOW):** MediaPipe Hands with seven gestures: `POINT`, `PINCH`, `TWO_FINGERS`, `PALM`, `FIST`, `THUMBS_UP`, `THUMBS_DOWN`.
- **Voice (WHAT):** local speech recognition with Faster-Whisper and energy-based voice activity detection.
- **Intent parser:** a fast rule-based parser for common commands, with an optional local LLM (Ollama) fallback for complex requests.
- **Fusion engine:** a 2.5 s temporal window combines modalities with a weighted confidence score (default 0.40 gaze, 0.30 gesture, 0.30 voice).
- **Context engine:** resolves references like "it", "this" and "the first result" using recent gaze targets and the active window.
- **Security layer:** classifies actions as LOW, MEDIUM or HIGH risk. High-risk actions wait for a thumbs-up or a spoken "confirm" (10 s timeout).
- **Personalization:** per-user profiles with their own weights, thresholds, dwell time and gesture-to-action mappings.
- **Analytics:** every action is logged with its confidence, latency and risk level, with a benchmark module for comparing modes.

## Architecture

```
Camera ──► Gaze Tracker ──┐
       └─► Gesture Tracker ┤
Mic ────► Voice Engine ────┤
                           ▼
                    Fusion Engine ◄── Context Engine
                           │
                           ▼
                    Security Layer
                           │
                           ▼
                    Action Executor ──► OS (mouse, keyboard, apps)
                           │
                           ▼
          SQLite/PostgreSQL logs ──► FastAPI REST + WebSocket ──► React dashboard
```

```
backend/
  run.py                  entry point (uvicorn on :8000)
  app/
    main.py               runtime loop, REST API, WebSocket
    config.py             settings and environment variables
    schemas.py            Pydantic models
    perception/           camera, gaze, gesture, voice, screen inspector
    intelligence/         intent parser, fusion engine, context engine
    execution/            security layer, action executor
    database/             SQLAlchemy models and CRUD
    evaluation/           benchmark recorder
  data/                   calibration matrix, benchmarks CSV, screenshots
frontend/                 React + Vite + Tailwind dashboard (in progress)
```

## Requirements

- Python 3.10 or newer
- A webcam and a microphone
- Windows is the primary target (window inspection uses `pygetwindow`)
- Node.js 18 or newer for the dashboard
- Optional: [Ollama](https://ollama.com) for the LLM fallback parser

## Getting started

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python run.py
```

The API is served at `http://localhost:8000`. The first run downloads the Whisper model.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Optional: PostgreSQL

SQLite is used by default. To use PostgreSQL, start the database and set `DATABASE_URL` (and install `psycopg2-binary`):

```bash
docker compose up -d
```

## Configuration

Create `backend/.env` to override any default:

| Variable | Default | Description |
| --- | --- | --- |
| `DATABASE_URL` | SQLite file in `backend/data` | Database connection string |
| `CAMERA_INDEX` | `0` | Webcam device index |
| `CAMERA_WIDTH` / `CAMERA_HEIGHT` / `CAMERA_FPS` | `1280` / `720` / `30` | Capture settings |
| `MIC_SAMPLE_RATE` | `16000` | Microphone sample rate |
| `WHISPER_MODEL_SIZE` | `base.en` | Faster-Whisper model |
| `WHISPER_COMPUTE_TYPE` | `int8` | Whisper compute type |
| `LLM_PROVIDER` | `ollama` | LLM fallback provider (only Ollama is implemented) |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` | `http://localhost:11434` / `llama3.2:3b` | Ollama settings |
| `SYNTRA_ENV` | `development` | Environment name |

## Using SYNTRA

### Gestures (default profile)

| Gesture | Action |
| --- | --- |
| Point (index finger) | Move the cursor |
| Pinch | Click, or open the app you are looking at |
| Two fingers | Scroll |
| Open palm | Stop |
| Fist | Grab (hold mouse button) |
| Thumbs up / down | Confirm / cancel a pending prompt |

### Example voice commands

`Open Chrome` · `Search for Python` · `Open the first result` · `Open the browser and search for MCA project ideas` · `Scroll down a little` · `Take a screenshot` · `Go back` · `Close that` · `Delete this file` (asks for confirmation)

### Calibration

Open the dashboard's calibration view and click through the 9 points while looking at each one. The mapping is saved to `backend/data/calibration_matrix.json`. Re-calibrate if you change monitor or sit differently.

## Demo scenarios

`POST /api/simulate/demo/{id}` triggers each scenario without hardware input:

| ID | Scenario |
| --- | --- |
| 1 | Look at Chrome and pinch to open it |
| 2 | Voice only: "Open YouTube" |
| 3 | Gaze dwell selection |
| 4 | Fusion: gaze + pinch + "Open it" |
| 5 | Context chain: "Open Chrome", then "Search for Python" |
| 6 | Security: "Delete this file" waits for confirmation |

## API overview

| Endpoint | Purpose |
| --- | --- |
| `WS /ws/telemetry` | Live telemetry stream (gaze, gesture, voice, fusion score, camera preview) |
| `GET /api/health` | Service status |
| `POST /api/control/toggle` | Enable or pause OS control |
| `GET/PUT /api/settings` | Read or update the active profile |
| `GET /api/profiles` | List profiles |
| `GET /api/calibration/current-feature` | Current raw gaze feature |
| `POST /api/calibration/save` | Save a calibration |
| `POST /api/security/respond` | Confirm or cancel a high-risk prompt |
| `GET /api/logs` | Recent action log |
| `GET /api/analytics` | Summary and benchmark metrics |

Interactive docs are at `http://localhost:8000/docs`.

## Safety and privacy

- Camera and microphone processing is **local**. Speech is transcribed on your machine, and the only network call is the optional local Ollama request.
- The server currently listens on all interfaces and allows any origin. Run it only on a trusted network, and see the roadmap below.
- High-risk actions require explicit confirmation, but they act on the currently focused window. Use with care.
- PyAutoGUI's fail-safe is on: move the mouse to a screen corner to abort automation.

## Known limitations

- Webcam gaze accuracy is limited; calibration helps but expect target-level, not pixel-level, precision.
- Voice is always listening and there is no wake word yet.
- Benchmark figures shown before any trials are recorded are placeholders, not measured results.
- The Docker image for the backend is not supported, since it needs a camera, microphone and display.

## Roadmap

- [ ] Restrict CORS and bind to localhost by default
- [ ] Wake word or push-to-talk
- [ ] Safer file deletion (move to recycle bin)
- [ ] Unit tests for the intent parser, security layer and fusion engine
- [ ] Real benchmark trial recording
- [ ] Gemini and OpenAI intent-parser providers

## Tech stack

Python, FastAPI, SQLAlchemy, MediaPipe, OpenCV, Faster-Whisper, scikit-learn, PyAutoGUI, React, Vite, Tailwind CSS.

## Author

Add your name, institution and contact details here.

## License

Add a license (for example MIT) before publishing.
