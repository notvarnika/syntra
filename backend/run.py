import os
import warnings

# Silence third-party C++ TensorFlow Lite, Protobuf, and HuggingFace Hub warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", category=FutureWarning)

import uvicorn

if __name__ == "__main__":
    print("=================================================================")
    print("  SYNTRA - AI Multimodal Human-Computer Interaction Framework")
    print("  Backend API & WebSocket Server starting on http://localhost:8000")
    print("=================================================================")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)