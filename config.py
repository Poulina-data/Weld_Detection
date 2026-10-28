import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Check for .env file if it exists, read key=value lines
_env_file = BASE_DIR / ".env"
if _env_file.exists():
    with open(_env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

class Config:
    # Server settings
    HOST = os.getenv("FLASK_HOST", "127.0.0.1")
    PORT = int(os.getenv("FLASK_PORT", "5000"))
    DEBUG = os.getenv("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")

    # Directory paths
    UPLOAD_FOLDER = BASE_DIR / "uploads"
    OUTPUT_FOLDER = BASE_DIR / "outputs"
    MODELS_FOLDER = BASE_DIR / "models"
    STATIC_FOLDER = BASE_DIR / "static"
    TEMPLATES_FOLDER = BASE_DIR / "templates"

    # Upload validation settings
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", str(10 * 1024 * 1024)))  # 10 MB limit
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

    # Model configuration
    # Search priorities for default model:
    # 1. Environment variable YOLO_MODEL_PATH
    # 2. models/best.pt
    # 3. runs/detect/YOLO weld detection project/weld_yolo_training/weights/best.pt
    # 4. Fallback yolo11n.pt
    _custom_env_model = os.getenv("YOLO_MODEL_PATH")
    if _custom_env_model and Path(_custom_env_model).exists():
        DEFAULT_MODEL_PATH = str(Path(_custom_env_model).resolve())
    elif (MODELS_FOLDER / "best.pt").exists():
        DEFAULT_MODEL_PATH = str((MODELS_FOLDER / "best.pt").resolve())
    elif (BASE_DIR / "runs" / "detect" / "YOLO weld detection project" / "weld_yolo_training" / "weights" / "best.pt").exists():
        DEFAULT_MODEL_PATH = str((BASE_DIR / "runs" / "detect" / "YOLO weld detection project" / "weld_yolo_training" / "weights" / "best.pt").resolve())
    else:
        DEFAULT_MODEL_PATH = "yolo11n.pt"

    # Default inference settings
    CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
    IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
    IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))

    # Webcam inference settings
    WEBCAM_FRAME_WIDTH = int(os.getenv("WEBCAM_FRAME_WIDTH", "640"))
    WEBCAM_FPS_LIMIT = int(os.getenv("WEBCAM_FPS_LIMIT", "10"))
