"""
config.py
---------
Purpose:
    Centralized configuration management for the AI Road Damage Detection
    System. All paths, constants, and tunable settings live here so that
    every other module (app, routes, detector, database, utils, logger)
    imports its configuration from a single, consistent source instead of
    hard-coding values throughout the codebase.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Base directories
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# ---------------------------------------------------------------------------
# Folder paths (all derived from BASE_DIR so the project is portable)
# ---------------------------------------------------------------------------
MODELS_DIR = BASE_DIR / "models"
DATABASE_DIR = BASE_DIR / "database"
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"

FRONTEND_DIR = PROJECT_ROOT / "frontend"

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
CUSTOM_MODEL_PATH = MODELS_DIR / "best.pt"
FALLBACK_MODEL_PATH = MODELS_DIR / "yolov8n.pt"

MODEL_MIN_SIZE_BYTES = 1 * 1024 * 1024

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# ---------------------------------------------------------------------------
# Video speed-optimization settings
# ---------------------------------------------------------------------------
# vid_stride: Ultralytics skips frames at the decoder level. vid_stride=10
# means it only decodes + runs inference on every 10th frame. The other 9
# frames are NEVER read from disk — this is 10x faster than reading every
# frame and discarding in Python.
VIDEO_VID_STRIDE = 10

# Smaller inference image = faster. 320 is very fast; 640 is default.
VIDEO_IMG_SIZE = 320

# FP16 on GPU = ~2x faster. Auto-ignored on CPU.
USE_HALF_PRECISION = True

# Max video duration (seconds) to prevent server hangs.
VIDEO_MAX_DURATION_SECONDS = 120

DEFAULT_DAMAGE_CLASSES = ["pothole", "crack", "alligator_crack", "road_damage"]

# Class name overrides: maps raw or numeric model class names to clean, readable titles.
CLASS_NAME_OVERRIDES = {
    "0": "Pothole",
    "1": "Longitudinal Crack",
    "2": "Transverse Crack",
    "3": "Alligator Crack",
    "4": "Rutting / Surface Defect",
    "pothole": "Pothole",
    "crack": "Road Crack",
    "alligator_crack": "Alligator Crack",
    "road_damage": "Surface Damage",
}

# ---------------------------------------------------------------------------
# Severity & Road Health Index (PCI) Parameters
# ---------------------------------------------------------------------------
SEVERITY_THRESHOLDS = {
    "CRITICAL": {"min_area_pct": 3.0, "min_conf": 0.80},
    "HIGH": {"min_area_pct": 1.5, "min_conf": 0.65},
    "MODERATE": {"min_area_pct": 0.4, "min_conf": 0.45},
}

# Penalty points deducted from 100 per detected damage item
SEVERITY_PENALTIES = {
    "Critical": 25,
    "High": 15,
    "Moderate": 8,
    "Minor": 4,
}

# ---------------------------------------------------------------------------
# Database configuration
# ---------------------------------------------------------------------------
DATABASE_NAME = "road_damage.db"
DATABASE_PATH = DATABASE_DIR / DATABASE_NAME

# ---------------------------------------------------------------------------
# Upload configuration
# ---------------------------------------------------------------------------
ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov"}
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS

MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200 MB

# ---------------------------------------------------------------------------
# Flask configuration
# ---------------------------------------------------------------------------
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
LOG_FILE_PATH = LOGS_DIR / "app.log"
LOG_LEVEL = "INFO"


def ensure_directories_exist() -> None:
    """Ensure every directory required by the application exists."""
    for directory in (
        MODELS_DIR,
        DATABASE_DIR,
        UPLOADS_DIR,
        OUTPUTS_DIR,
        REPORTS_DIR,
        LOGS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
