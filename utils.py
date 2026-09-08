"""
utils.py
--------
Purpose:
    Collection of small, reusable helper functions used across the
    backend: file extension validation, unique filename generation,
    JSON response helpers, and CSV report generation. Keeping these here
    avoids duplicating logic inside routes.py and detector.py.
"""

import uuid
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from flask import jsonify

import config
from logger import get_logger

logger = get_logger(__name__)


def allowed_file(filename: str) -> bool:
    """
    Check whether a filename has an extension permitted by the system.

    Args:
        filename: The original filename (e.g. "road.jpg").

    Returns:
        True if the extension is in config.ALLOWED_EXTENSIONS.
    """
    if "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[1].lower()
    return extension in config.ALLOWED_EXTENSIONS


def get_file_type(filename: str) -> str:
    """
    Determine whether a file is an "image" or a "video" based on its
    extension.

    Args:
        filename: The original filename.

    Returns:
        "image", "video", or "unknown".
    """
    extension = filename.rsplit(".", 1)[1].lower() if "." in filename else ""
    if extension in config.ALLOWED_IMAGE_EXTENSIONS:
        return "image"
    if extension in config.ALLOWED_VIDEO_EXTENSIONS:
        return "video"
    return "unknown"


def generate_unique_filename(original_filename: str) -> str:
    """
    Generate a collision-free filename by replacing the original base name
    with a UUID4 hex string, keeping the original extension.

    Args:
        original_filename: The filename as uploaded by the client.

    Returns:
        A unique filename, e.g. "3f2a1c9e4b7a.jpg".
    """
    extension = original_filename.rsplit(".", 1)[1].lower() if "." in original_filename else "dat"
    unique_id = uuid.uuid4().hex[:12]
    return f"{unique_id}.{extension}"


def success_response(data: Dict[str, Any], message: str = "Success", status_code: int = 200):
    """
    Build a standardized successful JSON response.

    Args:
        data: Payload dictionary to return to the client.
        message: Human-readable success message.
        status_code: HTTP status code (defaults to 200).

    Returns:
        A Flask response tuple (jsonify(...), status_code).
    """
    return jsonify({"success": True, "message": message, "data": data}), status_code


def error_response(message: str, status_code: int = 400):
    """
    Build a standardized error JSON response.

    Args:
        message: Human-readable error message.
        status_code: HTTP status code (defaults to 400).

    Returns:
        A Flask response tuple (jsonify(...), status_code).
    """
    logger.error("API error response: %s (status=%s)", message, status_code)
    return jsonify({"success": False, "message": message, "data": None}), status_code


def generate_csv_report(reports: List[Dict[str, Any]]) -> Path:
    """
    Generate a CSV report file from a list of detection report dicts
    using Pandas, and save it inside backend/reports/.

    Args:
        reports: List of dictionaries as returned by
                 database.get_all_reports().

    Returns:
        The Path to the generated CSV file.
    """
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(reports)
    report_filename = f"detection_report_{uuid.uuid4().hex[:8]}.csv"
    report_path = config.REPORTS_DIR / report_filename
    dataframe.to_csv(report_path, index=False)
    logger.info("Generated CSV report: %s", report_path)
    return report_path


def bytes_to_human_readable(num_bytes: int) -> str:
    """
    Convert a byte count into a human-readable string (e.g. "3.2 MB").

    Args:
        num_bytes: Size in bytes.

    Returns:
        Human-readable size string.
    """
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"
