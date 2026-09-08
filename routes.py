"""
routes.py
---------
Purpose:
    Defines all REST API endpoints for the AI Road Damage Detection
    System as a Flask Blueprint.
"""

from pathlib import Path

from flask import Blueprint, request, send_file, send_from_directory
from werkzeug.utils import secure_filename

import config
import database
import utils
from detector import get_detector
from logger import get_logger

logger = get_logger(__name__)

api_blueprint = Blueprint("api", __name__)


def _relative_output_path(output_path: Path) -> str:
    """
    Convert an absolute output_path to a path relative to PROJECT_ROOT.
    Stores e.g. 'backend/outputs/annotated_xxx.jpg' instead of the full
    machine path, so the project stays portable across machines.
    """
    try:
        return str(output_path.relative_to(config.PROJECT_ROOT))
    except ValueError:
        return output_path.name


@api_blueprint.route("/", methods=["GET"])
def serve_index():
    return send_from_directory(str(config.FRONTEND_DIR), "dashboard.html")


@api_blueprint.route("/<path:filename>", methods=["GET"])
def serve_frontend_assets(filename: str):
    return send_from_directory(str(config.FRONTEND_DIR), filename)


@api_blueprint.route("/outputs/<path:filename>", methods=["GET"])
def serve_output_file(filename: str):
    return send_from_directory(str(config.OUTPUTS_DIR), filename)


@api_blueprint.route("/upload-image", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return utils.error_response("No file part named 'file' in the request.", 400)

    uploaded_file = request.files["file"]
    if uploaded_file.filename == "":
        return utils.error_response("No file selected for upload.", 400)

    if not utils.allowed_file(uploaded_file.filename):
        return utils.error_response(
            f"Unsupported file extension. Allowed: {sorted(config.ALLOWED_IMAGE_EXTENSIONS)}", 400
        )

    if utils.get_file_type(uploaded_file.filename) != "image":
        return utils.error_response("This endpoint only accepts image files.", 400)

    try:
        original_name = secure_filename(uploaded_file.filename)
        unique_name = utils.generate_unique_filename(original_name)

        upload_path = config.UPLOADS_DIR / unique_name
        config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        uploaded_file.save(str(upload_path))

        output_path = config.OUTPUTS_DIR / f"annotated_{unique_name}"

        detector = get_detector()
        result = detector.detect_image(upload_path, output_path)

        report_id = database.insert_report(
            filename=original_name,
            file_type="image",
            damage_type=",".join(result["damage_types"]),
            confidence=result["average_confidence"],
            processing_time=result["processing_time"],
            output_path=_relative_output_path(output_path),
        )

        response_data = {
            "id": report_id,
            "filename": original_name,
            "damage_types": result["damage_types"],
            "detection_count": result["detection_count"],
            "average_confidence": result["average_confidence"],
            "processing_time": result["processing_time"],
            "output_url": f"/outputs/{output_path.name}",
        }
        return utils.success_response(response_data, "Image processed successfully.")

    except Exception as exc:
        logger.exception("Error processing image upload: %s", exc)
        return utils.error_response("Failed to process image. Check server logs for details.", 500)


@api_blueprint.route("/upload-video", methods=["POST"])
def upload_video():
    if "file" not in request.files:
        return utils.error_response("No file part named 'file' in the request.", 400)

    uploaded_file = request.files["file"]
    if uploaded_file.filename == "":
        return utils.error_response("No file selected for upload.", 400)

    if not utils.allowed_file(uploaded_file.filename):
        return utils.error_response(
            f"Unsupported file extension. Allowed: {sorted(config.ALLOWED_VIDEO_EXTENSIONS)}", 400
        )

    if utils.get_file_type(uploaded_file.filename) != "video":
        return utils.error_response("This endpoint only accepts video files.", 400)

    try:
        original_name = secure_filename(uploaded_file.filename)
        unique_name = utils.generate_unique_filename(original_name)

        upload_path = config.UPLOADS_DIR / unique_name
        config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        uploaded_file.save(str(upload_path))

        output_path = config.OUTPUTS_DIR / f"annotated_{Path(unique_name).stem}.mp4"

        detector = get_detector()
        result = detector.detect_video(upload_path, output_path)

        report_id = database.insert_report(
            filename=original_name,
            file_type="video",
            damage_type=",".join(result["damage_types"]),
            confidence=result["average_confidence"],
            processing_time=result["processing_time"],
            output_path=_relative_output_path(output_path),
        )

        response_data = {
            "id": report_id,
            "filename": original_name,
            "damage_types": result["damage_types"],
            "detection_count": result["detection_count"],
            "average_confidence": result["average_confidence"],
            "processing_time": result["processing_time"],
            "output_url": f"/outputs/{output_path.name}",
        }
        return utils.success_response(response_data, "Video processed successfully.")

    except Exception as exc:
        logger.exception("Error processing video upload: %s", exc)
        return utils.error_response("Failed to process video. Check server logs for details.", 500)


@api_blueprint.route("/history", methods=["GET"])
def get_history():
    try:
        search_term = request.args.get("search", "").strip() or None
        reports = database.get_all_reports(search=search_term)
        return utils.success_response({"reports": reports, "count": len(reports)})
    except Exception as exc:
        logger.exception("Error fetching history: %s", exc)
        return utils.error_response("Failed to fetch history. Check server logs for details.", 500)


@api_blueprint.route("/history/<int:report_id>", methods=["DELETE"])
def delete_history_item(report_id: int):
    try:
        deleted = database.delete_report(report_id)
        if not deleted:
            return utils.error_response(f"No report found with id {report_id}.", 404)
        return utils.success_response({"id": report_id}, "Report deleted successfully.")
    except Exception as exc:
        logger.exception("Error deleting report id=%s: %s", report_id, exc)
        return utils.error_response("Failed to delete report. Check server logs for details.", 500)


@api_blueprint.route("/statistics", methods=["GET"])
def get_statistics():
    try:
        stats = database.get_statistics()
        return utils.success_response(stats)
    except Exception as exc:
        logger.exception("Error computing statistics: %s", exc)
        return utils.error_response("Failed to compute statistics. Check server logs for details.", 500)


@api_blueprint.route("/download-report", methods=["GET"])
def download_report():
    try:
        reports = database.get_all_reports()
        if not reports:
            return utils.error_response("No detection history available to export.", 404)

        report_path = utils.generate_csv_report(reports)
        return send_file(
            str(report_path),
            as_attachment=True,
            download_name=report_path.name,
            mimetype="text/csv",
        )
    except Exception as exc:
        logger.exception("Error generating report: %s", exc)
        return utils.error_response("Failed to generate report. Check server logs for details.", 500)
