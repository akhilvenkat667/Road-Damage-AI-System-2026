"""
app.py
------
Purpose:
    The main entry point of the AI Road Damage Detection System backend.
    Creates and configures the Flask application, enables CORS for
    frontend-backend communication, ensures all required folders and the
    SQLite database exist, registers the API blueprint from routes.py,
    registers global error handlers, and starts the development server.

    Run with:  python app.py
"""

from flask import Flask
from flask_cors import CORS

import config
import database
from logger import get_logger
from routes import api_blueprint
from utils import error_response

logger = get_logger(__name__)


def create_app() -> Flask:
    """
    Application factory that builds and configures the Flask app.

    Returns:
        A fully configured Flask application instance.
    """
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

    # Enable Cross-Origin Resource Sharing so the vanilla JS frontend can
    # call the API even if served from a different port/origin.
    CORS(app)

    # Make sure every folder the app depends on exists (models, uploads,
    # outputs, reports, logs, database) before anything else runs.
    config.ensure_directories_exist()

    # Initialize the SQLite database and damage_reports table.
    database.init_db()

    # Register all API routes.
    app.register_blueprint(api_blueprint)

    register_error_handlers(app)

    logger.info("Flask application created and configured successfully.")
    return app


def register_error_handlers(app: Flask) -> None:
    """
    Register global error handlers so the API always returns
    consistent, structured JSON error responses instead of Flask's
    default HTML error pages.
    """

    @app.errorhandler(404)
    def handle_not_found(_error):
        return error_response("The requested resource was not found.", 404)

    @app.errorhandler(405)
    def handle_method_not_allowed(_error):
        return error_response("This HTTP method is not allowed for this endpoint.", 405)

    @app.errorhandler(413)
    def handle_file_too_large(_error):
        return error_response("Uploaded file is too large. Maximum size is 200MB.", 413)

    @app.errorhandler(500)
    def handle_internal_error(_error):
        return error_response("An unexpected internal server error occurred.", 500)


# Create the app at module level so it can be imported by WSGI servers
# (e.g. `gunicorn app:app`) in addition to being run directly.
app = create_app()


if __name__ == "__main__":
    logger.info(
        "Starting AI Road Damage Detection System on http://%s:%s",
        config.FLASK_HOST,
        config.FLASK_PORT,
    )
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG)