"""
database.py
-----------
Purpose:
    Handles all SQLite database interaction for the AI Road Damage
    Detection System. Responsible for initializing the `road_damage.db`
    database, creating the `damage_reports` table, inserting new
    detection records, querying detection history, computing dashboard
    statistics, and deleting records.

    All functions open short-lived connections (using a context manager)
    rather than keeping a single global connection open, which is the
    safest pattern for SQLite in a multi-threaded Flask application.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import config
from logger import get_logger

logger = get_logger(__name__)


@contextmanager
def get_connection():
    """
    Context manager that yields a SQLite connection with row factory set
    to `sqlite3.Row`, so query results can be accessed like dictionaries
    (e.g. row["filename"]). The connection is always closed afterward,
    even if an exception occurs, to avoid leaking file handles.
    """
    connection = sqlite3.connect(str(config.DATABASE_PATH))
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db() -> None:
    """
    Create the `damage_reports` table if it does not already exist.

    Table schema:
        id              INTEGER PRIMARY KEY AUTOINCREMENT
        filename        TEXT    - original uploaded file name
        file_type       TEXT    - "image" or "video"
        damage_type     TEXT    - comma-separated detected damage classes
        confidence      REAL    - average confidence score of detections
        processing_time REAL    - seconds taken to run inference
        created_at      TEXT    - ISO-8601 timestamp of the detection
        output_path     TEXT    - path to the annotated output file

    This function is idempotent: it is safe to call every time the app
    starts, since `CREATE TABLE IF NOT EXISTS` will not affect an
    already-initialized database.
    """
    config.DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    create_table_sql = """
        CREATE TABLE IF NOT EXISTS damage_reports (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            filename        TEXT NOT NULL,
            file_type       TEXT NOT NULL,
            damage_type     TEXT NOT NULL,
            confidence      REAL NOT NULL,
            processing_time REAL NOT NULL,
            created_at      TEXT NOT NULL,
            output_path     TEXT NOT NULL
        );
    """
    with get_connection() as conn:
        conn.execute(create_table_sql)
    logger.info("Database initialized at %s", config.DATABASE_PATH)


def insert_report(
    filename: str,
    file_type: str,
    damage_type: str,
    confidence: float,
    processing_time: float,
    output_path: str,
) -> int:
    """
    Insert a new detection record into `damage_reports`.

    Args:
        filename: Original uploaded file name.
        file_type: Either "image" or "video".
        damage_type: Comma-separated string of detected damage classes.
        confidence: Average confidence score across all detections.
        processing_time: Time (in seconds) the detection took.
        output_path: Path to the saved, annotated output file.

    Returns:
        The auto-generated integer id of the newly inserted row.
    """
    created_at = datetime.now().isoformat(timespec="seconds")
    insert_sql = """
        INSERT INTO damage_reports
            (filename, file_type, damage_type, confidence, processing_time, created_at, output_path)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    with get_connection() as conn:
        cursor = conn.execute(
            insert_sql,
            (filename, file_type, damage_type, confidence, processing_time, created_at, output_path),
        )
        new_id = cursor.lastrowid
    logger.info("Inserted detection report id=%s filename=%s", new_id, filename)
    return new_id


def get_all_reports(search: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieve all detection reports, most recent first.

    Args:
        search: Optional search string. If provided, filters results
                where the filename or damage_type contains the search
                term (case-insensitive), powering the frontend's
                searchable history table.

    Returns:
        A list of dictionaries, one per detection report.
    """
    with get_connection() as conn:
        if search:
            like_term = f"%{search.lower()}%"
            rows = conn.execute(
                """
                SELECT * FROM damage_reports
                WHERE LOWER(filename) LIKE ? OR LOWER(damage_type) LIKE ?
                ORDER BY created_at DESC
                """,
                (like_term, like_term),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM damage_reports ORDER BY created_at DESC"
            ).fetchall()
    return [dict(row) for row in rows]


def get_report_by_id(report_id: int) -> Optional[Dict[str, Any]]:
    """
    Retrieve a single detection report by its id.

    Args:
        report_id: The primary key of the report to fetch.

    Returns:
        A dictionary representing the row, or None if not found.
    """
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM damage_reports WHERE id = ?", (report_id,)
        ).fetchone()
    return dict(row) if row else None


def delete_report(report_id: int) -> bool:
    """
    Delete a detection report by its id.

    Args:
        report_id: The primary key of the report to delete.

    Returns:
        True if a row was deleted, False if no matching row existed.
    """
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM damage_reports WHERE id = ?", (report_id,))
        deleted = cursor.rowcount > 0
    if deleted:
        logger.info("Deleted detection report id=%s", report_id)
    else:
        logger.warning("Attempted to delete non-existent report id=%s", report_id)
    return deleted


def get_statistics() -> Dict[str, Any]:
    """
    Compute aggregate statistics for the dashboard/statistics page.

    Returns:
        A dictionary containing:
            total_uploads    - total number of detection records
            image_count      - number of image detections
            video_count      - number of video detections
            damage_count     - total number of individual damage
                                mentions across all records (a record
                                with "pothole,crack" counts as 2)
            average_confidence - mean confidence across all records
            latest_detections   - the 5 most recent records
            damage_distribution - dict mapping damage_type -> count,
                                   used to feed the Chart.js pie/bar chart
    """
    with get_connection() as conn:
        all_rows = [dict(row) for row in conn.execute("SELECT * FROM damage_reports")]

    total_uploads = len(all_rows)
    image_count = sum(1 for r in all_rows if r["file_type"] == "image")
    video_count = sum(1 for r in all_rows if r["file_type"] == "video")

    damage_distribution: Dict[str, int] = {}
    damage_count = 0
    for row in all_rows:
        for damage in [d.strip() for d in row["damage_type"].split(",") if d.strip()]:
            damage_distribution[damage] = damage_distribution.get(damage, 0) + 1
            damage_count += 1

    average_confidence = (
        round(sum(r["confidence"] for r in all_rows) / total_uploads, 4)
        if total_uploads
        else 0.0
    )

    latest_detections = sorted(
        all_rows, key=lambda r: r["created_at"], reverse=True
    )[:5]

    return {
        "total_uploads": total_uploads,
        "image_count": image_count,
        "video_count": video_count,
        "damage_count": damage_count,
        "average_confidence": average_confidence,
        "latest_detections": latest_detections,
        "damage_distribution": damage_distribution,
    }
