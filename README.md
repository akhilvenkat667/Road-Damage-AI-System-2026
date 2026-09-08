# AI Road Damage Detection System Using Python, YOLOv8, Flask, OpenCV and SQLite

A complete, production-structured web application that detects road damage — potholes, cracks, and other
surface defects — in images and videos using a YOLOv8 object detection model, with a Flask REST API backend,
a SQLite detection history database, and a modern, responsive vanilla HTML/CSS/JS frontend.

---

## Table of contents

1. [Project description](#project-description)
2. [Features](#features)
3. [Objectives](#objectives)
4. [Technology stack](#technology-stack)
5. [Software architecture](#software-architecture)
6. [Folder structure](#folder-structure)
7. [Supported operating systems & Python version](#supported-operating-systems--python-version)
8. [Installation](#installation)
9. [Placing the YOLO model](#placing-the-yolo-model)
10. [Running the backend](#running-the-backend)
11. [Running the frontend](#running-the-frontend)
12. [One-click Windows launch](#one-click-windows-launch)
13. [API documentation](#api-documentation)
14. [Database schema](#database-schema)
15. [Screenshots](#screenshots)
16. [Troubleshooting guide](#troubleshooting-guide)
17. [Frequently asked questions](#frequently-asked-questions)
18. [Future improvements](#future-improvements)
19. [Deployment suggestions](#deployment-suggestions)
20. [License](#license)

---

## Project description

Road authorities and civil engineers spend significant time manually surveying roads for damage. This system
automates that process: upload a photo or video captured during a road inspection, and the app runs it
through a YOLOv8 object detection model to locate and classify damage, draws bounding boxes around each
detection, records the results in a database, and presents an interactive dashboard of findings over time.

## Features

- Upload road images (JPG/JPEG/PNG) or videos (MP4/AVI/MOV) via drag-and-drop or file browser
- Real-time YOLOv8 inference with bounding boxes, class labels, and confidence scores
- Annotated output images/videos saved to disk and viewable/downloadable in the browser
- Full detection history stored in SQLite, searchable by filename or damage type
- Interactive dashboard with total uploads, damage counts, average confidence, and Chart.js visualizations
- Downloadable CSV detection reports (generated with Pandas)
- Delete individual history records
- Automatic fallback from a custom-trained model (`best.pt`) to the stock YOLOv8n model
- Modern glassmorphism UI, dark theme, responsive layout, drag-and-drop, progress bars, toasts
- Centralized configuration, structured logging, and consistent JSON error handling throughout the API

## Objectives

- Provide an end-to-end, runnable reference implementation of a computer-vision web application
- Demonstrate clean separation of concerns between model inference, data access, API routing, and UI
- Be approachable for beginners while following production-grade engineering practices

## Technology stack

| Layer        | Technology                                                             |
|--------------|-------------------------------------------------------------------------|
| Detection    | YOLOv8 (Ultralytics), OpenCV                                            |
| Backend      | Python 3.10+, Flask, Flask-CORS, Pandas, NumPy, Pillow, SQLite3          |
| Frontend     | HTML5, CSS3, vanilla JavaScript (ES6+), Chart.js (via CDN)               |
| Database     | SQLite (`road_damage.db`)                                                |
| Tooling      | pathlib, uuid, logging, werkzeug                                         |

No React, Angular, Vue, Bootstrap, Node.js, or Django are used anywhere in this project.

## Software architecture

```
                +-------------------+
                |   Frontend (JS)   |
                |  index / dashboard|
                |  / history .html  |
                +---------+---------+
                          | fetch() / XHR (JSON, multipart)
                          v
                +-------------------+
                |   Flask app.py    |
                |  (routes.py API)  |
                +----+---------+----+
                     |         |
        +------------+         +-------------+
        v                                    v
 +--------------+                    +----------------+
 |  detector.py |                    |   database.py  |
 |  (YOLOv8 +   |                    |  (SQLite CRUD) |
 |   OpenCV)    |                    +--------+-------+
 +------+-------+                             |
        |                                     v
        v                          backend/database/road_damage.db
 backend/models/best.pt
 (fallback: yolov8n.pt)
```

- **config.py** centralizes every path and constant.
- **logger.py** provides one shared, rotating-file logger used everywhere.
- **detector.py** loads the YOLO model once (singleton) and exposes `detect_image` / `detect_video`.
- **database.py** owns all SQLite access (init, insert, query, delete, statistics).
- **utils.py** holds small helpers: validation, unique filenames, JSON response helpers, CSV export.
- **routes.py** defines the Flask Blueprint with every REST endpoint.
- **app.py** is the application factory that wires everything together and starts the server.

## Folder structure

```
Road-Damage-AI-System/
├── backend/
│   ├── app.py
│   ├── detector.py
│   ├── database.py
│   ├── config.py
│   ├── routes.py
│   ├── utils.py
│   ├── logger.py
│   ├── requirements.txt
│   ├── models/
│   │   ├── best.pt            <- place your custom-trained model here (optional)
│   │   └── yolov8n.pt         <- fallback pretrained model (optional, auto-downloads if absent)
│   ├── database/
│   │   └── road_damage.db     <- created automatically on first run
│   ├── uploads/                <- raw uploaded files land here
│   ├── outputs/                <- annotated results land here
│   ├── reports/                <- generated CSV reports land here
│   └── logs/                   <- app.log (rotating) lands here
├── frontend/
│   ├── index.html
│   ├── dashboard.html
│   ├── history.html
│   ├── css/
│   │   ├── style.css
│   │   └── dashboard.css
│   ├── js/
│   │   ├── app.js
│   │   ├── dashboard.js
│   │   └── history.js
│   └── assets/
│       ├── images/
│       └── icons/
├── sample_data/
│   ├── images/
│   └── videos/
├── README.md
└── run_project.bat
```

## Supported operating systems & Python version

- **OS:** Windows 10/11, macOS 12+, Ubuntu 20.04+/Debian-based Linux
- **Python:** 3.10 or 3.11 recommended (Ultralytics also supports 3.9–3.12)

## Installation

### 1. Clone or copy the project

Copy the entire `Road-Damage-AI-System` folder to your machine.

### 2. Create a virtual environment

**Windows (PowerShell):**
```powershell
cd Road-Damage-AI-System
python -m venv venv
venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
cd Road-Damage-AI-System
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r backend/requirements.txt
```

> First install may take several minutes since Ultralytics and OpenCV are large packages.

## Placing the YOLO model

- Put a custom-trained road-damage model at `backend/models/best.pt` for best accuracy. This is used
  automatically if present.
- Otherwise, place the general-purpose `yolov8n.pt` at `backend/models/yolov8n.pt`. If neither file
  exists, Ultralytics will attempt to auto-download `yolov8n.pt` the first time detection runs (requires
  internet access). Note that the stock `yolov8n.pt` is trained on the COCO dataset and will **not**
  recognize potholes/cracks by name — for real road-damage classification, train or source a model on a
  pothole/crack dataset (e.g. RDD2022 / RDD2020) and save it as `best.pt`.

## Running the backend

```bash
cd backend
python app.py
```

The server starts at `http://127.0.0.1:5000/`. On first run it automatically creates the `database/`,
`uploads/`, `outputs/`, `reports/`, and `logs/` folders and initializes `road_damage.db`.

## Running the frontend

The Flask backend serves the frontend directly — no separate server is needed. Once `python app.py` is
running, simply open:

```
http://127.0.0.1:5000/
```

in your browser. This loads `frontend/index.html`; the navbar links to the dashboard and history pages.

## One-click Windows launch

Double-click `run_project.bat` (or run it from PowerShell/CMD). It will:

1. Create a virtual environment if one doesn't exist
2. Activate it
3. Install/update dependencies from `backend/requirements.txt`
4. Launch the Flask backend in its own window
5. Open the app in your default browser

## API documentation

Base URL: `http://127.0.0.1:5000`

### `GET /`
Serves the frontend's `index.html`.

### `POST /upload-image`
Upload a single image and run detection.

**Request:** `multipart/form-data`, field name `file` (jpg/jpeg/png)

**Response `200`:**
```json
{
  "success": true,
  "message": "Image processed successfully.",
  "data": {
    "id": 14,
    "filename": "road_survey_01.jpg",
    "damage_types": ["pothole", "crack"],
    "detection_count": 3,
    "average_confidence": 0.812,
    "processing_time": 0.734,
    "output_url": "/outputs/annotated_3f2a1c9e4b7a.jpg"
  }
}
```

### `POST /upload-video`
Upload a single video and run frame-by-frame detection.

**Request:** `multipart/form-data`, field name `file` (mp4/avi/mov)

**Response `200`:** same shape as `/upload-image`, with `output_url` pointing to an annotated `.mp4`.

### `GET /history?search=<term>`
Returns detection history, optionally filtered by filename/damage type.

**Response `200`:**
```json
{
  "success": true,
  "message": "Success",
  "data": {
    "count": 2,
    "reports": [
      {
        "id": 14,
        "filename": "road_survey_01.jpg",
        "file_type": "image",
        "damage_type": "pothole,crack",
        "confidence": 0.812,
        "processing_time": 0.734,
        "created_at": "2026-08-01T10:22:05",
        "output_path": "backend/outputs/annotated_3f2a1c9e4b7a.jpg"
      }
    ]
  }
}
```

### `DELETE /history/<id>`
Deletes a single detection record.

**Response `200`:** `{ "success": true, "message": "Report deleted successfully.", "data": { "id": 14 } }`
**Response `404`:** returned if the id does not exist.

### `GET /statistics`
Returns dashboard aggregate statistics (`total_uploads`, `image_count`, `video_count`, `damage_count`,
`average_confidence`, `latest_detections`, `damage_distribution`).

### `GET /download-report`
Generates and streams a CSV file of the full detection history as a downloadable attachment.

All endpoints return the standardized envelope `{ "success": bool, "message": str, "data": any }` and use
proper HTTP status codes (`400` validation errors, `404` not found, `413` file too large, `500` server
errors).

## Database schema

Table `damage_reports` inside `backend/database/road_damage.db`:

| Column           | Type    | Description                                      |
|------------------|---------|---------------------------------------------------|
| id               | INTEGER | Primary key, auto-increment                        |
| filename         | TEXT    | Original uploaded filename                          |
| file_type        | TEXT    | `"image"` or `"video"`                             |
| damage_type      | TEXT    | Comma-separated detected damage classes            |
| confidence       | REAL    | Average confidence score across detections          |
| processing_time  | REAL    | Seconds taken to process the file                   |
| created_at       | TEXT    | ISO-8601 timestamp                                  |
| output_path      | TEXT    | Path to the saved annotated output file             |

## Screenshots

> _Add screenshots of the upload page, dashboard, and history page here, e.g._
> `![Upload page](frontend/assets/images/screenshot-upload.png)`
> `![Dashboard](frontend/assets/images/screenshot-dashboard.png)`
> `![History](frontend/assets/images/screenshot-history.png)`

## Troubleshooting guide

**`ModuleNotFoundError: No module named 'flask'` (or ultralytics/cv2/etc.)**
Your virtual environment isn't activated, or dependencies weren't installed. Run
`venv\Scripts\activate` (Windows) or `source venv/bin/activate` (macOS/Linux), then
`pip install -r backend/requirements.txt`.

**`python` / `python3` not recognized**
Python isn't installed or isn't on your PATH. Install Python 3.10+ from python.org and make sure
"Add Python to PATH" is checked during Windows installation.

**Virtual environment activation blocked on Windows (PowerShell)**
Run PowerShell as Administrator and execute:
`Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`, then retry activation.

**`ultralytics` install fails / takes forever**
Ensure you have a stable internet connection and at least ~2GB free disk space. On low-resource
machines, install a CPU-only PyTorch wheel first: `pip install torch --index-url
https://download.pytorch.org/whl/cpu`, then re-run `pip install -r backend/requirements.txt`.

**`cv2.error` when processing a video**
The video codec may be unsupported. Convert the file to standard H.264 MP4, or try a different sample
video from `sample_data/videos/`.

**Flask server won't start / `Address already in use`**
Port 5000 is occupied by another process. Either stop that process, or change `FLASK_PORT` in
`backend/config.py`.

**`sqlite3.OperationalError: unable to open database file`**
The `backend/database/` folder is missing or not writable. Ensure the app has write permissions to the
project folder; the folder is created automatically on startup via `config.ensure_directories_exist()`.

**YOLO model fails to load / `RuntimeError: Could not load YOLO model`**
Confirm `backend/models/best.pt` or `backend/models/yolov8n.pt` exists and is a valid `.pt` file, or
ensure you have internet access so Ultralytics can auto-download `yolov8n.pt`.

**CORS errors in the browser console**
Make sure you are opening the app via the Flask server URL (`http://127.0.0.1:5000/`) rather than opening
`index.html` directly as a `file://` URL, and confirm `flask-cors` is installed.

**Frontend loads but "Run detection" does nothing**
Open the browser console (F12) for errors. Confirm the backend is running and reachable at
`http://127.0.0.1:5000`, and that no ad-blocker/extension is blocking `fetch`/`XHR` requests.

**Uploaded file is rejected as "unsupported extension"**
Only `.jpg/.jpeg/.png` (images) and `.mp4/.avi/.mov` (videos) are accepted. Rename or convert your file.

**`413 Request Entity Too Large`**
The file exceeds the 200MB limit set by `MAX_CONTENT_LENGTH` in `backend/config.py`. Either compress the
file or raise the limit in `config.py`.

## Frequently asked questions

**Q: Do I need a GPU?**
No — the app runs on CPU by default via Ultralytics, though inference (especially on video) is faster with
a CUDA-capable GPU and the GPU build of PyTorch installed.

**Q: Can I use my own trained model?**
Yes. Export your trained weights as `best.pt` and place them in `backend/models/`. The app will use it
automatically on next restart.

**Q: Where are my uploaded and output files stored?**
Raw uploads: `backend/uploads/`. Annotated results: `backend/outputs/`. CSV reports: `backend/reports/`.

**Q: How do I reset all history?**
Stop the server, delete `backend/database/road_damage.db`, and restart — a fresh, empty database will be
created automatically.

## Future improvements

- User authentication and per-user detection history
- Real-time webcam/live-stream detection
- Model retraining pipeline and dataset management UI
- Map-based geotagging of detections (if GPS EXIF/metadata is available)
- Pagination and advanced filtering on the history page
- Docker Compose setup for one-command deployment

## Deployment suggestions

- Run behind a production WSGI server such as **gunicorn** (Linux/macOS) or **waitress** (Windows) instead
  of Flask's built-in development server.
- Serve the `frontend/` folder via a reverse proxy (e.g. Nginx) in front of the Flask API for better static
  asset performance.
- Store `uploads/`, `outputs/`, and the SQLite database on persistent volumes if deploying in a container.
- Set `FLASK_DEBUG = False` in `backend/config.py` before deploying to production.

## License

This project is provided as an educational/portfolio reference implementation. You are free to use, modify,
and extend it for personal, academic, or commercial purposes.
