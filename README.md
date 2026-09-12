
# 🛣 Road Damage AI System

Road Damage AI System is a Flask-based web application that detects and tracks road surface damage (potholes, cracks, etc.) in images and videos using YOLOv8.

It provides:

- 📸 Image Damage Detection
- 🎥 Video Damage Tracking
- 📊 Dashboard Statistics
- 📁 Detection History
- 📈 CSV Report Export
- 💻 Simple Web Interface

---

# 📚 Complete Beginner Guide To Run This Project

This guide assumes you have **zero prior knowledge**.

Follow every step carefully.

---

# 🖥 STEP 1 — Install Required Software

You must install 3 things:

---

## 1️⃣ Install Python 3.11 (NOT 3.14)

This project requires Python 3.10 or 3.11.

> ⚠️ Do NOT install Python 3.14 — it is too new and causes compatibility errors with ultralytics/YOLOv8.

### Download Python:

Go to:
[https://www.python.org/downloads/release/python-3119/](https://www.python.org/downloads/release/python-3119/)

Download:
**Windows installer (64-bit)**

### Install Python:

- Run the installer
- ✅ **Check "Add Python to PATH"** (very important!)
- Click "Install Now"

### Verify Installation

Open Command Prompt and type:

```bash
python --version
```

If installed correctly, you will see:

```
Python 3.11.x
```

If you see an error → Python is not installed correctly or not added to PATH.

---

## 2️⃣ Install Git (Optional but Recommended)

Download:
[https://git-scm.com/downloads](https://git-scm.com/downloads)

Install normally.

Verify:

```bash
git --version
```

---

## 3️⃣ Install VS Code (Recommended Editor)

Download:
[https://code.visualstudio.com/](https://code.visualstudio.com/)

Install normally and open your project folder in it.

---

# 📥 STEP 2 — Download The Project

You have 2 methods.

---

### Method 1 — Download ZIP (Easiest)

Go to GitHub project page:
[https://github.com/akhilvenkat667/Road-Damage-AI-System-2026](https://github.com/akhilvenkat667/Road-Damage-AI-System-2026)

- Click green **"Code"** button
- Click **"Download ZIP"**
- Extract ZIP file
- Open extracted folder

### Method 2 — Clone Using Git

Open Command Prompt:

```bash
git clone https://github.com/akhilvenkat667/Road-Damage-AI-System-2026.git
```

Then enter folder:

```bash
cd Road-Damage-AI-System-2026
```

---

# 📂 STEP 3 — Project Structure

```
Road-Damage-AI-System-2026/
│
├── backend/
│   ├── app.py              ← Main Flask application entry point
│   ├── config.py           ← Configuration (paths, thresholds, ports)
│   ├── detector.py         ← YOLOv8 model wrapper (detection + tracking)
│   ├── routes.py           ← API endpoints (/upload-image, /upload-video, etc.)
│   ├── database.py         ← SQLite database operations
│   ├── utils.py            ← Helper functions (file handling, CSV export)
│   ├── logger.py           ← Logging configuration
│   ├── requirements.txt    ← Python dependencies list
│   │
│   ├── models/
│   │   └── best.pt         ← YOLOv8 trained road-damage model
│   │
│   ├── database/           ← SQLite DB file (auto-created)
│   ├── uploads/            ← Uploaded files (auto-created)
│   ├── outputs/            ← Annotated results (auto-created)
│   ├── reports/            ← CSV reports (auto-created)
│   └── logs/               ← Application logs (auto-created)
│
├── frontend/
│   ├── index.html          ← Upload / scan page
│   ├── dashboard.html      ← Statistics dashboard
│   ├── history.html        ← Detection history page
│   ├── css/
│   │   └── style.css       ← Global styling
│   └── js/
│       ├── app.js          ← Upload page logic
│       ├── dashboard.js    ← Dashboard logic
│       └── history.js      ← History page logic
│
├── requirements.txt        ← Python dependencies (fallback copy)
└── README.md               ← Project overview
```

---

# 🔧 STEP 4 — Create Virtual Environment & Install Dependencies

### Open Command Prompt / PowerShell in the project root folder:

```bash
cd "C:\Users\akhil\OneDrive\Desktop\my project\Road-Damage-AI-System-2026"
```

### Create a virtual environment:

```bash
python -m venv venv
```

### Activate the virtual environment:

**Windows (PowerShell):**
```powershell
venv\Scripts\activate
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

You should see `(venv)` appear at the start of your prompt:

```
(venv) C:\Users\akhil\...>
```

### Install all required packages:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing or you get errors, install manually:

```bash
pip install ultralytics flask pandas opencv-python pillow
```

> ⚠️ This will download YOLOv8, Flask, OpenCV and other libraries. It may take a few minutes.

---

# 🧠 STEP 5 — Verify The Model File

The road-damage detection model must be at:

```
backend\models\best.pt
```

### What the model detects:

| Class ID | Damage Type |
|----------|-------------------------------|
| 0 | Longitudinal Crack |
| 1 | Transverse Crack |
| 2 | Alligator Crack |
| 3 | Pothole |
| 4 | Other (Road Damage) |

> ⚠️ If `best.pt` is missing or too small (< 1 MB), the system will fall back to `yolov8n.pt` which detects **vehicles** (cars, motorcycles, people) — NOT road damage. Make sure the correct road-damage model is in place.

### To verify the model:

Check the file size — it should be **approximately 67 MB**. If it is only a few KB or MB, it is the wrong file.

---

# ▶ STEP 6 — Run The Application

### Make sure your virtual environment is activated:

```powershell
venv\Scripts\activate
```

### Navigate to backend folder:

```bash
cd backend
```

### Start the Flask server:

```bash
python app.py
```

### Wait for this output in the terminal:

```
2026-09-10 20:41:44 | INFO | database | Database initialized successfully.
2026-09-10 20:41:44 | INFO | main___ | Flask application created and configured successfully.
2026-09-10 20:41:44 | INFO | detector | Custom model found and validated. Loading best.pt
2026-09-10 20:41:44 | INFO | detector | YOLO model loaded successfully | classes=['Longitudinal Crack', 'Transverse Crack', 'Alligator Crack', 'Pothole', 'Other']
2026-09-10 20:41:44 | INFO | main___ | Starting AI Road Damage Detection System on http://0.0.0.0:5000
* Running on http://127.0.0.1:5000
```

If you see the classes as `['person', 'bicycle', 'car', ...]` → the wrong model is loaded. Replace `best.pt`.

---

# 🌐 STEP 7 — Open The Website

Open your browser and go to:

[http://127.0.0.1:5000](http://127.0.0.1:5000)

or

[http://localhost:5000](http://localhost:5000)

You will see the Road Damage AI interface with:

- **Detect** — Upload image/video for damage detection
- **Dashboard** — View detection statistics
- **History** — Browse past detection results

---

# 📸 How To Use

### Detect Road Damage:

1. Click **"Detect"** in the navbar
2. Choose **Image** or **Video** mode
3. Drag & drop a road photo/video (or click "Browse files")
4. Click **"Run detection"**
5. Wait for processing
6. View the annotated result with:
   - Bounding boxes around road damage
   - Damage class labels (Pothole, Crack, etc.)
   - Confidence scores
   - Tracking IDs (for videos)
7. Click **"Download annotated file"** to save the result

### Supported File Types:

| Type | Formats | Max Size |
|------|---------------------|----------|
| Images | JPG, JPEG, PNG | 200 MB |
| Videos | MP4, AVI, MOV | 200 MB |

---

# 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|------------------------|-----------------------------------|
| GET | `/` | Serve the detect page |
| POST | `/upload-image` | Upload & detect image |
| POST | `/upload-video` | Upload & detect video |
| GET | `/history` | Get all detection history |
| DELETE | `/history/<id>` | Delete a detection record |
| GET | `/statistics` | Get dashboard statistics |
| GET | `/download-report` | Download CSV report |

### API Testing (Optional)

You can test the API using Postman or Thunder Client:

**POST** [http://127.0.0.1:5000/upload-image](http://127.0.0.1:5000/upload-image)

- Body type: `form-data`
- Key: `file`
- Value: select an image file

Response:

```json
{
  "success": true,
  "message": "Image processed successfully.",
  "data": {
    "id": 1,
    "filename": "road_photo.jpg",
    "damage_types": ["Pothole", "Longitudinal Crack"],
    "detection_count": 3,
    "average_confidence": 0.82,
    "processing_time": 1.245,
    "output_url": "/outputs/annotated_road_photo.jpg"
  }
}
```

---

# 🛑 To Stop The Application

In Command Prompt / PowerShell press:

```
CTRL + C
```

---

# ⚙ Technologies Used

| Technology | Purpose |
|----------------------|--------------------------------------|
| Python 3.11 | Backend programming language |
| Flask | Web framework |
| YOLOv8 (Ultralytics) | AI object detection model |
| OpenCV | Image/video processing |
| SQLite | Database for detection history |
| Pandas | CSV report generation |
| HTML / CSS / JS | Frontend interface |
| ByteTrack | Video tracking algorithm |

---

# 📊 Configuration Values

All settings are in `backend/config.py`:

| Setting | Value | Description |
|-------------------------|------------------|--------------------------------------|
| `FLASK_PORT` | 5000 | Web server port |
| `FLASK_HOST` | 0.0.0.0 | Accessible from all interfaces |
| `FLASK_DEBUG` | False | Debug mode off (security) |
| `CONFIDENCE_THRESHOLD` | 0.25 | Minimum detection confidence |
| `IOU_THRESHOLD` | 0.45 | Non-max suppression threshold |
| `MAX_CONTENT_LENGTH` | 200 MB | Max upload file size |
| `TRACKER_CONFIG` | bytetrack.yaml | Video tracking algorithm |
| `MODEL_MIN_SIZE_BYTES` | 1 MB | Min valid model file size |

---

# ❓ Common Problems & Solutions

### Problem: `python` command not recognized

**Solution:**
Python is not added to PATH.

- Reinstall Python 3.11
- ✅ Check **"Add Python to PATH"** during installation
- Restart Command Prompt

---

### Problem: `ModuleNotFoundError: No module named 'ultralytics'`

**Solution:**
Dependencies not installed.

Make sure your virtual environment is activated, then:

```bash
pip install ultralytics flask pandas opencv-python pillow
```

---

### Problem: App detects cars / motorcycles instead of road damage

**Solution:**
The wrong model (`best.pt`) is being used.

- Replace `backend/models/best.pt` with the road-damage model (67 MB)
- The correct model detects: Pothole, Longitudinal Crack, Transverse Crack, Alligator Crack, Other
- Check terminal output for class names — if you see `['person', 'car', ...]`, the model is wrong

---

### Problem: Port 5000 already in use

**Solution:**
Close other applications using port 5000.

Or change port in `backend/config.py`:

```python
FLASK_PORT = 5001
```

Then open:
[http://localhost:5001](http://localhost:5001)

---

### Problem: `ERR_CONNECTION_REFUSED` in browser

**Solution:**
The Flask server is not running.

- Make sure you activated the virtual environment: `venv\Scripts\activate`
- Make sure you are in the `backend` folder
- Run: `python app.py`
- Wait for `Running on http://127.0.0.1:5000`

---

### Problem: `ModuleNotFoundError: No module named 'cv2'`

**Solution:**

```bash
pip install opencv-python
```

---

### Problem: Python 3.14 compatibility errors

**Solution:**
Python 3.14 is too new.

- Uninstall Python 3.14
- Install Python 3.11
- Delete old `venv` folder
- Recreate: `python -m venv venv`
- Activate: `venv\Scripts\activate`
- Reinstall: `pip install -r requirements.txt`

---

### Problem: Video upload takes very long

**Solution:**
Video processing runs YOLOv8 on every frame.

- Use shorter videos (under 30 seconds)
- Lower resolution videos process faster
- The system processes frames sequentially — be patient

---

# 🌍 Deployment

This project can be deployed to:

- **Render**
- **Railway**
- **Heroku**
- **PythonAnywhere**
- **AWS EC2**
- **Google Cloud Run**

For deployment, make sure to:

1. Use a production WSGI server (gunicorn):
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

2. Set `FLASK_DEBUG = False` in `config.py`

3. Ensure `best.pt` is included in the deployment package

---

# 📝 Project Information

| Detail | Value |
|----------------------|--------------------------------------------------|
| Project Name | Road Damage AI System |
| Version | 1.0.0 |
| Author | Teki Akhil Venkat |
| Department | Computer Science (AIML) |
| GitHub | [akhilvenkat667/Road-Damage-AI-System-2026](https://github.com/akhilvenkat667/Road-Damage-AI-System-2026) |
| License | MIT |

---

Would you like me to save this as a file (README.md or a Word document) that you can include in your project?

This project is intended for educational and demonstration purposes.

Author: Teki Akhil Venkat

B.Tech – Computer Science / AIML

GitHub: https://github.com/akhilvenkat667/Road-Damage-AI-System-2026
