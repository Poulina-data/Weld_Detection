# YOLO Vision Studio: Real-Time Object & Weld Defect Inspection

A complete, high-performance, locally hosted computer vision web application built with **Python 3.10**, **Flask**, **Ultralytics YOLO**, **OpenCV**, and a responsive dark-theme dashboard.

The application supports both static image upload detection and real-time live webcam analysis using custom trained models (such as welding defect detection weights `best.pt`) or standard pretrained YOLO models.

---

## Table of Contents

1. [Features](#1-features)
2. [Technology Stack](#2-technology-stack)
3. [Project Folder Structure](#3-project-folder-structure)
4. [Prerequisites & System Requirements](#4-prerequisites--system-requirements)
5. [Installation & Setup](#5-installation--setup)
6. [Running the Application](#6-running-the-application)
7. [How to Use Image Detection](#7-how-to-use-image-detection)
8. [How to Use Live Webcam Detection](#8-how-to-use-live-webcam-detection)
9. [Configuring Model Weights & Custom Datasets](#9-configuring-model-weights--custom-datasets)
10. [Hardware Acceleration (CPU vs GPU)](#10-hardware-acceleration-cpu-vs-gpu)
11. [API Reference](#11-api-reference)
12. [Troubleshooting & FAQs](#12-troubleshooting--faqs)

---

## 1. Features

- **Custom-Trained Model Support (`best.pt`):** Pre-configured to inspect weld seams for defects (`Bad Weld`, `Good Weld`, `Defect`), with real-time pass/fail quality verdicts.
- **Image Upload & Detection:** Drag-and-drop or file browse for JPG, JPEG, PNG, and WEBP formats up to 10 MB.
- **One-Click Quick Samples:** Built-in sample chips from the dataset for instant 1-click test runs.
- **Real-Time Webcam Streaming:** Browser-based capture via `navigator.mediaDevices.getUserMedia()` with throttled single in-flight frame processing to prevent queue buildup.
- **Detailed Itemized Metrics:** Bounding box coordinates `[x1, y1, x2, y2]`, dimensions (W × H), class confidence percentage, and inference latency in milliseconds.
- **Downloadable Annotated Results:** Download processed high-resolution images with crisp bounding box annotations and labels.
- **Interactive Threshold Controls:** Live sliders to adjust confidence (5% – 95%) and IoU overlap thresholds without restarting the server.
- **Modern Cyber Dashboard:** Dark-theme glassmorphism, responsive grid layout, laser-scan animations, and non-intrusive toast notification alerts.

---

## 2. Technology Stack

- **Backend:** Python 3.10, Flask 3.x
- **Computer Vision & Inference:** Ultralytics YOLO (YOLOv8/YOLO11), PyTorch, OpenCV (`cv2`)
- **Image Processing:** Pillow (`PIL`), NumPy
- **Frontend:** Semantic HTML5, Vanilla CSS3 (Custom Design System with Glassmorphism), Modern Vanilla JavaScript (ES6 Modules)
- **Networking:** REST JSON APIs & Base64 image serialization

---

## 3. Project Folder Structure

```text
Weld_Project/
│
├── app.py                      # Flask backend entrypoint & REST API routes
├── config.py                   # Centralized application configuration
├── requirements.txt            # Python dependencies
├── README.md                   # Complete application guide
├── run.bat                     # 1-Click Windows launch script
├── .env.example                # Example environment variable template
├── .env                        # Local runtime environment file
├── .gitignore                  # Git ignore rules
│
├── models/
│   ├── best.pt                 # Active trained YOLO model weights
│   └── .gitkeep
│
├── uploads/                    # Temporary uploaded images
│   └── .gitkeep
│
├── outputs/                    # Processed / saved detection outputs
│   └── .gitkeep
│
├── services/
│   ├── __init__.py
│   └── yolo_service.py         # Reusable YOLO inference & visualization service
│
├── utils/
│   ├── __init__.py
│   ├── image_utils.py          # Safe decoding, base64 encoding, dimensions
│   └── validation.py           # Upload size, extension & integrity validation
│
├── templates/
│   └── index.html              # Single Page Application HTML5 dashboard
│
└── static/
    ├── css/
    │   └── style.css           # Premium dark theme stylesheet
    ├── js/
    │   ├── app.js              # Tab switcher, toast alerts, telemetry
    │   ├── image_detection.js  # Upload, drag-and-drop, detection table, download
    │   └── webcam.js           # Live stream capture, frame throttle, HUD overlay
    ├── images/
    │   └── .gitkeep
    └── samples/                # Bundled weld test samples
        ├── sample_defect_crack.jpg
        ├── sample_bad_weld.jpg
        └── sample_weld_joint.jpg
```

---

## 4. Prerequisites & System Requirements

- **Operating System:** Windows 10 or 11 (64-bit)
- **Python:** Python 3.10.x (Recommended: 3.10.11)
- **Web Browser:** Google Chrome, Microsoft Edge, or Mozilla Firefox
- **Camera:** Any USB or built-in webcam (for live mode)
- **Memory:** Minimum 4 GB RAM (8 GB recommended)
- **GPU (Optional):** NVIDIA GPU with CUDA 11.8+ for accelerated inference; will automatically fall back to CPU if unavailable.

---

## 5. Installation & Setup

### Automatic Setup (Windows)

Simply double-click `run.bat` in the project root folder. It will:
1. Verify Python 3.10 is installed.
2. Create a virtual environment `venv\` automatically if one does not exist.
3. Install or update all dependencies from `requirements.txt`.
4. Launch the application on `http://127.0.0.1:5000`.

### Manual Setup (Command Prompt or PowerShell)

1. Open PowerShell or Command Prompt in the project folder:
   ```powershell
   cd c:\1_PROJECT\Deep_Learning\Weld_Project
   ```

2. Create a Python virtual environment:
   ```powershell
   python -m venv venv
   ```

3. Activate the virtual environment:
   - **Command Prompt (`cmd.exe`):**
     ```cmd
     venv\Scripts\activate.bat
     ```
   - **PowerShell:**
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```

4. Upgrade `pip` and install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 6. Running the Application

1. Make sure your virtual environment is active.
2. Run `app.py`:
   ```powershell
   python app.py
   ```
3. Open your browser and navigate to:
   ```text
   http://127.0.0.1:5000
   ```
4. To stop the application, press `Ctrl + C` in the terminal window.

---

## 7. How to Use Image Detection

1. **Select an Image:**
   - Drag and drop any `.jpg`, `.png`, or `.webp` file into the left upload box, OR
   - Click **Browse** / **Select from Device**, OR
   - Click one of the **Quick Test Samples** chips (e.g. `Hot Crack Defect`, `Bad Weld Sample`).
2. **Adjust Thresholds:**
   - Use the **Confidence Threshold** slider (default: `25%`).
   - Use the **IoU Threshold** slider (default: `45%`).
3. **Run Detection:**
   - Click the blue **Run Detection** button.
   - The scanner laser animation will indicate inference in progress.
4. **Inspect Results:**
   - The right panel displays the annotated image with colored bounding boxes and labels.
   - The **Inspection Verdict Banner** displays overall status (`DEFECT DETECTED`, `GOOD WELD`, or `NO OBJECTS DETECTED`).
   - The **Detection Table** below lists individual bounding box coordinates `[x1, y1, x2, y2]`, box dimensions, and exact confidence percentages.
5. **Download or Compare:**
   - Click **Download Result** to save the full annotated image to your computer.
   - Click **Compare View** to switch back and forth between the original and annotated image.

---

## 8. How to Use Live Webcam Detection

1. Click the **Live Webcam** tab in the navigation bar.
2. Click **Start Camera**:
   - The browser will ask for camera permission. Click **Allow**.
   - Your live feed will appear with real-time resolution and camera status.
3. Click **Start Detection**:
   - The backend will begin processing frames at the configured FPS rate (default: 8 FPS).
   - Real-time bounding boxes, defect labels, and confidence overlays will render directly on the stream.
   - The **Telemetry Panel** updates live with:
     - Active detection counts
     - Per-class breakdown (e.g., `Defect: 1`, `Bad Weld: 0`, `Good Weld: 0`)
     - Inference latency (in milliseconds)
     - Measured frames per second (FPS)
4. To pause inference while keeping the camera on, click **Pause Detection**.
5. To stop the camera and release all hardware resources, click **Stop Camera** (or switch to the Image Detection tab).

---

## 9. Configuring Model Weights & Custom Datasets

The application is configured to search for model weights in the following priority order:
1. `YOLO_MODEL_PATH` in `.env` or system environment variable
2. `models/best.pt`
3. `runs/detect/YOLO weld detection project/weld_yolo_training/weights/best.pt`
4. Pretrained fallback: `yolo11n.pt`

### Changing Model in `.env`

Edit the `.env` file in the project root:
```env
# Custom model path
YOLO_MODEL_PATH=models/best.pt

# Adjust default confidence threshold
CONF_THRESHOLD=0.30
```

When changing the model, restart the Flask server. The application automatically queries `model.names` and updates all class names, colors, and table labels dynamically.

---

## 10. Hardware Acceleration (CPU vs GPU)

- **CPU Inference (Default):** Runs reliably on any Windows PC using PyTorch CPU. Typical inference latency is ~150–250 ms per frame.
- **CUDA GPU Inference:** If an NVIDIA GPU with a CUDA-enabled PyTorch build is detected (`torch.cuda.is_available() == True`), the application automatically routes tensors to GPU device `0`, achieving 10–30 ms inference times.
- The active device (`CPU` or `CUDA`) is displayed in the top header status pills.

---

## 11. API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the main Single Page Application interface |
| `GET` | `/api/health` | Health check, device, model status, and error logs |
| `GET` | `/api/model-info` | Metadata about active YOLO model and classes |
| `POST` | `/api/detect` | Multipart/Form-data or Base64 image inference |
| `POST` | `/api/detect-frame` | Throttled Base64 webcam frame inference |
| `GET` | `/api/samples` | Lists bundled test samples in `static/samples/` |
| `POST` | `/api/detect-sample` | Runs detection on bundled test sample by filename |
| `GET` | `/api/stats` | Telemetry: total detections, uptime, frames evaluated |

---

## 12. Troubleshooting & FAQs

### Q1: `Python was not found` or `Command not recognized`
**Fix:** Ensure Python 3.10 is added to Windows PATH.
1. Press `Win + R`, type `sysdm.cpl`, press Enter.
2. Go to **Advanced** > **Environment Variables**.
3. Under System Variables, select **Path** and click **Edit**.
4. Add: `C:\Users\<YourUser>\AppData\Local\Programs\Python\Python310` and `...\Python310\Scripts`.

### Q2: PowerShell says `Execution of scripts is disabled on this system`
**Fix:** Open PowerShell as Administrator and run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Q3: Webcam access fails or says "Permission Denied"
**Fix:**
1. Check your browser address bar: click the camera icon on the left of `http://127.0.0.1:5000` and select **Always Allow**.
2. Make sure Windows Settings > **Privacy & Security** > **Camera** has "Allow desktop apps to access your camera" toggled **ON**.
3. Close other apps that may be using the camera (Zoom, Teams, Skype, Windows Camera).

### Q4: Model weights file not found
**Fix:** Ensure `models/best.pt` exists. If not, copy your trained `.pt` file to `models/best.pt` or specify its path in `.env`.
"# Weld_Detection" 
