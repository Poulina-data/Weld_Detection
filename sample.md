# Project: YOLO Object Detection Web Application

## 1. Project Objective

Act as a senior Python developer, computer vision engineer, and full-stack web developer.

Build a complete, functional, locally hosted **YOLO Object Detection Web Application** using Python, Flask, Ultralytics YOLO, OpenCV, HTML, CSS, and JavaScript.

The application must allow users to:

1. Upload an image and detect objects in it.
2. Open their computer's webcam and perform real-time object detection.
3. View bounding boxes, object class labels, and confidence scores.
4. View detection statistics, including the number of detected objects and individual class counts.
5. Download the processed image containing detection results.

This must be a working application, not just a UI prototype or static demo.

The application must run locally on Windows using localhost. Do not require cloud hosting, external APIs, Docker, or paid services.

---

## 2. Technology Stack

Use the following technologies:

- **Programming language:** Python 3.10
- **Backend:** Flask
- **Object detection:** Ultralytics YOLO
- **Computer vision:** OpenCV
- **Image processing:** Pillow and NumPy
- **Frontend:** HTML5, CSS3, JavaScript
- **UI icons:** Lucide icons or inline SVG
- **Frontend communication:** Fetch API
- **Model inference:** Local machine inference using PyTorch and YOLO

Use a compatible stable version of each library. Avoid unnecessary dependencies and incompatible package combinations.

Do not use TensorFlow or build a CNN from scratch. Use a pretrained Ultralytics YOLO model for object detection.

---

## 3. YOLO Model Configuration

Use a pretrained lightweight YOLO detection model by default.

Default model:
`yolo11n.pt`

Keep the model path configurable so I can replace it with my own trained model later.

For example:

`runs/detect/weld_detection_project/weld_yolo_training-3/weights/best.pt`

Important requirements:

- Load the model once when the Flask application starts, rather than reloading it for every request.
- Support pretrained YOLO models and custom-trained `.pt` detection models.
- Store the selected model path in a configuration file or environment variable.
- Use CPU by default when CUDA is unavailable.
- Allow GPU inference when a compatible CUDA-enabled PyTorch installation is available.
- Display the active model name and inference device in the interface.
- Handle missing model files with a clear error message.
- If the pretrained weights are not available locally, allow Ultralytics to download them when internet access is available.
- Do not silently replace a missing custom-trained model with a pretrained model.

Use configurable inference settings:

- Confidence threshold: 0.25 by default.
- IoU threshold: 0.45 by default.
- Image inference size: 640 pixels by default.

Allow the user to adjust the confidence threshold from the interface.

---

## 4. Application Pages and Layout

Build a professional, modern computer vision dashboard.

Application name: **YOLO Vision Studio**

Design style:

- Dark theme by default.
- Background: near-black and dark charcoal.
- Accent colors: electric blue or cyan.
- Clear typography and readable labels.
- Rounded cards, subtle borders, and consistent spacing.
- Professional technical dashboard aesthetic.
- Responsive layout for desktop and tablet screens.
- Avoid excessive animations, visual clutter, and oversized elements.

### Header

Include:

- Application logo or simple computer vision icon.
- Application name: YOLO Vision Studio.
- Subtitle: Real-Time Object Detection.
- Backend status indicator.
- Active YOLO model name.
- Inference device: CPU or GPU.

### Navigation

Provide two main modes:

1. Image Detection
2. Live Webcam

Switching modes should not require reloading the entire application.

---

## 5. Feature A: Image Upload and Object Detection

Create a dedicated image detection interface.

### Upload functionality

Allow users to:

- Click an upload button to select an image.
- Drag and drop an image into the upload area.
- Preview the selected image before inference.
- Upload JPG, JPEG, PNG, and WEBP images.
- Replace the selected image with another image.
- Remove the selected image and reset the interface.

Validate file types and file sizes on the backend. Use a configurable upload limit, with a default of 10 MB.

### Detection process

When the user clicks **Run Detection**:

1. Upload the image to the Flask backend.
2. Validate the uploaded file.
3. Decode the image safely.
4. Run YOLO inference using the loaded model.
5. Extract detected bounding boxes, class IDs, class names, and confidence scores.
6. Draw bounding boxes and labels on the image.
7. Return the processed image and detection metadata to the frontend.
8. Display the processed result without refreshing the page.

Display a loading indicator while inference is running.

Prevent duplicate submissions while a request is in progress.

If no objects are detected, display a clear message rather than an error.

### Results interface

Use a two-column layout on desktop.

Left panel:
- Original image.
- Original image dimensions.
- Upload status.

Right panel:
- Annotated image with bounding boxes.
- Number of detected objects.
- Number of unique detected classes.
- Inference time in milliseconds.
- Model name.
- Detection confidence information.

Below the results, display a detection table with:

- Object number.
- Class name.
- Confidence percentage.
- Bounding box coordinates: x1, y1, x2, y2.

Display confidence values in a readable format, such as 94.25%.

### Image controls

Provide the following controls:

- Run Detection.
- Reset.
- Download Annotated Image.
- Show/Hide Original Image.
- Confidence Threshold slider.

The downloaded image must contain the actual YOLO-generated bounding boxes and labels, not a screenshot of the webpage.

---

## 6. Feature B: Live Webcam Object Detection

Implement actual webcam-based object detection, not a simulated camera preview.

### Webcam interface

Create a dedicated webcam page containing:

- Live video preview.
- Start Camera button.
- Stop Camera button.
- Start Detection button.
- Stop Detection button.
- Current camera status.
- Detection status.
- Detected object count.
- Inference FPS.
- Confidence threshold slider.

### Webcam access

Use the browser's `navigator.mediaDevices.getUserMedia()` API to request camera permission.

Requirements:

- Use the user's actual computer webcam.
- Display the browser's native camera permission prompt.
- Show a helpful message if permission is denied.
- Handle cases where no webcam is available.
- Show a clear error if the camera is being used by another application.
- Allow the user to stop the camera and release its resources.
- Stop camera tracks when the user leaves the webcam mode or closes the page.

The webcam must work on localhost in a modern browser.

### Real-time inference architecture

Use the browser webcam for video capture and send frames to the Flask backend for YOLO inference.

Recommended implementation:

1. Capture webcam frames from an HTML video element.
2. Resize frames to a configurable processing resolution, initially 640 pixels wide.
3. Encode frames as JPEG.
4. Send frames to a dedicated Flask inference endpoint.
5. Run YOLO detection on the backend.
6. Return detection metadata and annotated frame data.
7. Render the annotated frame in a canvas or image overlay.
8. Update the detection statistics in real time.

Optimize the implementation to avoid sending multiple inference requests simultaneously. Use a frame queue or a single in-flight request strategy to prevent uncontrolled request buildup.

Start with a configurable processing rate of approximately 5–10 FPS, depending on the machine's performance.

Use a separate display loop from the inference loop so that the camera preview remains responsive.

Do not send every webcam frame without throttling.

### Webcam results

Display:

- Bounding boxes around detected objects.
- Class labels.
- Confidence percentages.
- Current number of detected objects.
- Per-class object counts.
- Approximate inference FPS.
- Current confidence threshold.

The user must be able to stop inference without necessarily closing the webcam, and stop the webcam completely when finished.

If the backend becomes unavailable, display a recoverable error and stop sending inference requests.

---

## 7. Backend Architecture

Implement the Flask backend using a clean, modular structure.

Required routes:

### GET /

Serve the main web application.

### GET /api/health

Return backend health status, model availability, model name, and inference device.

### GET /api/model-info

Return the active model name and supported class names when available.

### POST /api/detect

Accept an uploaded image and return:

- Detection success status.
- Annotated image encoded as Base64 or another appropriate response format.
- Original image dimensions.
- Number of detections.
- Unique class count.
- Detection results.
- Bounding box coordinates.
- Class IDs and names.
- Confidence scores.
- Inference time.

### POST /api/detect-frame

Accept a webcam frame as image data and return:

- Annotated frame.
- Detection metadata.
- Object counts.
- Inference time.

### GET /api/stats

Return basic application statistics if needed.

Use consistent JSON response formats for API errors.

Return appropriate HTTP status codes for invalid requests, unsupported files, oversized images, and inference failures.

Never expose Python stack traces to the browser in production-style error responses.

---

## 8. Detection Visualization

Use YOLO's actual detection outputs.

For each detection:

- Draw a visible rectangular bounding box.
- Display the class name.
- Display the confidence percentage.
- Use readable text with a contrasting background.
- Ensure labels do not unnecessarily overlap.
- Keep bounding boxes aligned with the original image coordinates.

Use a consistent color scheme for each class where practical.

Ensure that bounding boxes remain correct when the image is resized for display.

If the frontend scales an image to fit its container, preserve the original inference coordinates in the detection table.

Use OpenCV or the Ultralytics plotting functionality to create the annotated image.

Do not generate fake bounding boxes or hardcode sample detections.

---

## 9. Frontend User Experience

Create a polished and functional interface.

### Required UI states

Implement the following states clearly:

- Ready.
- Waiting for image upload.
- Image selected.
- Processing.
- Detection completed.
- No objects detected.
- Camera permission requested.
- Camera connected.
- Live detection running.
- Detection stopped.
- Model unavailable.
- Backend unavailable.
- Invalid image.
- Inference failed.

Use toast notifications or inline status messages for errors and successful actions.

Disable buttons when their actions are unavailable.

Display meaningful error messages and recovery instructions.

Ensure all buttons, sliders, navigation controls, upload interactions, and download controls work.

Do not include buttons that are merely decorative.

---

## 10. Project Folder Structure

Create the following project structure:

yolo_vision_studio/
│
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
│
├── models/
│   └── .gitkeep
│
├── uploads/
│   └── .gitkeep
│
├── outputs/
│   └── .gitkeep
│
├── services/
│   ├── __init__.py
│   └── yolo_service.py
│
├── utils/
│   ├── __init__.py
│   ├── image_utils.py
│   └── validation.py
│
├── templates/
│   └── index.html
│
└── static/
    ├── css/
    │   └── style.css
    ├── js/
    │   ├── app.js
    │   ├── image_detection.js
    │   └── webcam.js
    └── images/
        └── .gitkeep

You may adjust this structure if a better design is justified, but keep backend logic, image processing, validation, and frontend JavaScript separated.

Do not put the entire application into one enormous Python file.

---

## 11. Python Implementation Requirements

### app.py

- Initialize Flask.
- Load configuration.
- Initialize the YOLO service once.
- Register the required routes.
- Serve the frontend.
- Configure logging.
- Handle exceptions cleanly.
- Run the development server on localhost.

### services/yolo_service.py

Create a reusable YOLO service responsible for:

- Loading the configured model.
- Checking model availability.
- Selecting CPU or GPU appropriately.
- Running inference.
- Applying confidence and IoU thresholds.
- Extracting detection metadata.
- Producing annotated images.
- Measuring inference time.
- Returning structured results.

Avoid duplicating YOLO inference code across endpoints.

### utils/image_utils.py

Implement functions for:

- Safe image decoding.
- Image encoding.
- Base64 conversion where appropriate.
- Image dimension extraction.
- Annotated image serialization.

### utils/validation.py

Implement validation for:

- File extensions.
- Uploaded file sizes.
- Empty files.
- Invalid image contents.
- Malformed webcam frame data.

Use secure temporary file handling and avoid trusting user-provided filenames.

---

## 12. Configuration

Provide configurable values for:

- Flask host and port.
- Debug mode.
- YOLO model path.
- Default confidence threshold.
- Default IoU threshold.
- Image inference size.
- Maximum upload size.
- Webcam processing resolution.
- Webcam inference FPS.

Use environment variables where appropriate.

The default local address must be:

http://127.0.0.1:5000

Bind the server to `127.0.0.1` by default. Do not expose the application to the local network unless explicitly configured.

Do not hardcode machine-specific absolute paths.

Support relative paths and environment-based configuration for custom model weights.

---

## 13. Dependencies and Installation

Create a complete `requirements.txt` with compatible dependencies, including:

- Flask
- ultralytics
- opencv-python
- Pillow
- NumPy

Add other dependencies only when they are genuinely needed.

Avoid installing both `opencv-python` and `opencv-python-headless` unnecessarily in the same environment.

Account for PyTorch and CUDA compatibility. Do not assume that a GPU is available.

Provide a Windows installation procedure in README.md.

The application should be runnable using a Python virtual environment.

---

## 14. Windows Startup Instructions

Create a Windows batch file named `run.bat` in the project root.

It should:

1. Change to the project directory.
2. Check whether the virtual environment exists.
3. Create the virtual environment if it does not exist.
4. Activate the virtual environment.
5. Install dependencies from requirements.txt when necessary, or provide a clearly documented first-time setup command.
6. Start the Flask application.
7. Print the local URL.

Do not reinstall every package on every application launch if it can be avoided.

Also provide manual commands in README.md.

The intended first-time setup should be similar to:

```bat
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Then open:

http://127.0.0.1:5000

Ensure the instructions match the actual project files you create.

---

## 15. Security and Performance

Implement the following safeguards:

- Validate image uploads on the server.
- Limit upload size.
- Handle corrupt and unsupported images.
- Use safe filenames or generated temporary filenames.
- Avoid arbitrary filesystem access through request parameters.
- Do not execute uploaded content.
- Do not expose local file paths or stack traces in API responses.
- Clean up temporary files when no longer needed.
- Avoid storing webcam frames permanently.
- Avoid unnecessary memory growth during long webcam sessions.
- Prevent overlapping inference requests.
- Avoid reloading model weights for every request.
- Configure logging without logging sensitive image data.

Do not introduce a database or authentication system unless necessary. This is a local computer vision tool, so keep the initial implementation simple.

---

## 16. README Documentation

Write a complete README.md that explains:

1. What the application does.
2. Features and technology stack.
3. Project folder structure.
4. Python version and prerequisites.
5. Virtual environment setup.
6. Dependency installation.
7. How to run the application.
8. How to access localhost.
9. How to use image detection.
10. How to use webcam detection.
11. How to change the confidence threshold.
12. How to use a custom-trained YOLO model.
13. How CPU and GPU inference are selected.
14. Common installation and runtime errors.
15. Troubleshooting webcam permission problems.
16. Troubleshooting missing model weights.
17. How to stop the application.

Include Windows-specific troubleshooting for Python PATH problems and virtual environment activation.

---

## 17. Testing and Validation

After implementing the application, verify the following.

### Backend tests

- The Flask server starts successfully.
- `/api/health` returns valid JSON.
- `/api/model-info` returns model information.
- Image uploads are accepted.
- Invalid file uploads are rejected.
- YOLO inference produces real detection results.
- Bounding box coordinates are valid.
- Annotated images can be decoded.
- Inference errors are handled correctly.

### Frontend tests

- The home page loads.
- Image upload and preview work.
- Drag-and-drop works.
- Detection results appear without refreshing the page.
- The confidence slider updates the inference settings.
- The annotated image downloads correctly.
- Webcam permissions are handled.
- Start and stop camera controls work.
- Start and stop detection controls work.
- Detection statistics update during webcam inference.
- The application handles backend errors without crashing.

### Acceptance criteria

The project is complete only when:

1. The server starts locally.
2. The frontend opens at `http://127.0.0.1:5000`.
3. Uploading an image triggers actual YOLO inference.
4. The application displays actual detection bounding boxes.
5. The annotated image can be downloaded.
6. The webcam opens and provides live frames.
7. Live webcam frames are processed by the backend YOLO model.
8. Real-time detections and confidence scores appear.
9. The user can stop the webcam and release camera resources.
10. Errors are handled with understandable messages.
11. All documented commands match the generated files.

Do not declare success simply because the Flask server starts or the webpage loads. Test the actual inference workflow.

---

## 18. Custom Model Compatibility

I may use a custom YOLO-trained model for a project such as welding defect detection.

Design the application so I can change the model path in configuration without rewriting the frontend or backend.

For example:

`YOLO_MODEL_PATH=runs/detect/weld_detection_project/weld_yolo_training-3/weights/best.pt`

When a custom model is configured:

- Use its own learned class names.
- Display the correct class labels.
- Display its actual detection results.
- Do not assume that it detects standard COCO objects.
- Do not hardcode the 80 COCO class names.
- Validate that the configured file exists.
- Display a clear error if the model cannot be loaded.

Use the configured model for both uploaded images and webcam frames.

---

## 19. Final Instructions to Antigravity

Work directly in the project workspace and create every required source file.

Follow this implementation order:

1. Inspect the existing workspace before creating files.
2. Create the project structure.
3. Implement configuration and YOLO service logic.
4. Implement Flask API endpoints.
5. Build the frontend and styling.
6. Implement image upload and detection.
7. Implement webcam capture and real-time detection.
8. Add error handling and validation.
9. Create requirements.txt, README.md, and run.bat.
10. Install dependencies in a project virtual environment when permitted.
11. Run the application and test its actual functionality.
12. Fix any errors found during testing.
13. Provide the exact commands to launch the completed application.

If a library installation fails, diagnose the compatibility issue rather than silently removing the required functionality.

If model weights cannot be downloaded, explain how to obtain or configure them. Do not fabricate successful inference results.

Do not leave TODO comments, placeholder API responses, fake detection data, or unfinished buttons in the final implementation.

**Final deliverable:** A complete, working YOLO Vision Studio web application that runs locally on Windows, supports image upload and real-time webcam object detection, and can later use my own trained YOLO `.pt` model.

Start implementing the application now. Do not stop after generating a plan or describing the files you intend to create.