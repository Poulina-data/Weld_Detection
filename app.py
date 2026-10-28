import os
import sys
import time
import logging
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge

from config import Config
from services.yolo_service import YOLOService
from utils.image_utils import (
    decode_image_bytes,
    decode_base64_image,
    encode_image_to_base64,
    get_image_dimensions,
)
from utils.validation import validate_image_file, validate_base64_frame

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("yolo_vision_studio")

# Initialize Flask application
app = Flask(
    __name__,
    template_folder=str(Config.TEMPLATES_FOLDER),
    static_folder=str(Config.STATIC_FOLDER),
)
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH

# Ensure required runtime folders exist
Config.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
Config.OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
Config.MODELS_FOLDER.mkdir(parents=True, exist_ok=True)

# In-memory application statistics & startup timestamp
SERVER_START_TIME = time.time()
APP_STATS = {
    "total_detections": 0,
    "images_processed": 0,
    "frames_processed": 0,
}

# Initialize YOLO Service once at application startup
logger.info(f"Starting YOLO Vision Studio. Initializing model from: {Config.DEFAULT_MODEL_PATH}")
yolo_service = YOLOService(model_path=Config.DEFAULT_MODEL_PATH)


@app.route("/")
def index():
    """
    Serve the main application interface.
    """
    return render_template(
        "index.html",
        model_name=yolo_service.model_name,
        device=yolo_service.device.upper(),
        is_model_loaded=yolo_service.is_loaded,
        default_conf=Config.CONF_THRESHOLD,
        default_iou=Config.IOU_THRESHOLD,
    )


@app.route("/api/health", methods=["GET"])
def health_check():
    """
    Return backend health status, model availability, and device info.
    """
    return jsonify({
        "status": "ok",
        "service": "YOLO Vision Studio",
        "model_loaded": yolo_service.is_loaded,
        "model_name": yolo_service.model_name,
        "model_path": yolo_service.model_path,
        "device": yolo_service.device.upper(),
        "device_name": yolo_service.device_name,
        "classes_count": len(yolo_service.class_names),
        "error": yolo_service.error_message,
    })


@app.route("/api/model-info", methods=["GET"])
def model_info():
    """
    Return active model details, classes, and capability metadata.
    """
    return jsonify({
        "model_name": yolo_service.model_name,
        "model_path": yolo_service.model_path,
        "device": yolo_service.device.upper(),
        "device_name": yolo_service.device_name,
        "classes": {str(k): v for k, v in yolo_service.class_names.items()},
        "total_classes": len(yolo_service.class_names),
        "is_custom_weld_model": yolo_service.is_custom_weld_model,
        "default_conf": Config.CONF_THRESHOLD,
        "default_iou": Config.IOU_THRESHOLD,
    })


@app.route("/api/detect", methods=["POST"])
def detect_image():
    """
    Endpoint for single image detection via file upload or base64.
    """
    if not yolo_service.is_loaded:
        return jsonify({
            "success": False,
            "error": yolo_service.error_message or "YOLO model is not loaded.",
        }), 503

    # Extract optional confidence and IoU thresholds
    try:
        conf = float(request.form.get("conf", Config.CONF_THRESHOLD))
        conf = max(0.01, min(1.0, conf))
    except (ValueError, TypeError):
        conf = Config.CONF_THRESHOLD

    try:
        iou = float(request.form.get("iou", Config.IOU_THRESHOLD))
        iou = max(0.01, min(1.0, iou))
    except (ValueError, TypeError):
        iou = Config.IOU_THRESHOLD

    image_bgr = None
    original_filename = "upload.jpg"

    # Handle standard multipart file upload
    if "image" in request.files:
        file = request.files["image"]
        is_valid, err_msg, img_bytes = validate_image_file(file)
        if not is_valid:
            return jsonify({"success": False, "error": err_msg}), 400
        original_filename = file.filename or "upload.jpg"
        try:
            image_bgr = decode_image_bytes(img_bytes)
        except Exception as e:
            logger.warning(f"Failed to decode uploaded image: {e}")
            return jsonify({"success": False, "error": f"Invalid image content: {str(e)}"}), 400

    # Fallback to base64 image in form or JSON
    elif request.is_json and "image" in request.json:
        base64_str = request.json.get("image")
        try:
            image_bgr = decode_base64_image(base64_str)
        except Exception as e:
            return jsonify({"success": False, "error": f"Failed to decode base64 image: {str(e)}"}), 400
    else:
        return jsonify({"success": False, "error": "No image file provided in request."}), 400

    if image_bgr is None:
        return jsonify({"success": False, "error": "Could not read or parse image data."}), 400

    # Run inference
    try:
        result = yolo_service.predict(
            image_bgr=image_bgr,
            conf_threshold=conf,
            iou_threshold=iou,
            imgsz=Config.IMAGE_SIZE,
        )
    except Exception as e:
        logger.exception("Error during YOLO inference:")
        return jsonify({"success": False, "error": f"Inference failed: {str(e)}"}), 500

    # Encode annotated image to Base64
    try:
        annotated_b64 = encode_image_to_base64(result["annotated_image"], quality=90)
    except Exception as e:
        return jsonify({"success": False, "error": f"Image encoding failed: {str(e)}"}), 500

    # Update global metrics
    APP_STATS["images_processed"] += 1
    APP_STATS["total_detections"] += result["detection_count"]

    return jsonify({
        "success": True,
        "image": annotated_b64,
        "original_filename": original_filename,
        "original_dims": result["original_dims"],
        "detection_count": result["detection_count"],
        "class_counts": result["class_counts"],
        "unique_classes_count": result["unique_classes_count"],
        "detections": result["detections"],
        "inference_time_ms": result["inference_time_ms"],
        "verdict": result["verdict"],
        "conf_used": conf,
        "iou_used": iou,
        "model_name": yolo_service.model_name,
    })


@app.route("/api/detect-frame", methods=["POST"])
def detect_frame():
    """
    Dedicated lightweight endpoint for real-time webcam frame inference.
    Throttled and optimized for low latency.
    """
    if not yolo_service.is_loaded:
        return jsonify({
            "success": False,
            "error": yolo_service.error_message or "YOLO model is not loaded.",
        }), 503

    payload = request.get_json(silent=True)
    is_valid, err_msg, base64_str = validate_base64_frame(payload)
    if not is_valid:
        return jsonify({"success": False, "error": err_msg}), 400

    try:
        conf = float(payload.get("conf", Config.CONF_THRESHOLD))
        conf = max(0.05, min(0.95, conf))
    except (ValueError, TypeError):
        conf = Config.CONF_THRESHOLD

    try:
        iou = float(payload.get("iou", Config.IOU_THRESHOLD))
        iou = max(0.05, min(0.95, iou))
    except (ValueError, TypeError):
        iou = Config.IOU_THRESHOLD

    try:
        frame_bgr = decode_base64_image(base64_str)
    except Exception as e:
        return jsonify({"success": False, "error": f"Frame decoding error: {str(e)}"}), 400

    try:
        result = yolo_service.predict(
            image_bgr=frame_bgr,
            conf_threshold=conf,
            iou_threshold=iou,
            imgsz=Config.IMAGE_SIZE,
        )
    except Exception as e:
        return jsonify({"success": False, "error": f"Frame inference error: {str(e)}"}), 500

    # Encode with lower quality for webcam stream responsiveness
    try:
        annotated_b64 = encode_image_to_base64(result["annotated_image"], quality=75)
    except Exception as e:
        return jsonify({"success": False, "error": f"Frame encoding error: {str(e)}"}), 500

    APP_STATS["frames_processed"] += 1
    APP_STATS["total_detections"] += result["detection_count"]

    return jsonify({
        "success": True,
        "image": annotated_b64,
        "detection_count": result["detection_count"],
        "class_counts": result["class_counts"],
        "detections": result["detections"],
        "inference_time_ms": result["inference_time_ms"],
        "verdict": result["verdict"],
    })


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """
    Return available sample images for immediate 1-click testing.
    """
    sample_dir = Config.STATIC_FOLDER / "samples"
    samples = []
    if sample_dir.exists():
        for f in sample_dir.glob("*.jpg"):
            name = f.stem.replace("sample_", "").replace("_", " ").title()
            samples.append({
                "id": f.name,
                "label": name,
                "url": f"/static/samples/{f.name}",
            })
    return jsonify({"samples": samples})


@app.route("/api/detect-sample", methods=["POST"])
def detect_sample():
    """
    Run detection directly on a bundled sample image.
    """
    if not yolo_service.is_loaded:
        return jsonify({"success": False, "error": "Model is not loaded."}), 503

    payload = request.get_json(silent=True) or {}
    sample_id = payload.get("sample_id")
    if not sample_id:
        return jsonify({"success": False, "error": "Sample ID required."}), 400

    # Prevent directory traversal attacks
    safe_filename = Path(sample_id).name
    sample_path = Config.STATIC_FOLDER / "samples" / safe_filename
    if not sample_path.exists():
        return jsonify({"success": False, "error": "Sample file not found."}), 404

    try:
        conf = float(payload.get("conf", Config.CONF_THRESHOLD))
    except (ValueError, TypeError):
        conf = Config.CONF_THRESHOLD

    try:
        iou = float(payload.get("iou", Config.IOU_THRESHOLD))
    except (ValueError, TypeError):
        iou = Config.IOU_THRESHOLD

    with open(sample_path, "rb") as f:
        img_bytes = f.read()

    image_bgr = decode_image_bytes(img_bytes)
    result = yolo_service.predict(
        image_bgr=image_bgr,
        conf_threshold=conf,
        iou_threshold=iou,
        imgsz=Config.IMAGE_SIZE,
    )
    annotated_b64 = encode_image_to_base64(result["annotated_image"], quality=90)
    original_b64 = encode_image_to_base64(image_bgr, quality=90)

    APP_STATS["images_processed"] += 1
    APP_STATS["total_detections"] += result["detection_count"]

    return jsonify({
        "success": True,
        "image": annotated_b64,
        "original_image": original_b64,
        "original_filename": safe_filename,
        "original_dims": result["original_dims"],
        "detection_count": result["detection_count"],
        "class_counts": result["class_counts"],
        "unique_classes_count": result["unique_classes_count"],
        "detections": result["detections"],
        "inference_time_ms": result["inference_time_ms"],
        "verdict": result["verdict"],
        "conf_used": conf,
        "model_name": yolo_service.model_name,
    })


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """
    Return runtime statistics for monitoring.
    """
    uptime = int(time.time() - SERVER_START_TIME)
    return jsonify({
        "status": "healthy",
        "uptime_seconds": uptime,
        "uptime_formatted": f"{uptime // 3600}h {(uptime % 3600) // 60}m {uptime % 60}s",
        "images_processed": APP_STATS["images_processed"],
        "frames_processed": APP_STATS["frames_processed"],
        "total_detections": APP_STATS["total_detections"],
        "active_model": yolo_service.model_name,
        "device": yolo_service.device.upper(),
    })


# Custom Error Handlers
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    max_mb = Config.MAX_CONTENT_LENGTH / (1024 * 1024)
    return jsonify({
        "success": False,
        "error": f"Uploaded file exceeds maximum size limit of {max_mb:.0f} MB.",
        "status_code": 413,
    }), 413


@app.errorhandler(404)
def handle_not_found(e):
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "error": "Requested API route was not found.", "status_code": 404}), 404
    return render_template("index.html"), 404


@app.errorhandler(500)
def handle_server_error(e):
    logger.exception("Internal Server Error occurred:")
    return jsonify({
        "success": False,
        "error": "An internal server error occurred while processing the request.",
        "status_code": 500,
    }), 500


if __name__ == "__main__":
    logger.info(f"Starting server on http://{Config.HOST}:{Config.PORT}")
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG,
    )
