import os
import io
from PIL import Image
from config import Config


def allowed_file(filename: str) -> bool:
    """
    Check if the file has an allowed image extension.
    """
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in Config.ALLOWED_EXTENSIONS


def validate_image_file(file_storage) -> tuple[bool, str, bytes]:
    """
    Validate Flask FileStorage object:
    - Check presence
    - Check filename extension
    - Check content size against limit
    - Verify image bytes can be opened by PIL without corruption
    Returns: (is_valid: bool, error_message: str, image_bytes: bytes)
    """
    if not file_storage or not file_storage.filename:
        return False, "No file was selected for upload.", b""

    filename = file_storage.filename
    if not allowed_file(filename):
        allowed_list = ", ".join(Config.ALLOWED_EXTENSIONS).upper()
        return False, f"Invalid file format. Allowed formats are: {allowed_list}", b""

    # Read bytes safely
    try:
        image_bytes = file_storage.read()
    except Exception as e:
        return False, f"Error reading uploaded file: {str(e)}", b""

    if not image_bytes or len(image_bytes) == 0:
        return False, "Uploaded file is empty (0 bytes).", b""

    if len(image_bytes) > Config.MAX_CONTENT_LENGTH:
        max_mb = Config.MAX_CONTENT_LENGTH / (1024 * 1024)
        return False, f"File exceeds maximum upload size of {max_mb:.1f} MB.", b""

    # Validate image integrity
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.verify()
    except Exception:
        return False, "The uploaded file is corrupt or not a valid image.", b""

    return True, "", image_bytes


def validate_base64_frame(payload: dict) -> tuple[bool, str, str]:
    """
    Validate webcam JSON payload.
    Expected structure: { "image": "data:image/jpeg;base64,...", "conf": 0.25, ... }
    Returns: (is_valid: bool, error_message: str, base64_str: str)
    """
    if not isinstance(payload, dict):
        return False, "Malformed request body. Expected JSON object.", ""

    base64_str = payload.get("image")
    if not base64_str or not isinstance(base64_str, str):
        return False, "Missing or invalid 'image' parameter in frame data.", ""

    # Check reasonable frame size (e.g. 5 MB base64 string max)
    if len(base64_str) > 7 * 1024 * 1024:
        return False, "Frame data exceeds allowed payload limit.", ""

    return True, "", base64_str
