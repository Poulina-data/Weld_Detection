import base64
import io
import time
import numpy as np
import cv2
from PIL import Image, ImageOps


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Safely decode raw image bytes into an OpenCV BGR numpy array.
    Uses PIL first to handle EXIF orientation properly, then converts to OpenCV BGR.
    Falls back to cv2.imdecode if PIL fails.
    """
    if not image_bytes:
        raise ValueError("Image bytes are empty.")

    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        # Handle EXIF orientation (common with phone/webcam uploads)
        pil_img = ImageOps.exif_transpose(pil_img)

        # Convert to RGB if RGBA, grayscale, or palette
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        # Convert RGB PIL to BGR OpenCV
        rgb_arr = np.array(pil_img)
        bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        return bgr_arr
    except Exception:
        # Fallback to direct OpenCV decode
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image data into valid pixel array.")
        return img


def decode_base64_image(base64_str: str) -> np.ndarray:
    """
    Decode base64 string (with or without data:image/...;base64, header) into OpenCV BGR numpy array.
    """
    if not base64_str or not isinstance(base64_str, str):
        raise ValueError("Invalid or empty base64 string provided.")

    if "," in base64_str:
        # Strip header like data:image/jpeg;base64,
        _, base64_data = base64_str.split(",", 1)
    else:
        base64_data = base64_str

    try:
        image_bytes = base64.b64decode(base64_data)
    except Exception as e:
        raise ValueError(f"Base64 decoding failed: {e}")

    return decode_image_bytes(image_bytes)


def encode_image_to_base64(image_bgr: np.ndarray, quality: int = 85, fmt: str = "jpeg") -> str:
    """
    Encode an OpenCV BGR numpy array to a base64 data URI string.
    """
    if image_bgr is None or not isinstance(image_bgr, np.ndarray):
        raise ValueError("Valid OpenCV image array required for encoding.")

    ext = f".{fmt.lower()}"
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality] if fmt.lower() in ("jpg", "jpeg") else []
    success, buffer = cv2.imencode(ext, image_bgr, encode_params)

    if not success:
        raise RuntimeError("Failed to encode image to buffer.")

    b64_str = base64.b64encode(buffer).decode("utf-8")
    mime = "image/jpeg" if fmt.lower() in ("jpg", "jpeg") else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64_str}"


def encode_image_to_bytes(image_bgr: np.ndarray, quality: int = 90, fmt: str = "jpeg") -> bytes:
    """
    Encode an OpenCV BGR numpy array directly to raw bytes.
    """
    if image_bgr is None or not isinstance(image_bgr, np.ndarray):
        raise ValueError("Valid OpenCV image array required for encoding.")

    ext = f".{fmt.lower()}"
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality] if fmt.lower() in ("jpg", "jpeg") else []
    success, buffer = cv2.imencode(ext, image_bgr, encode_params)

    if not success:
        raise RuntimeError("Failed to encode image to buffer.")

    return buffer.tobytes()


def get_image_dimensions(image_bgr: np.ndarray) -> dict:
    """
    Extract height and width from OpenCV image array.
    """
    if image_bgr is None or not hasattr(image_bgr, "shape"):
        return {"width": 0, "height": 0}
    h, w = image_bgr.shape[:2]
    return {"width": int(w), "height": int(h)}


def resize_image_max_dim(image_bgr: np.ndarray, max_dim: int = 640) -> np.ndarray:
    """
    Resize image proportionally so that the largest dimension does not exceed max_dim.
    """
    h, w = image_bgr.shape[:2]
    if max(h, w) <= max_dim:
        return image_bgr

    if w > h:
        new_w = max_dim
        new_h = int(h * (max_dim / float(w)))
    else:
        new_h = max_dim
        new_w = int(w * (max_dim / float(h)))

    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
