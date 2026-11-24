"""Optional image fitting around the unchanged pretrained attribution core."""
from dataclasses import dataclass
import hashlib
import io

from PIL import Image, JpegImagePlugin, UnidentifiedImageError

MAX_PIXELS = 16_777_216
MAX_SIDE = 8192
MODES = ("native", "fit")


@dataclass
class PreparedImage:
    original: bytes
    encoded: bytes
    container: str
    jpeg_sampling: int | None
    preprocessing: dict | None


def prepare_image(data, mode="native"):
    if mode not in MODES:
        raise ValueError("invalid_configuration")
    try:
        with Image.open(io.BytesIO(data)) as image:
            container = image.format
            if container not in ("PNG", "JPEG"):
                raise ValueError("unsupported_format")
            width, height = image.size
            if mode == "native" and (width, height) != (512, 512):
                raise ValueError("wrong_dimensions")
            if width * height > MAX_PIXELS or max(width, height) > MAX_SIDE:
                raise ValueError("too_many_pixels")
            sampling = JpegImagePlugin.get_sampling(image) if container == "JPEG" else None
            image.verify()
        if mode == "native":
            return PreparedImage(data, data, container, sampling, None)
        applied = (width, height) != (512, 512)
        scale = min(512 / width, 512 / height)
        fitted = (max(1, min(512, round(width * scale))), max(1, min(512, round(height * scale))))
        left, top = (512 - fitted[0]) // 2, (512 - fitted[1]) // 2
        metadata = {"mode": "fit", "applied": applied, "source_size": [width, height],
                    "target_size": [512, 512], "fitted_size": list(fitted),
                    "padding_ltrb": [left, top, 512 - fitted[0] - left, 512 - fitted[1] - top],
                    "resampling": "LANCZOS" if applied else "none", "padding_rgb": [128, 128, 128],
                    "aspect_ratio_preserved": True, "cropping": False, "exif_rotation_applied": False,
                    "JPEG_reencoding": False, "original_bytes_preserved": not applied,
                    "accuracy_scope": "experimental_geometry" if applied else "unchanged_native_input"}
        encoded = data
        if applied:
            with Image.open(io.BytesIO(data)) as image:
                pixels = image.convert("RGB").resize(fitted, Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (512, 512), (128, 128, 128))
            canvas.paste(pixels, (left, top))
            buffer = io.BytesIO()
            canvas.save(buffer, format="PNG")
            encoded = buffer.getvalue()
        metadata["processed_bytes_sha256"] = hashlib.sha256(encoded).hexdigest()
        return PreparedImage(data, encoded, container, sampling, metadata)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("invalid_image") from error


def predict_prepared(predictor, prepared, code=11, image_id="encoded_image"):
    metadata = prepared.preprocessing
    if metadata is None or not metadata["applied"]:
        result = predictor.predict_encoded(prepared.original, code, image_id)
    else:
        from ecc_wouaf import Predictor
        version = predictor.version
        selected = version
        if version == "v3":
            selected = "v2" if prepared.container == "JPEG" and prepared.jpeg_sampling == 2 else "v1"
        # Preserve routing from the uploaded file before the lossless PNG handoff.
        fixed = Predictor(selected, predictor.device, predictor.strict)
        fixed.models = predictor.models
        result = fixed.predict_encoded(prepared.encoded, code, image_id)
        result.update({"requested_version": version,
                       "model_version": predictor.system["version"] if version == "v3" else version,
                       "model_status": "trained_final_research_system" if version == "v3" else "trained_comparison_extractor",
                       "input_sha256": hashlib.sha256(prepared.original).hexdigest(),
                       "container": prepared.container, "jpeg_sampling": prepared.jpeg_sampling,
                       "route_source": "actual_uploaded_file_header_before_resize" if version == "v3" else "explicit_fixed_extractor_comparison",
                       "validated_container_sampling": prepared.container == "PNG" or prepared.jpeg_sampling in (0, 2)})
    if metadata is not None:
        result["preprocessing"] = metadata
    return result


def predict_encoded(predictor, data, code=11, image_id="encoded_image", mode="native"):
    return predict_prepared(predictor, prepare_image(data, mode), code, image_id)
