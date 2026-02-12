"""
Lightweight computer-vision scoring placeholder for uploaded creatives.
Uses PIL and optional Tesseract OCR; safe to import even if OCR not installed.
Returns a quality score (0-100) and diagnostics without altering core flows.
"""
from dataclasses import dataclass
from typing import Dict, Any
from PIL import Image, ImageStat
import math
import io

try:
    import pytesseract
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None


@dataclass
class CreativeScore:
    quality_score: float
    diagnostics: Dict[str, Any]


class CreativeCVScorer:
    def __init__(self, target_aspect: float = 1.91 / 1.0, aspect_tolerance: float = 0.15):
        self.target_aspect = target_aspect
        self.aspect_tolerance = aspect_tolerance

    def score_image_bytes(self, img_bytes: bytes) -> CreativeScore:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = img.size
        aspect = w / max(h, 1)
        aspect_score = self._aspect_score(aspect)

        stat = ImageStat.Stat(img)
        brightness = sum(stat.mean) / 3.0
        brightness_score = self._normalize(brightness, low=60, high=200)

        text_ratio = self._estimate_text_ratio(img)
        text_score = self._normalize(1 - text_ratio, low=0.7, high=1.0)  # prefer less clutter

        base_score = (brightness_score * 0.4) + (text_score * 0.4) + (aspect_score * 0.2)
        quality_score = max(0, min(100, base_score * 100))

        diagnostics = {
            "width": w,
            "height": h,
            "aspect_ratio": round(aspect, 3),
            "brightness": round(brightness, 2),
            "text_ratio": round(text_ratio, 3),
        }
        return CreativeScore(quality_score=quality_score, diagnostics=diagnostics)

    def _aspect_score(self, aspect: float) -> float:
        delta = abs(aspect - self.target_aspect)
        if delta <= self.aspect_tolerance:
            return 1.0
        return max(0.0, 1.0 - delta)

    def _estimate_text_ratio(self, img: Image.Image) -> float:
        if pytesseract is None:
            return 0.0
        try:
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            n_boxes = len([t for t in data.get("text", []) if str(t).strip()])
            return min(1.0, n_boxes / 500.0)
        except Exception:
            return 0.0

    def _normalize(self, val: float, low: float, high: float) -> float:
        if val <= low:
            return 0.0
        if val >= high:
            return 1.0
        return (val - low) / max(high - low, 1e-6)
