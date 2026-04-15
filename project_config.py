from __future__ import annotations

from pathlib import Path


BRAND_NAME = "AtlasScope Studio"
BRAND_TAGLINE = "Local image geolocation explorer for your own visual prediction workflow."

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
UPLOADS_DIR = OUTPUTS_DIR / "uploads"
MAPS_DIR = OUTPUTS_DIR / "maps"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
COOKIE_PATH = ROOT / "cookie.txt"
MAPPING_PATH = MODELS_DIR / "mapping.json"
LEGACY_CHECKPOINT_PATH = MODELS_DIR / "v0.3.0.pth"
MODERN_CHECKPOINT_PATH = MODELS_DIR / "convnext_tiny_geolocator.pth"
HISTORY_PATH = OUTPUTS_DIR / "history.json"

DEFAULT_TOPK = 5
DEFAULT_BACKBONE = "legacy_mobilenet_v3"
DEFAULT_DEVICE = "auto"

BACKBONE_CHOICES = [
    "convnext_tiny",
    "efficientnet_v2_s",
    "vit_b_16",
    "legacy_mobilenet_v3",
]
