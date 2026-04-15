from __future__ import annotations

import re
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SCRATCH_DIR = Path("outputs") / "ocr_scratch"


@dataclass(frozen=True)
class ScriptHint:
    script: str
    confidence: float
    text_density: float
    source: str


class SceneHeuristics:
    EAST_ASIA_TARGETS = {
        "\u4e2d\u56fd": 24.0,
        "\u65e5\u672c": 14.0,
        "\u97e9\u56fd": 8.0,
        "\u671d\u9c9c": 4.0,
        "\u65b0\u52a0\u5761": 7.0,
        "\u9a6c\u6765\u897f\u4e9a": 5.0,
        "\u6cf0\u56fd": 3.0,
        "\u8d8a\u5357": 3.0,
    }

    ARABIC_TARGETS = {
        "\u6c99\u7279\u963f\u62c9\u4f2f": 16.0,
        "\u57c3\u53ca": 12.0,
        "\u963f\u5c14\u53ca\u5229\u4e9a": 10.0,
        "\u6469\u6d1b\u54e5": 10.0,
        "\u7ea6\u65e6": 8.0,
        "\u963f\u62c9\u4f2f\u8054\u5408\u914b\u957f\u56fd": 8.0,
        "\u7a81\u5c3c\u65af": 8.0,
    }

    CYRILLIC_TARGETS = {
        "\u4fc4\u7f57\u65af": 14.0,
        "\u4e4c\u514b\u5170": 10.0,
        "\u4fdd\u52a0\u5229\u4e9a": 8.0,
        "\u585e\u5c14\u7ef4\u4e9a": 8.0,
        "\u54c8\u8428\u514b\u65af\u5766": 7.0,
    }

    SCRIPT_GROUPS = {
        "Japanese": EAST_ASIA_TARGETS,
        "Han": EAST_ASIA_TARGETS,
        "Hangul": EAST_ASIA_TARGETS,
        "Katakana": EAST_ASIA_TARGETS,
        "Hiragana": EAST_ASIA_TARGETS,
        "Arabic": ARABIC_TARGETS,
        "Cyrillic": CYRILLIC_TARGETS,
    }

    @classmethod
    def analyze(cls, image: Image.Image) -> ScriptHint | None:
        prepared = cls._prepare_image_for_osd(image)
        text_density = cls._estimate_text_density(prepared)
        script, confidence = cls._run_osd(prepared)
        if not script:
            return None
        return ScriptHint(
            script=script,
            confidence=confidence,
            text_density=text_density,
            source="tesseract_osd",
        )

    @classmethod
    def reweight_probabilities(
        cls,
        probabilities,
        mapping: dict[int, dict],
        hint: ScriptHint | None,
        original_top1: float | None = None,
    ):
        if hint is None:
            return probabilities, {}

        if original_top1 is not None and original_top1 >= 0.12:
            return probabilities, {"script": hint.script, "confidence": hint.confidence, "applied": False, "reason": "model_confident"}

        if hint.confidence < 0.15 and hint.text_density < 0.015:
            return probabilities, {"script": hint.script, "confidence": hint.confidence, "applied": False}

        group = cls.SCRIPT_GROUPS.get(hint.script)
        if not group:
            return probabilities, {"script": hint.script, "confidence": hint.confidence, "applied": False}

        boosted = probabilities.clone() if hasattr(probabilities, "clone") else probabilities.copy()
        applied_targets: list[str] = []
        strength = cls._script_strength_multiplier(hint)
        for idx, target in mapping.items():
            weight = group.get(target["name"])
            if weight:
                boosted[idx] = boosted[idx] * weight * strength
                applied_targets.append(target["name"])

        total = boosted.sum().item() if hasattr(boosted.sum(), "item") else float(boosted.sum())
        boosted = boosted / total
        return boosted, {
            "script": hint.script,
            "confidence": hint.confidence,
            "text_density": hint.text_density,
            "applied": bool(applied_targets),
            "target_count": len(applied_targets),
        }

    @staticmethod
    def _script_strength_multiplier(hint: ScriptHint) -> float:
        base = 1.0
        if hint.text_density > 0.03:
            base += 0.5
        if hint.confidence > 3:
            base += 0.4
        elif hint.confidence > 1:
            base += 0.2
        return base

    @staticmethod
    def _prepare_image_for_osd(image: Image.Image) -> Image.Image:
        array = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        _, thresh = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(thresh)

    @staticmethod
    def _estimate_text_density(image: Image.Image) -> float:
        arr = np.array(image)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(arr, 100, 220)
        horizontal = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 15), np.uint8))
        return float(np.count_nonzero(horizontal)) / horizontal.size

    @staticmethod
    def _run_osd(image: Image.Image) -> tuple[str | None, float]:
        SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
        temp_dir_path = SCRATCH_DIR / uuid.uuid4().hex
        temp_dir_path.mkdir(parents=True, exist_ok=True)
        image_path = temp_dir_path / "osd_input.png"
        output_base = temp_dir_path / "osd_output"
        image.save(image_path)
        result = subprocess.run(
            ["tesseract", str(image_path), str(output_base), "--psm", "0"],
            capture_output=True,
            text=True,
        )
        osd_path = output_base.with_suffix(".osd")
        if result.returncode != 0 and not osd_path.exists():
            return None, 0.0
        content = osd_path.read_text(encoding="utf-8", errors="ignore") if osd_path.exists() else result.stderr
        script_match = re.search(r"Script:\s*([A-Za-z]+)", content)
        confidence_match = re.search(r"Script confidence:\s*([0-9.]+)", content)
        script = script_match.group(1) if script_match else None
        confidence = float(confidence_match.group(1)) if confidence_match else 0.0
        return script, confidence
