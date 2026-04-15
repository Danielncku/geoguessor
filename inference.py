from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image, ImageOps

from Dataset import StreetViewImageDataset
from Model import build_model, load_checkpoint
from scene_heuristics import SceneHeuristics

try:
    import torch
    import torchvision.transforms as transforms
except Exception:  # pragma: no cover - validated by runtime
    torch = None
    transforms = None


def _require_runtime() -> None:
    if torch is None or transforms is None:
        raise RuntimeError(
            "PyTorch / torchvision is required for inference. "
            "Install the dependencies in requirements.txt with Python 3.10 or 3.11."
        )


@dataclass(frozen=True)
class Prediction:
    rank: int
    class_index: int
    label: str
    confidence: float
    lng: float
    lat: float


@dataclass(frozen=True)
class PredictionBundle:
    predictions: list[Prediction]
    image_size: tuple[int, int]
    source_name: str
    source_type: str
    model_name: str
    game_id: str
    round_index: int
    diagnostics: dict
    explanation: dict | None = None


class ModernGeolocator:
    def __init__(
        self,
        mapping_path: str,
        checkpoint_path: str,
        backbone: str = "convnext_tiny",
        device: str = "cpu",
        topk: int = 5,
        use_pretrained_backbone: bool = False,
    ) -> None:
        _require_runtime()
        self.mapping_path = Path(mapping_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.backbone = backbone
        self.device = self._select_device(device)
        self.topk = topk
        self.mapping = self._load_mapping(self.mapping_path)
        self.model = build_model(
            backbone=backbone,
            num_classes=len(self.mapping),
            use_pretrained_backbone=use_pretrained_backbone,
        )
        self.model = load_checkpoint(
            self.model,
            checkpoint_path=str(self.checkpoint_path),
            device=self.device,
            strict=False,
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.tta_transforms: list[Callable[[Image.Image], Image.Image]] = [
            lambda image: image,
            lambda image: ImageOps.mirror(image),
            lambda image: image.crop((0, 0, image.width, max(1, image.height - 8))),
        ]

    @staticmethod
    def _select_device(device: str) -> str:
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _load_mapping(mapping_path: Path) -> dict[int, dict]:
        raw = json.loads(mapping_path.read_text(encoding="utf-8"))
        return {int(k): v for k, v in raw.items()}

    def _prepare_variants(self, image: Image.Image) -> list[Image.Image]:
        base = StreetViewImageDataset.trim_image_bottom_blank(image.convert("RGB"))
        variants = []
        for crop in StreetViewImageDataset.enhance_methods:
            cropped = crop(base.copy())
            for tta in self.tta_transforms:
                variants.append(tta(cropped.copy()))
        return variants

    def predict(
        self,
        image: Image.Image,
        game_id: str,
        round_index: int,
        source_name: str | None = None,
        source_type: str = "image",
    ) -> PredictionBundle:
        variants = self._prepare_variants(image)
        script_hint = SceneHeuristics.analyze(image)

        with torch.inference_mode():
            logits = None
            for variant in variants:
                tensor = self.transform(variant).unsqueeze(0).to(self.device)
                current = self.model(tensor)
                logits = current if logits is None else logits + current

            probabilities = torch.softmax(logits, dim=1)[0]
            original_top1 = float(torch.max(probabilities).item())
            probabilities, heuristic_info = SceneHeuristics.reweight_probabilities(
                probabilities=probabilities,
                mapping=self.mapping,
                hint=script_hint,
                original_top1=original_top1,
            )
            top_indices = torch.topk(probabilities, k=min(self.topk, len(self.mapping))).indices.tolist()

        predictions: list[Prediction] = []
        for rank, class_index in enumerate(top_indices, start=1):
            target = self.mapping[class_index]
            predictions.append(
                Prediction(
                    rank=rank,
                    class_index=class_index,
                    label=target["name"],
                    confidence=float(probabilities[class_index].item()),
                    lng=float(target["lng"]),
                    lat=float(target["lat"]),
                )
            )

        return PredictionBundle(
            predictions=predictions,
            image_size=image.size,
            source_name=source_name or self.mapping_path.name,
            source_type=source_type,
            model_name=self.backbone,
            game_id=game_id,
            round_index=round_index,
            diagnostics={
                "original_top1_confidence": original_top1,
                "script_hint": {
                    "script": script_hint.script,
                    "confidence": script_hint.confidence,
                    "text_density": script_hint.text_density,
                }
                if script_hint
                else None,
                "heuristic_rerank": heuristic_info,
            },
        )

    def predict_file(self, image_path: str | Path) -> PredictionBundle:
        image_path = Path(image_path)
        with Image.open(image_path) as image:
            prepared = image.convert("RGB")
        return self.predict(
            image=prepared,
            game_id=image_path.stem,
            round_index=1,
            source_name=image_path.name,
            source_type="upload",
        )
