from __future__ import annotations

from dataclasses import dataclass
from typing import Any


try:
    import torch
    import torch.nn as nn
    import torchvision.models as tv_models
except Exception:  # pragma: no cover - import is validated at runtime
    torch = None
    nn = None
    tv_models = None


def _require_torch() -> None:
    if torch is None or nn is None or tv_models is None:
        raise RuntimeError(
            "PyTorch / torchvision is required for model loading. "
            "Install the dependencies in requirements.txt with a supported Python version."
        )


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    feature_dim: int
    builder_name: str
    weights_enum: str | None = None


BACKBONE_SPECS: dict[str, BackboneSpec] = {
    "convnext_tiny": BackboneSpec(
        name="convnext_tiny",
        feature_dim=768,
        builder_name="convnext_tiny",
        weights_enum="ConvNeXt_Tiny_Weights",
    ),
    "efficientnet_v2_s": BackboneSpec(
        name="efficientnet_v2_s",
        feature_dim=1280,
        builder_name="efficientnet_v2_s",
        weights_enum="EfficientNet_V2_S_Weights",
    ),
    "vit_b_16": BackboneSpec(
        name="vit_b_16",
        feature_dim=768,
        builder_name="vit_b_16",
        weights_enum="ViT_B_16_Weights",
    ),
    "legacy_mobilenet_v3": BackboneSpec(
        name="legacy_mobilenet_v3",
        feature_dim=960,
        builder_name="mobilenet_v3_large",
        weights_enum="MobileNet_V3_Large_Weights",
    ),
}


class LegacyMobileNetV3Model(nn.Module):
    def __init__(self, num_classes: int) -> None:
        _require_torch()
        super().__init__()
        model = tv_models.mobilenet_v3_large(weights=None)
        model.classifier = nn.Sequential(
            nn.Linear(960, 1140),
            nn.Hardswish(),
            nn.Dropout(0.38),
            nn.Linear(1140, num_classes),
        )
        self.backbone = model

    def forward(self, x):
        return self.backbone(x)


class GeolocationVisionModel(nn.Module):
    """
    Modernized image classification model for geolocation prediction.

    The model is intentionally backbone-agnostic so we can swap between
    ConvNeXt / EfficientNetV2 / ViT without rewriting the inference code.
    """

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        dropout: float = 0.2,
        use_pretrained_backbone: bool = False,
    ) -> None:
        _require_torch()
        super().__init__()

        if backbone not in BACKBONE_SPECS:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. "
                f"Available backbones: {', '.join(sorted(BACKBONE_SPECS))}"
            )

        self.spec = BACKBONE_SPECS[backbone]
        weights = self._resolve_weights(use_pretrained_backbone)
        builder = getattr(tv_models, self.spec.builder_name)
        self.backbone_name = backbone
        self.backbone = builder(weights=weights)
        self.feature_dim = self.spec.feature_dim
        self.classifier = self._build_classifier(self.feature_dim, num_classes, dropout)
        self._replace_classifier(self.backbone, self.classifier)

    def _resolve_weights(self, use_pretrained_backbone: bool) -> Any | None:
        if not use_pretrained_backbone or not self.spec.weights_enum:
            return None
        weights_enum = getattr(tv_models, self.spec.weights_enum, None)
        return weights_enum.DEFAULT if weights_enum else None

    @staticmethod
    def _build_classifier(feature_dim: int, num_classes: int, dropout: float) -> nn.Sequential:
        hidden_dim = max(512, feature_dim)
        return nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _replace_classifier(self, backbone: nn.Module, classifier: nn.Module) -> None:
        if hasattr(backbone, "classifier"):
            if isinstance(backbone.classifier, nn.Sequential):
                seq = list(backbone.classifier.children())
                if seq and isinstance(seq[-1], nn.Linear):
                    in_features = seq[-1].in_features
                    self.classifier = self._build_classifier(
                        in_features, classifier[-1].out_features, classifier[3].p
                    )
                    seq[-1] = self.classifier
                    backbone.classifier = nn.Sequential(*seq)
                    return
            backbone.classifier = classifier
            return

        if hasattr(backbone, "heads"):
            backbone.heads.head = classifier
            return

        if hasattr(backbone, "head"):
            backbone.head = classifier
            return

        raise ValueError(f"Unable to replace classifier for backbone '{self.backbone_name}'.")

    def forward(self, x):
        return self.backbone(x)


def build_model(
    backbone: str,
    num_classes: int,
    dropout: float = 0.2,
    use_pretrained_backbone: bool = False,
) -> nn.Module:
    if backbone == "legacy_mobilenet_v3":
        return LegacyMobileNetV3Model(num_classes=num_classes)
    return GeolocationVisionModel(
        backbone=backbone,
        num_classes=num_classes,
        dropout=dropout,
        use_pretrained_backbone=use_pretrained_backbone,
    )


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: str = "cpu",
    strict: bool = True,
) -> nn.Module:
    _require_torch()
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=strict)
    model.to(device)
    model.eval()
    return model
