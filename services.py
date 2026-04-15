from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

from explanations import PredictionExplainer
from history import PredictionHistoryStore
from inference import ModernGeolocator, PredictionBundle
from project_config import (
    COOKIE_PATH,
    DEFAULT_DEVICE,
    DEFAULT_TOPK,
    HISTORY_PATH,
    MAPS_DIR,
    MAPPING_PATH,
    PREDICTIONS_DIR,
    UPLOADS_DIR,
)
from TuxunAgent import StreetView, StreetViewException, TuxunAgent, TuxunGame
from visualization import PredictionMapRenderer


def ensure_cookie_file(cookie_path: Path = COOKIE_PATH) -> str:
    cookie_path.parent.mkdir(parents=True, exist_ok=True)
    if not cookie_path.exists():
        cookie_path.touch()
    return cookie_path.read_text(encoding="utf-8").replace("\n", "").replace("\r", "").strip()


def resolve_checkpoint(backbone: str, explicit_checkpoint: str | Path | None, legacy_checkpoint: Path, modern_checkpoint: Path) -> Path:
    if explicit_checkpoint:
        return Path(explicit_checkpoint)
    return legacy_checkpoint if backbone == "legacy_mobilenet_v3" else modern_checkpoint


def validate_runtime_paths(mapping_path: Path, checkpoint_path: Path, backbone: str) -> None:
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
    if not checkpoint_path.exists():
        if backbone == "legacy_mobilenet_v3":
            raise FileNotFoundError(f"Legacy checkpoint not found: {checkpoint_path}")
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Modern backbones need a retrained checkpoint. "
            "Provide one explicitly or switch to legacy_mobilenet_v3."
        )


class AtlasScopeService:
    def __init__(
        self,
        mapping_path: str | Path = MAPPING_PATH,
        checkpoint_path: str | Path | None = None,
        backbone: str = "legacy_mobilenet_v3",
        device: str = DEFAULT_DEVICE,
        topk: int = DEFAULT_TOPK,
        use_pretrained_backbone: bool = False,
    ) -> None:
        self.mapping_path = Path(mapping_path)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.backbone = backbone
        self.device = device
        self.topk = topk
        self.use_pretrained_backbone = use_pretrained_backbone
        self.predictor: ModernGeolocator | None = None
        self.map_renderer = PredictionMapRenderer(MAPS_DIR)
        self.history = PredictionHistoryStore(HISTORY_PATH)
        self.explainer = PredictionExplainer()
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    def load_predictor(self) -> ModernGeolocator:
        if self.predictor is None:
            if self.checkpoint_path is None:
                raise RuntimeError("Checkpoint path must be set before loading predictor.")
            validate_runtime_paths(self.mapping_path, self.checkpoint_path, self.backbone)
            self.predictor = ModernGeolocator(
                mapping_path=str(self.mapping_path),
                checkpoint_path=str(self.checkpoint_path),
                backbone=self.backbone,
                device=self.device,
                topk=self.topk,
                use_pretrained_backbone=self.use_pretrained_backbone,
            )
        return self.predictor

    def predict_uploaded_image(self, image_path: str | Path) -> tuple[PredictionBundle, Path, Path]:
        predictor = self.load_predictor()
        image_path = Path(image_path)
        bundle = self.attach_explanation(predictor.predict_file(image_path))
        json_path = self.write_prediction_json(bundle)
        map_path = self.map_renderer.render(bundle)
        self.history.append(bundle=bundle, map_path=map_path, image_path=image_path)
        return bundle, json_path, map_path

    def save_upload(self, uploaded_path: str | Path) -> Path:
        uploaded_path = Path(uploaded_path)
        safe_name = f"{uuid.uuid4().hex}{uploaded_path.suffix.lower()}"
        target = UPLOADS_DIR / safe_name
        shutil.copy2(uploaded_path, target)
        return target

    def fetch_game(self, agent: TuxunAgent, game_id: str, mode: str) -> TuxunGame:
        game = agent.get(game_id, mode=mode)
        if not isinstance(game, TuxunGame):
            raise RuntimeError(f"Unable to load game: {game}")
        return game

    def fetch_street_view(self, game: TuxunGame):
        street_view = StreetView(game.pano)
        sv_type = street_view.get_type()
        if isinstance(sv_type, StreetViewException):
            raise RuntimeError(f"Unable to detect street view type: {sv_type}")
        image = street_view.get_image()
        if not hasattr(image, "convert"):
            raise RuntimeError(f"Unable to download street view image: {image}")
        return image.convert("RGB")

    def predict_tuxun_game(self, game_id: str, mode: str = "solo", cookie_path: Path = COOKIE_PATH):
        predictor = self.load_predictor()
        cookie = ensure_cookie_file(cookie_path)
        agent = TuxunAgent()
        agent.set_cookie(cookie)
        user_id = agent.get_user_id()
        if not isinstance(user_id, str):
            raise RuntimeError(
                "Cookie validation failed. Fill cookie.txt with a valid Tuxun session cookie first. "
                f"Original error: {user_id}"
            )
        game = self.fetch_game(agent, game_id=game_id, mode=mode)
        image = self.fetch_street_view(game)
        bundle = self.attach_explanation(
            predictor.predict(
            image=image,
            game_id=game.id,
            round_index=len(game.rounds),
            source_name=f"tuxun:{game.id}",
            source_type="tuxun",
            )
        )
        json_path = self.write_prediction_json(bundle)
        map_path = self.map_renderer.render(bundle)
        self.history.append(bundle=bundle, map_path=map_path, image_path=None)
        return user_id, bundle, json_path, map_path

    def attach_explanation(self, bundle: PredictionBundle) -> PredictionBundle:
        explanation = self.explainer.explain(bundle)
        payload = {
            "summary": explanation.summary,
            "rationale": explanation.rationale,
            "provider": explanation.provider,
            "used_llm": explanation.used_llm,
            "prompt_preview": explanation.prompt_preview,
        }
        return PredictionBundle(
            predictions=bundle.predictions,
            image_size=bundle.image_size,
            source_name=bundle.source_name,
            source_type=bundle.source_type,
            model_name=bundle.model_name,
            game_id=bundle.game_id,
            round_index=bundle.round_index,
            diagnostics=bundle.diagnostics,
            explanation=payload,
        )

    @staticmethod
    def print_predictions(bundle: PredictionBundle) -> None:
        print("")
        print(f"Source: {bundle.source_name}")
        print(f"Ref ID: {bundle.game_id}")
        print(f"Backbone: {bundle.model_name}")
        if bundle.explanation:
            print(f"Explanation provider: {bundle.explanation['provider']}")
        print("")
        for item in bundle.predictions:
            conf = item.confidence * 100
            conf_str = "<1%" if conf < 1 else f"{conf:.2f}%"
            print(
                f"TOP {item.rank}: {item.label}\t"
                f"Confidence {conf_str}\t"
                f"Coordinates ({item.lng:.5f}, {item.lat:.5f})"
            )

    @staticmethod
    def write_prediction_json(bundle: PredictionBundle) -> Path:
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PREDICTIONS_DIR / f"prediction-{bundle.game_id}-round-{bundle.round_index}.json"
        payload = {
            "game_id": bundle.game_id,
            "round_index": bundle.round_index,
            "model_name": bundle.model_name,
            "source_name": bundle.source_name,
            "source_type": bundle.source_type,
            "image_size": list(bundle.image_size),
            "diagnostics": bundle.diagnostics,
            "explanation": bundle.explanation,
            "predictions": [
                {
                    "rank": item.rank,
                    "class_index": item.class_index,
                    "label": item.label,
                    "confidence": item.confidence,
                    "lng": item.lng,
                    "lat": item.lat,
                }
                for item in bundle.predictions
            ],
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path.resolve()


def normalize_local_image_name(image_path: str | Path) -> str:
    return Path(image_path).stem or f"upload-{uuid.uuid4().hex[:8]}"


def prepare_local_runtime() -> None:
    os.chdir(Path(__file__).resolve().parent)
