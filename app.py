from __future__ import annotations

import uuid
from pathlib import Path

from flask import Flask, redirect, render_template, request, send_from_directory, url_for

from history import PredictionHistoryStore
from project_config import (
    BACKBONE_CHOICES,
    BRAND_NAME,
    BRAND_TAGLINE,
    DEFAULT_BACKBONE,
    DEFAULT_DEVICE,
    DEFAULT_TOPK,
    HISTORY_PATH,
    LEGACY_CHECKPOINT_PATH,
    MAPPING_PATH,
    MODERN_CHECKPOINT_PATH,
    OUTPUTS_DIR,
    UPLOADS_DIR,
)
from services import AtlasScopeService, prepare_local_runtime, resolve_checkpoint
from visualization import PredictionMapRenderer


prepare_local_runtime()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


def build_service(backbone: str = DEFAULT_BACKBONE) -> AtlasScopeService:
    checkpoint_path = resolve_checkpoint(
        backbone=backbone,
        explicit_checkpoint=None,
        legacy_checkpoint=LEGACY_CHECKPOINT_PATH,
        modern_checkpoint=MODERN_CHECKPOINT_PATH,
    )
    return AtlasScopeService(
        mapping_path=MAPPING_PATH,
        checkpoint_path=checkpoint_path,
        backbone=backbone,
        device=DEFAULT_DEVICE,
        topk=DEFAULT_TOPK,
    )


@app.get("/")
def index():
    history = PredictionHistoryStore(HISTORY_PATH).load()[:6]
    return render_template(
        "index.html",
        brand_name=BRAND_NAME,
        tagline=BRAND_TAGLINE,
        backbones=BACKBONE_CHOICES,
        default_backbone=DEFAULT_BACKBONE,
        history=history,
    )


@app.post("/predict")
def predict():
    upload = request.files.get("image")
    backbone = request.form.get("backbone", DEFAULT_BACKBONE)
    if upload is None or not upload.filename:
        return redirect(url_for("index"))

    suffix = Path(upload.filename).suffix.lower() or ".jpg"
    saved_path = UPLOADS_DIR / f"{uuid.uuid4().hex}{suffix}"
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    upload.save(saved_path)

    service = build_service(backbone=backbone)
    bundle, json_path, map_path = service.predict_uploaded_image(saved_path)
    renderer = PredictionMapRenderer()
    embedded_map = renderer.build_map_embed(bundle, map_id="result-map")

    return render_template(
        "result.html",
        brand_name=BRAND_NAME,
        bundle=bundle,
        predictions=bundle.predictions,
        embedded_map=embedded_map,
        image_url=url_for("output_file", path=f"uploads/{saved_path.name}"),
        json_url=url_for("output_file", path=f"predictions/{json_path.name}"),
        map_url=url_for("output_file", path=f"maps/{map_path.name}"),
    )


@app.get("/outputs/<path:path>")
def output_file(path: str):
    return send_from_directory(OUTPUTS_DIR, path)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
