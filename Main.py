from __future__ import annotations

import argparse
from pathlib import Path

from project_config import (
    BACKBONE_CHOICES,
    BRAND_NAME,
    COOKIE_PATH,
    DEFAULT_BACKBONE,
    DEFAULT_DEVICE,
    DEFAULT_TOPK,
    LEGACY_CHECKPOINT_PATH,
    MAPPING_PATH,
    MODERN_CHECKPOINT_PATH,
)
from services import AtlasScopeService, prepare_local_runtime, resolve_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"{BRAND_NAME}: local image geolocation prediction with map output."
    )
    parser.add_argument("--image", help="Local image path for prediction.")
    parser.add_argument("--game-id", help="Tuxun game UUID.")
    parser.add_argument("--mode", default="solo", choices=["solo", "streak"], help="Tuxun game mode.")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE, choices=BACKBONE_CHOICES)
    parser.add_argument("--checkpoint", help="Checkpoint path.")
    parser.add_argument("--mapping", default=str(MAPPING_PATH), help="Class mapping JSON path.")
    parser.add_argument("--cookie-path", default=str(COOKIE_PATH), help="Cookie file path for Tuxun mode.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Inference device: auto / cpu / cuda / mps.")
    parser.add_argument("--topk", default=DEFAULT_TOPK, type=int, help="Number of predictions to show.")
    parser.add_argument("--use-pretrained-backbone", action="store_true")
    return parser.parse_args()


def main() -> None:
    prepare_local_runtime()
    args = parse_args()

    if not args.image and not args.game_id:
        raise RuntimeError("Provide either --image for local prediction or --game-id for Tuxun mode.")

    checkpoint_path = resolve_checkpoint(
        backbone=args.backbone,
        explicit_checkpoint=args.checkpoint,
        legacy_checkpoint=LEGACY_CHECKPOINT_PATH,
        modern_checkpoint=MODERN_CHECKPOINT_PATH,
    )

    service = AtlasScopeService(
        mapping_path=Path(args.mapping),
        checkpoint_path=checkpoint_path,
        backbone=args.backbone,
        device=args.device,
        topk=args.topk,
        use_pretrained_backbone=args.use_pretrained_backbone,
    )

    if args.image:
        bundle, json_path, map_path = service.predict_uploaded_image(args.image)
    else:
        user_id, bundle, json_path, map_path = service.predict_tuxun_game(
            game_id=args.game_id,
            mode=args.mode,
            cookie_path=Path(args.cookie_path),
        )
        print(f"Authenticated user: {user_id}")

    service.print_predictions(bundle)
    print(f"\nPrediction JSON: {json_path}")
    print(f"Prediction Map: {map_path}")


if __name__ == "__main__":
    main()
