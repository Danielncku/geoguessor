from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inference import PredictionBundle


class PredictionHistoryStore:
    def __init__(self, history_path: str | Path) -> None:
        self.path = Path(history_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def append(self, bundle: PredictionBundle, map_path: str | Path | None, image_path: str | Path | None) -> None:
        history = self.load()
        history.insert(
            0,
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "game_id": bundle.game_id,
                "round_index": bundle.round_index,
                "model_name": bundle.model_name,
                "source_name": bundle.source_name,
                "image_size": list(bundle.image_size),
                "image_path": str(image_path) if image_path else None,
                "map_path": str(map_path) if map_path else None,
                "predictions": [
                    {
                        "rank": item.rank,
                        "label": item.label,
                        "confidence": item.confidence,
                        "lng": item.lng,
                        "lat": item.lat,
                    }
                    for item in bundle.predictions
                ],
                "diagnostics": bundle.diagnostics,
                "explanation": bundle.explanation,
            },
        )
        self.path.write_text(json.dumps(history[:30], ensure_ascii=False, indent=2), encoding="utf-8")
