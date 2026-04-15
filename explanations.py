from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import requests

from inference import PredictionBundle


@dataclass(frozen=True)
class ExplanationResult:
    summary: str
    rationale: list[str]
    provider: str
    used_llm: bool
    prompt_preview: str


class PredictionExplainer:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    def explain(self, bundle: PredictionBundle) -> ExplanationResult:
        prompt = self._build_prompt(bundle)
        if self.api_key:
            try:
                return self._explain_with_llm(bundle, prompt)
            except Exception as exc:
                fallback = self._fallback(bundle, prompt_preview=prompt[:700], extra_note=f"LLM failed: {exc}")
                return fallback
        return self._fallback(bundle, prompt_preview=prompt[:700], extra_note="LLM not configured.")

    def _explain_with_llm(self, bundle: PredictionBundle, prompt: str) -> ExplanationResult:
        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You are an explainability assistant for an image geolocation product. "
                                "Be concise, specific, and honest about uncertainty. "
                                "Return strict JSON with keys: summary (string), rationale (array of 3 short strings)."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                },
            ],
        }
        response = requests.post(
            f"{self.base_url}/responses",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=40,
        )
        response.raise_for_status()
        data = response.json()
        text = self._extract_text(data)
        parsed = json.loads(text)
        return ExplanationResult(
            summary=str(parsed["summary"]),
            rationale=[str(item) for item in parsed["rationale"][:3]],
            provider=self.model,
            used_llm=True,
            prompt_preview=prompt[:700],
        )

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        if "output_text" in data and data["output_text"]:
            return data["output_text"]
        outputs = data.get("output", [])
        chunks: list[str] = []
        for item in outputs:
            for content in item.get("content", []):
                text = content.get("text")
                if text:
                    chunks.append(text)
        if not chunks:
            raise ValueError("No output text returned by LLM provider.")
        return "\n".join(chunks)

    def _fallback(self, bundle: PredictionBundle, prompt_preview: str, extra_note: str) -> ExplanationResult:
        diagnostics = bundle.diagnostics or {}
        original_top1 = diagnostics.get("original_top1_confidence", 0.0) * 100
        script_hint = diagnostics.get("script_hint") or {}
        heuristic = diagnostics.get("heuristic_rerank") or {}
        top = bundle.predictions[0]

        summary = (
            f"Top-1 candidate is {top.label} with {top.confidence * 100:.2f}% confidence. "
            f"The raw model confidence before reranking was {original_top1:.2f}%, "
            f"so this prediction should be treated as directional rather than final."
        )

        rationale = [
            (
                f"The image was evaluated with backbone {bundle.model_name}, and the score distribution "
                f"looks {'flat' if original_top1 < 15 else 'moderately decisive'}, which affects reliability."
            ),
            (
                f"Detected script hint: {script_hint.get('script', 'none')} "
                f"with density {script_hint.get('text_density', 0):.4f}. "
                f"Heuristic rerank applied: {heuristic.get('applied', False)}."
            ),
            (
                f"Top alternatives remain geographically meaningful: "
                f"{', '.join(item.label for item in bundle.predictions[:3])}. {extra_note}"
            ),
        ]
        return ExplanationResult(
            summary=summary,
            rationale=rationale,
            provider="local-fallback",
            used_llm=False,
            prompt_preview=prompt_preview,
        )

    @staticmethod
    def _build_prompt(bundle: PredictionBundle) -> str:
        diagnostics = bundle.diagnostics or {}
        serializable = {
            "source_name": bundle.source_name,
            "source_type": bundle.source_type,
            "model_name": bundle.model_name,
            "image_size": bundle.image_size,
            "predictions": [
                {
                    "rank": item.rank,
                    "label": item.label,
                    "confidence": round(item.confidence * 100, 2),
                    "lat": round(item.lat, 5),
                    "lng": round(item.lng, 5),
                }
                for item in bundle.predictions
            ],
            "diagnostics": diagnostics,
        }
        return (
            "Explain this geolocation prediction for a product UI. "
            "Mention uncertainty, script hints, heuristic reranking, and why the top candidates are plausible.\n\n"
            + json.dumps(serializable, ensure_ascii=False, indent=2)
        )
