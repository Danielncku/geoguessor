from __future__ import annotations

import json
from html import escape
from pathlib import Path

from inference import PredictionBundle
from project_config import BRAND_NAME, MAPS_DIR


class PredictionMapRenderer:
    def __init__(self, output_dir: str | Path = MAPS_DIR) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def render(self, bundle: PredictionBundle) -> Path:
        html = self.build_page(bundle)
        out_path = self.output_dir / f"prediction-map-{bundle.game_id}-round-{bundle.round_index}.html"
        out_path.write_text(html, encoding="utf-8")
        return out_path.resolve()

    def build_page(self, bundle: PredictionBundle) -> str:
        summary_rows = self._build_summary_rows(bundle)
        return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{escape(BRAND_NAME)} Map</title>
  {self._leaflet_head()}
  <style>
    :root {{
      --bg: #f0eee8;
      --panel: rgba(255, 250, 240, 0.94);
      --ink: #19231d;
      --accent: #bf5b32;
      --muted: rgba(25, 35, 29, 0.66);
      --border: rgba(25, 35, 29, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(191, 91, 50, 0.18), transparent 28%),
        linear-gradient(145deg, #ece7dc, var(--bg));
      font-family: "Segoe UI", "Noto Sans TC", sans-serif;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(300px, 380px) 1fr;
      min-height: 100vh;
    }}
    .panel {{
      padding: 28px;
      background: var(--panel);
      border-right: 1px solid var(--border);
      backdrop-filter: blur(14px);
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 30px;
    }}
    .eyebrow {{
      margin: 0 0 18px;
      color: var(--muted);
      font-size: 14px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .meta {{
      line-height: 1.7;
      margin-bottom: 18px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      border-bottom: 1px solid var(--border);
      padding: 10px 8px;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    #map {{
      min-height: 100vh;
    }}
    @media (max-width: 960px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .panel {{
        border-right: none;
        border-bottom: 1px solid var(--border);
      }}
      #map {{ min-height: 70vh; }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="panel">
      <p class="eyebrow">{escape(BRAND_NAME)}</p>
      <h1>Prediction Map</h1>
      <div class="meta">
        Source: <strong>{escape(bundle.source_name)}</strong><br />
        Type: <strong>{escape(bundle.source_type)}</strong><br />
        Backbone: <strong>{escape(bundle.model_name)}</strong><br />
        Ref ID: <strong>{escape(bundle.game_id)}</strong>
      </div>
      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Location</th>
            <th>Confidence</th>
          </tr>
        </thead>
        <tbody>{summary_rows}</tbody>
      </table>
    </aside>
    <main id="map"></main>
  </div>
  {self.build_map_embed(bundle, map_id="map")}
</body>
</html>
"""

    def build_map_embed(self, bundle: PredictionBundle, map_id: str = "map") -> str:
        center = self._get_center(bundle)
        markers = [self._marker_to_dict(item) for item in bundle.predictions]
        return f"""
{self._leaflet_script()}
<script>
  const center = {json.dumps({"lat": center[0], "lng": center[1]}, ensure_ascii=False)};
  const markers = {json.dumps(markers, ensure_ascii=False)};
  const map = L.map("{map_id}", {{
    worldCopyJump: true,
    zoomControl: true
  }}).setView([center.lat, center.lng], 4);

  L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap contributors"
  }}).addTo(map);

  const bounds = [];
  markers.forEach((marker, index) => {{
    const color = index === 0 ? "#bf5b32" : "#287089";
    const radius = Math.max(50000, marker.confidence * 2200);
    const latlng = [marker.lat, marker.lng];
    bounds.push(latlng);
    const popup = `
      <strong>TOP ${{marker.rank}}</strong><br/>
      ${{marker.label}}<br/>
      Confidence: ${{marker.confidence.toFixed(2)}}%<br/>
      Lat/Lng: ${{marker.lat.toFixed(5)}}, ${{marker.lng.toFixed(5)}}
    `;

    L.circleMarker(latlng, {{
      radius: index === 0 ? 10 : 8,
      color,
      weight: 2,
      fillColor: color,
      fillOpacity: 0.82
    }}).addTo(map).bindPopup(popup);

    L.circle(latlng, {{
      radius,
      color,
      weight: 1,
      fillColor: color,
      fillOpacity: 0.14
    }}).addTo(map);
  }});

  if (bounds.length > 1) {{
    map.fitBounds(bounds, {{ padding: [36, 36] }});
  }}
</script>
"""

    @staticmethod
    def _leaflet_head() -> str:
        return """
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
    crossorigin=""
  />
"""

    @staticmethod
    def _leaflet_script() -> str:
        return """
<script
  src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
  integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
  crossorigin=""
></script>
"""

    @staticmethod
    def _get_center(bundle: PredictionBundle) -> tuple[float, float]:
        top = bundle.predictions[0]
        return top.lat, top.lng

    @staticmethod
    def _marker_to_dict(item) -> dict:
        return {
            "rank": item.rank,
            "label": item.label,
            "confidence": round(item.confidence * 100, 2),
            "lat": item.lat,
            "lng": item.lng,
        }

    @staticmethod
    def _build_summary_rows(bundle: PredictionBundle) -> str:
        return "".join(
            (
                f"<tr><td>TOP {item.rank}</td>"
                f"<td>{escape(item.label)}</td>"
                f"<td>{item.confidence * 100:.2f}%</td></tr>"
            )
            for item in bundle.predictions
        )
