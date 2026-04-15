# AtlasScope Studio

AtlasScope Studio is a local image geolocation project I built as a CS student project / experimentation playground.

The basic idea is pretty simple:

1. Upload an image from your computer.
2. Run a geolocation prediction model on it.
3. Show the top candidate locations on a map.
4. Explain why the system made that prediction instead of only showing raw scores.

So this repo is not just a classifier demo. I tried to turn it into a small but usable local product with a web UI, prediction history, map output, and an explainability layer.

## What This Project Does

- Predicts likely locations from a local image
- Shows Top-K candidate places with confidence scores
- Plots predictions on an interactive map
- Keeps recent prediction history
- Adds OCR / script-based heuristics for reranking
- Generates explainability output for every prediction
- Supports optional LLM-generated reasoning for the UI
- Still keeps the older Tuxun / GeoGuessr-style integration path

## Why I Made It

I wanted to explore a geolocation-style computer vision project, but I also wanted it to feel like an actual application instead of a notebook-only experiment.

A lot of image prediction demos stop at:

> "Here are the logits. Good luck."

I wanted to go a bit further and make something that shows:

- what the model predicted
- how uncertain it is
- whether text or script hints changed the ranking
- how the result looks on a map
- how an LLM can help turn technical signals into human-readable explanations

## Main Features

### Web UI



- Upload an image from your local machine
- Pick a backbone for inference
- Preview the uploaded image
- View Top-K predictions in a clean result layout
- See prediction points on an embedded interactive map
- Read a model explanation report
- Inspect prompt preview and explanation provider info

### Explainability

For each prediction, the app can show:

- original model confidence
- OCR / script hint
- whether heuristic reranking was applied
- a natural language explanation

If `OPENAI_API_KEY` is configured, the app can send the prediction bundle through an LLM and generate a more readable explanation.

If not, it still shows a local fallback explanation, so the UI is always informative.

### CLI Mode

- Predict from a local image with `--image`
- Predict from a Tuxun game with `--game-id`
- Change backbone, checkpoint, and inference settings

## Project Structure

```text
app.py               Flask web app
Main.py              CLI entry point
services.py          shared prediction / output / history service layer
inference.py         model inference pipeline
scene_heuristics.py  OCR/script-based reranking logic
explanations.py      LLM + fallback explanation layer
visualization.py     map rendering
history.py           prediction history storage
project_config.py    project paths and defaults
Model.py             backbone + checkpoint loading
TuxunAgent.py        Tuxun API integration
templates/           HTML templates
static/              CSS styles
models/              mapping + checkpoint files
```

## Tech Stack

- Python
- Flask
- PyTorch
- torchvision
- Pillow
- OpenCV
- Tesseract OCR
- Leaflet
- OpenStreetMap

## Recommended Environment

I recommend using:

- Python `3.10` or `3.11`
- a virtual environment

I do **not** recommend using Python `3.14` for this project right now, because PyTorch package support can be annoying there.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run The Web App

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Run From CLI

### Predict a local image

```bash
python Main.py --image C:\path\to\sample.jpg --backbone legacy_mobilenet_v3
```

### Predict from Tuxun mode

Put your session cookie into `cookie.txt`, then run:

```bash
python Main.py --game-id <GAME_ID> --backbone legacy_mobilenet_v3
```

## LLM Explainability Setup

If you want the UI to generate explanation text through an LLM, set:

```bash
set OPENAI_API_KEY=your_api_key
set OPENAI_MODEL=gpt-4.1-mini
```

If you are using an OpenAI-compatible endpoint, you can also set:

```bash
set OPENAI_BASE_URL=https://your-endpoint/v1
```

Without these variables, the project will still work and will use a local fallback explanation.

## Current Model Situation

The codebase supports multiple backbones:

- `legacy_mobilenet_v3`
- `convnext_tiny`
- `efficientnet_v2_s`
- `vit_b_16`

But right now, the repo only includes the old checkpoint:

```text
models/v0.3.0.pth
```

So the safest default is still:

```text
legacy_mobilenet_v3
```

If you train your own newer checkpoint, you can run something like:

```bash
python Main.py --image C:\path\to\sample.jpg --backbone convnext_tiny --checkpoint models\your_checkpoint.pth
```

## Known Limitations

This project is way more useful now than the original version, but it is still not magically accurate.

Some of the main issues right now:

- the legacy model is still weak on many real-world uploads
- the confidence distribution is often too flat
- OCR-based reranking helps in some cases, but it is still heuristic
- true accuracy gains will need retraining, not only UI improvements

So I would describe the current version as:

> a much better product layer on top of a still-imperfect geolocation model

## What I’d Improve Next

If I keep iterating on this project, these are the next things I would build:

1. A proper training pipeline for newer backbones like ConvNeXt or ViT
2. Batch testing on a folder of images with evaluation reports
3. Better OCR support for Chinese / multilingual signs
4. Distance-based evaluation, not just classification ranking
5. Heatmap-style geographic output instead of only Top-K markers
6. Better prompt engineering for the explanation layer

## Final Note

This repo is basically me trying to make a computer vision project feel more like a real software product.

I wanted the model output to be visible, debuggable, and explainable, not just "the answer is probably somewhere here."

If you are also building CV side projects as a student, I think this kind of product thinking is actually really fun.
