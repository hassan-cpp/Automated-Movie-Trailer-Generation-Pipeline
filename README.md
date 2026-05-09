# AI Cinematic Trailer Generator

# Movie Trailer Generator

A Python-based video processing pipeline built for the AI Lab OEL. The system analyzes raw movie clips, extracts features to determine scene impact, and generates a cohesive, stylized trailer with automated captioning.

## Project Structure

The project is split into two core tasks:

### 1. Object Detection & Image Processing
Found in `character_detection.py`. Uses YOLOv8 to detect objects/characters, extracts the bounding box ROIs, applies local transformations (edge detection, tinting, sharpening), and merges them back into the original video frames.

### 2. Trailer Generation Pipeline
A modular pipeline that processes a dataset of raw clips to build a horror-themed trailer.

* **`extract_features.py`**: Calculates motion intensity via grayscale frame differencing and extracts object density using YOLO.
* **`train_model.py`**: Applies a heuristic (motion + object density) to assign impact scores to clips. Trains a Logistic Regression classifier on the normalized features.
* **`generate_trailer.py`**: Selects the top 5 clips based on impact score. Sorts them narratively (lowest motion first to build suspense, followed by increasing object density) and concatenates them using `moviepy`.
* **`apply_creepy_effects.py`**: Applies a global desaturation/fog filter. Uses YOLO to isolate persons, mapping a darkened shadow and subtle red aura onto them, alongside a randomized chromatic aberration effect for VHS distortion.
* **`generate_captions.py`**: Feeds frames into the HuggingFace BLIP model to generate standard captions, then passes the text through a rule-based dictionary script to swap words for eerie synonyms before overlaying them onto the video.
* **`evaluate_trailer.py`**: Runs the generated trailer back through the classifier to plot the impact score timeline using `matplotlib`.

## Requirements
- Python 3.9+
- PyTorch with CUDA support (for accelerated inference)
- See `requirements.txt` for all dependencies.

## Usage
Run the scripts sequentially:
```bash
python extract_features.py
python train_model.py
python generate_trailer.py
python apply_creepy_effects.py
python generate_captions.py
python evaluate_trailer.py
```
