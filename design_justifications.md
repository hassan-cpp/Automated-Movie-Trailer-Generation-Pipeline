# Task 2: AI Movie Trailer Generator - Design Justifications

This document provides the rationale and design justifications for the end-to-end AI Movie Trailer generation pipeline implemented for the AI LAB OEL.

## 1. Feature Extraction Pipeline (`task2_1_feature_extraction.py`)
- **Visual Intensity (Motion):** We used simple but computationally efficient absolute frame differencing (`cv2.absdiff`) across grayscale frames. High frame differencing correlates heavily with action, fast cuts, and intense scenes.
- **Object Density:** YOLOv8n was chosen to count objects per frame. A higher object count often indicates complex, interesting, or chaotic scenes (e.g., crowds, multiple vehicles, action set-pieces), which are ideal for trailers.
- **Visual Embeddings (ResNet18):** We used a pre-trained ResNet18 model to extract 512-dimensional semantic embeddings from the frames. We stored the top 10 dimensions to give our classifier mathematical awareness of the scene's spatial features and textures without overwhelming it with 512 parameters given the small dataset.
- *Note: Audio extraction was removed from the final pipeline as the provided dataset clips lacked audio tracks.*

## 2. Trained Classification Model (`task2_2_model_training.py`)
- **Heuristic Labeling:** Since the dataset was unlabeled, we used an unsupervised heuristic approach. We normalized the `motion_intensity` and `object_density` metrics, combined them into an `impact_score`, and labeled the top 40% of clips as `1` (High Impact) and the rest as `-1` (Low Impact). 
- **Logistic Regression Model:** We trained a supervised Logistic Regression classifier on these heuristic labels. This model learned to predict the "Impact" based on the extracted feature vectors. Logistic Regression was chosen for its interpretability and speed on small datasets.

## 3. Trailer Generation (`task2_3_trailer_generation.py`)
- **Selection:** We filtered for clips labeled `1` by the model and selected the top 5 with the highest heuristic scores.
- **Narrative Sorting:** Instead of random concatenation, we sorted the 5 clips strategically to build suspense. The clip with the *lowest* motion intensity is placed first to act as a slow, suspenseful intro. The remaining 4 clips are sorted by *increasing* object density, creating an escalating climax of action/complexity.

## 4. Creepy Visual Transformations (`task2_4_creepy_transformation.py`)
- **YOLO-Targeted Person Shadowing:** Rather than applying a blanket filter, we used YOLO to specifically target human actors. Their bounding boxes are darkened significantly to turn them into eerie, anonymous shadows, and blended with a subtle red aura to simulate a demonic/supernatural presence.
- **Cinematic Glitch Effect:** A randomized chromatic aberration effect was implemented (splitting and shifting the RGB channels). This simulates analog VHS distortion, a staple of analog horror.
- **Fog Tint:** The background is heavily desaturated and overlaid with a cold, sickly blue/green tint to establish a gloomy atmosphere.

## 5. NLP-Generated Captions (`task2_5_nlp_captions.py`)
- **BLIP Architecture:** We used Salesforce's `blip-image-captioning-base` from HuggingFace to visually "watch" and describe the scenes.
- **Rule-Based Creepy NLP Processing:** Because standard BLIP generates mundane captions (e.g., "A man walking down a street"), we implemented a rule-based NLP synonym dictionary that intercepts the caption and rewrites it (e.g., "A shadow creeping down an abandoned path"). We also append randomized eerie suffixes (e.g., "...and they are never coming back.") to turn the AI's literal observations into psychological horror subtitles.

## 6. Impact Score Evaluation (`task2_6_evaluation.py`)
- **Real-Time Evaluation:** The script runs the final transformed trailer back through the same feature extraction logic (YOLO, ResNet, Frame Differencing) and feeds it into the trained Logistic Regression model.
- **Timeline Graph:** It plots the probability/impact score over time using `matplotlib`. This proves that our engineered trailer maintains high impact scores throughout its runtime, successfully validating the pipeline.
