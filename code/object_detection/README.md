# Object Detection And Posture Classification

This folder contains the object detection and posture classification part of
the VSLAM AI-deck project. It includes the source datasets, training notebooks,
selected model artifacts, and evaluation notebooks needed to reproduce the
current human detection and posture classification workflow.

The runtime pipeline is:

1. Receive an AI-deck camera frame.
2. Detect every visible human with a YOLO detector.
3. Crop each detected human box.
4. Classify each crop with a posture classifier.
5. Draw the detection box and posture label back on the full camera image.

## Current Runtime Models

Human detection model:

```text
runs/human_detection/weights/best.pt
```

Posture classification model:

```text
runs/posture_classification/mobilenet_v3_small/best.pt
runs/posture_classification/mobilenet_v3_small/class_to_idx.json
```

The live drone pipeline currently uses these settings:

```text
detector confidence: 0.23
posture unknown confidence: 0.50
posture margin rule: disabled
analyze every N frames: 2
```

The posture classifier marks a crop as `unknown` when the top class probability
is below `0.50`. There is no extra top1/top2 margin rule now because it did not
add useful behavior on top of the confidence threshold.

## Repository Layout

```text
data/
  labeled/
    human_detection/
    posture_classification/
  processed/
notebooks/
  human_detection_retraining.ipynb
  posture_classification_retraining.ipynb
  tune_realtime_pipeline_thresholds.ipynb
  benchmark_realtime_inference.ipynb
runs/
  human_detection/weights/best.pt
  posture_classification/mobilenet_v3_small/best.pt
tools/
requirements.txt
```

## Setup

From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For Jupyter notebooks:

```powershell
pip install notebook ipykernel
python -m ipykernel install --user --name vslam-object-detection
jupyter notebook
```

Use the `vslam-object-detection` kernel when running the notebooks.

## Data

Human detection data is stored here:

```text
data/labeled/human_detection/
```

Expected layout:

```text
data/labeled/human_detection/
  data.yaml
  train/
    images/
    labels/
```

The Roboflow human detection export can be train-only. The retraining notebook
creates the local train/validation/test split under:

```text
data/processed/human_detection_yolo26_full_boxes_split/
```

Posture classification data is stored here:

```text
data/labeled/posture_classification/
```

Expected layout:

```text
data/labeled/posture_classification/
  train/
    person_laying/
    person_sitting/
    person_standing/
```

If `val/` and `test/` are not present, the classification notebook creates a
local stratified split under `data/processed/`.

## Training

Train the human detector with:

```text
notebooks/human_detection_retraining.ipynb
```

This notebook:

- reads `data/labeled/human_detection/`
- prepares YOLO box labels and local splits
- trains from the pretrained YOLO baseline
- evaluates the trained model
- keeps the selected runtime model under `runs/human_detection/weights/best.pt`

Train the posture classifier with:

```text
notebooks/posture_classification_retraining.ipynb
```

This notebook:

- reads `data/labeled/posture_classification/`
- creates local classification splits when needed
- trains MobileNetV3 Small
- evaluates per-class behavior
- saves the selected model and class mapping under
  `runs/posture_classification/mobilenet_v3_small/`

The detection notebook automatically uses MPS when available, then CUDA when
available, then CPU as fallback. MPS or CUDA is recommended for detector
training. The classifier notebook is small enough to train on CPU.

## Evaluation Notebooks

Threshold tuning:

```text
notebooks/tune_realtime_pipeline_thresholds.ipynb
```

Use this notebook when choosing the detector confidence for the live pipeline.
The current approach prefers high recall while keeping precision at an
acceptable level, instead of using an arbitrary confidence value.

Inference benchmark:

```text
notebooks/benchmark_realtime_inference.ipynb
```

Use this notebook to measure detector, classifier, and full pipeline inference
time on another machine. It prints timing tables only and does not save charts or
CSV output.

Reference CPU benchmark result:

```text
detector median:      25.666 ms
classifier median:     8.759 ms
full pipeline median: 31.588 ms
full pipeline p95:    48.944 ms
```

Based on these numbers, quantization is not needed for the current runtime
setup. The full pipeline is already fast enough on CPU, and quantizing the
classifier would save little because the detector is the main cost. Quantizing
the detector could reduce accuracy, so it should only be considered if benchmark
results on the target machine show that inference is too slow.

## Drone Runtime

The live camera integration is in the drone connection part of the repository:

```text
../../testing/drone_connection/
```

That runtime loads the selected detector and classifier artifacts from this
folder. It can show the AI-deck stream with human boxes and posture labels
without saving frames to disk.

## Current Limitations

The posture classifier depends on the detector crop quality. If the detector box
cuts off important body parts, classification quality can drop even when the
classifier itself is correct.

The `unknown` class is threshold-based, not a separately trained class. This is
useful for low-confidence crops, but it does not replace adding more labeled
examples for difficult postures or camera angles.

The AI-deck image stream has low resolution and can be noisy. Final performance
should be judged on live or representative AI-deck frames, not only on clean
dataset images.
