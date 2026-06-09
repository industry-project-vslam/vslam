# Object Detection And Posture Classification

This folder contains the model notebooks and runtime assets for the VSLAM
AI-deck human detection and posture classification pipeline.

The pipeline is:

1. Receive an AI-deck camera frame.
2. Detect humans with a YOLO detector.
3. Crop each detected human box.
4. Classify the crop as a posture class.
5. Draw boxes and posture labels on the live camera frame.

## Current Models

Human detection:

```text
runs/human_detection/weights/best.pt
```

Posture classification:

```text
runs/posture_classification/mobilenet_v3_small/best.pt
runs/posture_classification/mobilenet_v3_small/class_to_idx.json
```

The live drone pipeline currently uses:

- detector confidence: `0.23`
- posture unknown confidence: `0.50`
- posture margin rule: disabled

## Training Data

The labeled training datasets are stored in the repository so another laptop can
train and run benchmark checks without a separate dataset transfer.

Expected human detection layout:

```text
data/labeled/human_detection/
|-- data.yaml
`-- train/
    |-- images/
    `-- labels/
```

The current Roboflow human-detection export can be train-only. That is okay.
`notebooks/human_detection_retraining.ipynb` reads this local export and creates
a derived split under the ignored local folder:

```text
data/processed/human_detection_yolo26_full_boxes_split/
```

Expected posture classification layout:

```text
data/labeled/posture_classification/
|-- train/
|   |-- person_laying/
|   |-- person_sitting/
|   `-- person_standing/
|-- val/
`-- test/
```

If posture validation and test folders are missing, the classification notebook
creates a local stratified split under `data/processed/`.

## Notebooks

Main training notebooks:

```text
notebooks/human_detection_retraining.ipynb
notebooks/posture_classification_retraining.ipynb
```

Evaluation and utility notebooks:

```text
notebooks/tune_realtime_pipeline_thresholds.ipynb
notebooks/benchmark_realtime_inference.ipynb
```

Use `human_detection_retraining.ipynb` for retraining the YOLO human detector.
Use `posture_classification_retraining.ipynb` for retraining the MobileNetV3
posture classifier.

## Split Policy

The committed `data/labeled/` folder is the source dataset. The notebooks create
`data/processed/` locally for training/evaluation splits. Those processed splits
are generated artifacts and are not committed.

For development, a local train-only export is enough because the notebooks can
build local processed splits. For model claims, threshold tuning, or comparison
with older models, use those generated validation/test splits or a Roboflow
export that already contains validation/test data.

Commit:

- notebooks and tools needed to recreate training
- requirements files
- `data/labeled/` source datasets
- final selected model artifacts, if the repo should carry runnable inference
  without a separate model download

Do not commit `data/processed/`, benchmark outputs, threshold tuning outputs, or
other generated intermediate datasets.
