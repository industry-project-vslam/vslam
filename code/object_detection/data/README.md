# Data

This folder separates committed source datasets from generated local splits.

## Committed Source Data

These folders are part of the repository because another machine must be able to
train and benchmark the models without downloading a separate dataset:

```text
data/labeled/human_detection/
data/labeled/posture_classification/
```

`data/labeled/human_detection/` contains the Roboflow YOLO human detection
export used by `notebooks/human_detection_retraining.ipynb`.

`data/labeled/posture_classification/` contains posture crop images grouped by
class, for example `person_laying`, `person_sitting`, and `person_standing`.

## Generated Data

The notebooks create derived splits and prepared labels under:

```text
data/processed/
```

`data/processed/` is ignored by Git. It can be deleted and recreated by running
the training notebooks again.

Do not manually edit generated files in `data/processed/`. Fix labels or source
images in `data/labeled/`, then rerun the relevant notebook.
