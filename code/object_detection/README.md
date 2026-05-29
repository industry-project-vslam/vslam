# Drone Vision Model

This repository is for a Crazyflie / Bitcraze AI-deck indoor vision project.

Right now, the repository only keeps the raw drone images that will be used to
build a future custom dataset. There is no finished training dataset yet.

## Current Structure

```text
|-- data/
|   `-- raw/
|       `-- images/
|           `-- drone_classroom_images/
|-- scripts/
`-- README.md
```

## Current Data

Raw AI-deck drone images are stored in:

```text
data/raw/images/drone_classroom_images
```

These images are original source images. They are not split into train,
validation, or test folders yet.

## Project Goal

The goal is to create a custom object-detection dataset for low-quality
AI-deck indoor images.

Planned first classes:

- `person`
- `door`
- `window`

## Next Steps

1. Review the raw drone images.
2. Label the useful images manually.
3. Create a YOLO dataset later, after labels exist.
4. Split the labeled dataset into train, validation, and test sets.
5. Train a model only after the labeled dataset is ready.

## Notes

- Do not train directly from `data/raw/images/drone_classroom_images`.
- Raw images should stay unchanged.
- Validation and test data should use real AI-deck images.
- `scripts/` is currently empty and can be used later for project scripts.
