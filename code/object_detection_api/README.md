# Object Detection API

Small inference service for the VSLAM human detection and posture classification
models.

The service receives an image frame and returns JSON detections. It does not
return an annotated image.

## Endpoints

```text
GET  /health
POST /predict
WS   /stream
```

`POST /predict` expects raw image bytes in the request body. JPEG and PNG input
are supported because OpenCV reads the image size from the encoded image.

Example:

```powershell
curl.exe -X POST http://localhost:8000/predict `
  -H "Content-Type: image/jpeg" `
  --data-binary "@frame.jpg"
```

Example response:

```json
{
  "image": {
    "width": 200,
    "height": 200,
    "channels": 3
  },
  "detections": [
    {
      "bbox": [34, 51, 120, 180],
      "human_confidence": 0.82,
      "posture": "person_sitting",
      "posture_confidence": 0.91,
      "unknown": false
    }
  ],
  "timing_ms": {
    "decode": 1.1,
    "inference": 32.4,
    "total": 33.5
  }
}
```

`WS /stream` expects binary image frames. For every received frame, it returns a
JSON message with the same response shape plus a `frame_id`.

## Setup

From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run locally:

```powershell
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## Model Paths

By default, the API loads:

```text
code/object_detection/runs/human_detection/weights/best.pt
code/object_detection/runs/posture_classification/mobilenet_v3_small/best.pt
code/object_detection/runs/posture_classification/mobilenet_v3_small/class_to_idx.json
```

Override with environment variables if needed:

```text
DETECTOR_MODEL
CLASSIFIER_MODEL
CLASS_MAP
DETECTOR_CONFIDENCE
POSTURE_CONFIDENCE
DETECTOR_IMAGE_SIZE
DEVICE
MAX_IMAGE_BYTES
```

Defaults:

```text
DETECTOR_CONFIDENCE=0.23
POSTURE_CONFIDENCE=0.50
DETECTOR_IMAGE_SIZE=320
DEVICE=cpu
MAX_IMAGE_BYTES=5242880
```

## Docker

Build from the repository root, not from this folder:

```powershell
docker build -f code/object_detection_api/Dockerfile -t object-detection-api .
```

Run:

```powershell
docker run --rm -p 8000:8000 object-detection-api
```

The Docker build copies only the API code and selected runtime model artifacts.

## Docker Compose

When the repository root owns `compose.yaml`, add the service like this:

```yaml
services:
  object-detection-api:
    build:
      context: .
      dockerfile: code/object_detection_api/Dockerfile
    ports:
      - "8000:8000"
```

Environment variables are optional because the service has defaults in
`src/config.py`. Add them only when the deployment needs to override the default
model paths, thresholds, image size, device, or upload limit.
