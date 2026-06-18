FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/code/object_detection_api

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY code/object_detection_api/requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY code/object_detection_api /app/code/object_detection_api
COPY code/object_detection/runs/human_detection/weights/best.pt /app/code/object_detection/runs/human_detection/weights/best.pt
COPY code/object_detection/runs/posture_classification/mobilenet_v3_small/best.pt /app/code/object_detection/runs/posture_classification/mobilenet_v3_small/best.pt
COPY code/object_detection/runs/posture_classification/mobilenet_v3_small/class_to_idx.json /app/code/object_detection/runs/posture_classification/mobilenet_v3_small/class_to_idx.json

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

