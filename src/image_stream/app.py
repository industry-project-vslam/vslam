import time
import socket
import struct
import numpy as np
import cv2
from ultralytics import YOLO
import requests
import os
import mimetypes
import tempfile

DRONE_IP = "192.168.4.1"
DRONE_PORT = 5000

API_URL = "http://0.0.0.0:8000"

DRONE_ID = "test"

SCALE = 2.0
CONF = 0.25

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((DRONE_IP, DRONE_PORT))

def rx_bytes(size):
    data = bytearray()
    while len(data) < size:
        packet = client_socket.recv(size - len(data))
        if not packet:
            raise ConnectionError("Socket closed")
        data.extend(packet)
    return data

def _send_frames_batch(api_url: str, drone_id: str, file_paths: list, params: dict, map_id: str | None = None) -> dict:
    """Send a list of local file paths as a multipart POST."""
    url = f"{api_url}/api/upload_frames/{drone_id}"
    open_handles = []
    try:
        file_tuples = []
        for src in file_paths:
            mime = mimetypes.guess_type(src)[0] or "image/jpeg"
            fh = open(src, "rb")
            open_handles.append(fh)
            file_tuples.append(("files", (os.path.basename(src), fh, mime)))

        data = {**params}
        if map_id:
            data["map_id"] = map_id

        r = requests.post(url, files=file_tuples, data=data, timeout=300)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}
    finally:
        for fh in open_handles:
            fh.close()

def upload_frame_once(frame, api_url, drone_id, params=None):
    """Send a single frame to the upload endpoint using multipart form-data."""
    params = params or {}
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        return {"error": "failed to encode frame"}

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(buf.tobytes())
        tmp_path = tmp.name

    try:
        return _send_frames_batch(api_url, drone_id, [tmp_path], params)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

model = YOLO("yolo26n.pt")

start = time.time()
count = 0

cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)

while True:
    packetInfoRaw = rx_bytes(4)
    length, routing, function = struct.unpack('<HBB', packetInfoRaw)

    imgHeader = rx_bytes(length - 2)
    magic, width, height, depth, fmt, size = struct.unpack('<BHHBBI', imgHeader)

    if magic != 0xBC:
        continue

    imgStream = bytearray()
    while len(imgStream) < size:
        packetInfoRaw = rx_bytes(4)
        length, dst, src = struct.unpack('<HBB', packetInfoRaw)
        chunk = rx_bytes(length - 2)
        imgStream.extend(chunk)

    count += 1
    meanTimePerImage = (time.time() - start) / count
    print(f"{meanTimePerImage:.4f} sec/img")
    print(f"{1/meanTimePerImage:.2f} FPS")

    if fmt == 0:
        bayer_img = np.frombuffer(imgStream, dtype=np.uint8)
        bayer_img.shape = (244, 324)
        color_img = cv2.cvtColor(bayer_img, cv2.COLOR_BayerBG2BGR)
        frame = color_img
    else:
        nparr = np.frombuffer(imgStream, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        continue

    result = upload_frame_once(frame, API_URL, DRONE_ID, params={})
    if "error" in result:
        print(f"✗ Frame {count} upload error: {result['error']}")
    else:
        print(f"✓ Frame {count} uploaded successfully")

    results = model.predict(frame, conf=CONF, verbose=False)
    annotated = results[0].plot()

    h, w = annotated.shape[:2]
    display = cv2.resize(annotated, (int(w * SCALE), int(h * SCALE)), interpolation=cv2.INTER_LINEAR)

    cv2.imshow("Detections", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
client_socket.close()