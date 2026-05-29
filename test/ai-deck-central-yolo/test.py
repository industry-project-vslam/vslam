#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import socket
import struct
import numpy as np
import cv2
from ultralytics import YOLO

parser = argparse.ArgumentParser(description='Connect to AI-deck JPEG streamer example')
parser.add_argument("-n", default="192.168.4.1", metavar="ip", help="AI-deck IP")
parser.add_argument("-p", type=int, default=5000, metavar="port", help="AI-deck port")
parser.add_argument('--save', action='store_true', help="Save streamed images")
parser.add_argument('--scale', type=float, default=2.0, help="Display scale factor")
parser.add_argument('--conf', type=float, default=0.25, help="Detection confidence threshold")
args = parser.parse_args()

deck_port = args.p
deck_ip = args.n

print(f"Connecting to socket on {deck_ip}:{deck_port}...")
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((deck_ip, deck_port))
print("Socket connected")

def rx_bytes(size):
    data = bytearray()
    while len(data) < size:
        packet = client_socket.recv(size - len(data))
        if not packet:
            raise ConnectionError("Socket closed")
        data.extend(packet)
    return data

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

    results = model.predict(frame, conf=args.conf, verbose=False)
    annotated = results[0].plot()

    h, w = annotated.shape[:2]
    display = cv2.resize(annotated, (int(w * args.scale), int(h * args.scale)), interpolation=cv2.INTER_LINEAR)

    cv2.imshow("Detections", display)

    if args.save:
        cv2.imwrite(f"stream_out/img_{count:06d}.png", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
client_socket.close()