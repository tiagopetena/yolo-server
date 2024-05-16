from pathlib import Path

from fastapi import FastAPI, WebSocket
import numpy as np
import cv2
from deep_sort import DeepSort


CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3


app = FastAPI()


def load_net(weights_path: str, cfg_path: str):

    net = cv2.dnn.readNet(weights_path, cfg_path, framework="darknet")

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    return net


def detect_people(frame, net, output_layers, classes):
    frame_height, frame_width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (608, 608), crop=False, swapRB=True)
    net.setInput(blob)
    prediction = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in prediction:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if classes[class_id] != "person":
                continue
            if confidence > CONFIDENCE_THRESHOLD:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    frame_detections = []
    for i in indices:
        detection = {"confidence": confidences[i], "bbox": boxes[i]}
        frame_detections.append(detection)

    return frame_detections


@app.websocket("/video")
async def video_websocket(websocket: WebSocket):
    await websocket.accept()

    net = load_net("yolov4.weights", "yolov4.cfg")

    ln = net.getLayerNames()
    output_layers = ln[net.getUnconnectedOutLayers()[0] - 1]
    output_layers = net.getUnconnectedOutLayersNames()
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    while True:
        data = await websocket.receive_bytes()

        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), -1)

        frame_detections = detect_people(frame, net, output_layers, classes)

        await websocket.send_json(frame_detections)
