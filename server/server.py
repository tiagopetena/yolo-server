from typing import List

from fastapi import FastAPI, WebSocket
import numpy as np
import cv2


CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3


app = FastAPI()


class Tracker():
    def __init__(self):
        self.frames = []
        self.frame_number = 0
        self.people_number = 0

        self.iou_threshold = 0.5

    def add_frame(self, frame: np.ndarray, bboxes: List, confidences: List):
        new_bboxes = []
        for bbox, confidence in zip(bboxes, confidences):
            new_bbox = {
                    "bbox": bbox,
                    "confidence": confidence,
                    "id": None
            }
            new_bboxes.append(new_bbox)
        new_frame = {
            "img": frame,
            "bboxes": new_bboxes
        }
        self.frame_number += 1
        self.frames.append(new_frame)

        return new_frame

    @staticmethod
    def compute_IOU(bbox1, bbox2):
        # Intersection
        # Top left coordinates
        inter_x1 = max(bbox1[0], bbox2[0])
        inter_y1 = max(bbox1[1], bbox2[1])
        # Bottom right coordinates
        inter_x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
        inter_y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
        inter_width = max(0, inter_x2 - inter_x1 + 1)
        inter_height = max(0, inter_y2 - inter_y1)
        intersection = inter_width * inter_height
        # Union
        area1 = bbox1[2] * bbox1[3]
        area2 = bbox2[2] * bbox2[3]
        union = area1 + area2 - intersection

        return intersection / union

    def get_new_id(self):
        return self.people_number

    def update(self):
        frame1 = self.frames[self.frame_number - 1]
        # For the first frame
        if self.frame_number == 1:
            for bbox in frame1["bboxes"]:
                self.people_number += 1
                bbox["id"] = self.get_new_id()
            return frame1["bboxes"]

        # Match with last frame
        frame0 = self.frames[self.frame_number - 2]
        for bbox1 in frame1["bboxes"]:
            for bbox0 in frame0["bboxes"]:
                iou = self.compute_IOU(bbox0["bbox"], bbox1["bbox"])
                if iou > self.iou_threshold:
                    bbox1["id"] = bbox0["id"]
                    break

        # Fill in missing IDs
        for i, bbox in enumerate(frame1["bboxes"]):
            if bbox["id"] is None:
                self.people_number += 1
                bbox["id"] = self.get_new_id()

        return frame1["bboxes"]


def load_net(weights_path: str, cfg_path: str):

    net = cv2.dnn.readNet(weights_path, cfg_path, framework="darknet")

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    return net


def filter_detections(bboxes, confidences):
    indices = cv2.dnn.NMSBoxes(
            bboxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    filtered_confidences = []
    filtered_bboxes = []
    for i in indices:
        filtered_confidences.append(confidences[i])
        filtered_bboxes.append(bboxes[i])
    return filtered_bboxes, filtered_confidences


def detect_people(frame, net, output_layers, classes):
    frame_height, frame_width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
            frame, 1 / 255, (608, 608), crop=False, swapRB=True)
    net.setInput(blob)
    prediction = net.forward(output_layers)

    class_ids = []
    confidences = []
    bboxes = []
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
                bboxes.append([left, top, width, height])

    bboxes, confidences = filter_detections(bboxes, confidences)

    return bboxes, confidences


@app.websocket("/video")
async def video_websocket(websocket: WebSocket):
    await websocket.accept()

    tracker = Tracker()

    net = load_net("yolov4.weights", "yolov4.cfg")

    ln = net.getLayerNames()
    output_layers = ln[net.getUnconnectedOutLayers()[0] - 1]
    output_layers = net.getUnconnectedOutLayersNames()
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    while True:

        # Receive new Frame
        data = await websocket.receive_bytes()
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), -1)

        # Detect people using object detector
        bboxes, confidences = detect_people(frame, net, output_layers, classes)

        tracker.add_frame(frame, bboxes, confidences)
        tracking_result = tracker.update()

        await websocket.send_json(tracking_result)
