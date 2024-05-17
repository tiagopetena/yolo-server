import cv2
import json
import asyncio
import websockets

import os


VIDEO_PATH = os.getenv('VIDEO_PATH')


async def send_frames():
    async with websockets.connect("ws://localhost:8000/video") as websocket:
        video_path = str(VIDEO_PATH)
        cap = cv2.VideoCapture(video_path)
        frame_n = -1
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_n += 1
            if frame_n <= 360:
                continue
            # Convert frame to bytes
            _, encoded = cv2.imencode(".jpg", frame)
            await websocket.send(encoded.tobytes())

            ack = await websocket.recv()

            detections = json.loads(str(ack))

            for i, bbox in enumerate(detections):
                bbox_x = bbox["bbox"][0]
                bbox_y = bbox["bbox"][1]
                bbox_width = bbox["bbox"][2]
                bbox_height = bbox["bbox"][3]

                cv2.rectangle(
                    frame,
                    (bbox_x, bbox_y),
                    (bbox_x + bbox_width, bbox_y + bbox_height),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    str(bbox["id"]),
                    (bbox_x, bbox_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("frame", frame)
            cv2.waitKey(1)


asyncio.run(send_frames())
