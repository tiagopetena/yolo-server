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
        people_count = 0
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
                if bbox["id"] >= people_count:
                    people_count = bbox["id"]

        print(people_count)


asyncio.run(send_frames())
