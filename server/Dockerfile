FROM python:3.12-slim

WORKDIR /yolo-tracker


COPY . .

EXPOSE 8000

# Install system pkgs
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -y

RUN apt install -y libgl1-mesa-glx \
    ffmpeg \
    libsm6 \
    libxext6 \
    wget

RUN wget "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights"

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
