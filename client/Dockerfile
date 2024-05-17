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


RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "client.py"]

