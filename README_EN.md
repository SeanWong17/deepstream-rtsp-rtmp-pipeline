# Python DeepStream RTSP-to-RTMP High-Performance Video Pipeline

[中文说明](README.md)

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](https://www.docker.com/)

This project is a Python-based NVIDIA DeepStream scaffold that demonstrates a full RTSP-to-RTMP workflow: pull an RTSP stream, decode on GPU, process frames in Python, then encode and push the result to an RTMP server.

## Project Layout

- `main.py`: Core pipeline implementation, including RTSP ingest, frame processing, RTMP publishing, and shutdown handling.
- `Dockerfile`: Runtime image built on top of the official NVIDIA DeepStream container.
- `requirements.txt`: Extra Python dependencies used by the demo.
- `README.md`: Chinese documentation.
- `README_EN.md`: English documentation.

## Features

- GPU-accelerated decode and encode with NVIDIA DeepStream.
- Python/OpenCV processing stage that is easy to replace with AI inference.
- RTSP ingest and RTMP publishing in one script.
- Docker-based deployment path for servers and edge devices.
- Safer runtime behavior for quoted stream URLs, frame extraction, and shutdown.

## Architecture

The pipeline has three stages:

1. Decode
   `rtspsrc` pulls the RTSP stream, `nvv4l2decoder` decodes H.264 on GPU, `nvvideoconvert` converts frames to CPU-side BGR, and `appsink` hands the frames to Python.

2. Process
   A background Python thread reads BGR frames from a bounded queue, applies image processing, and prepares the output frames.

3. Encode and Publish
   `appsrc` feeds the processed frames back into GStreamer, `nvv4l2h264enc` encodes them on GPU, `flvmux` wraps the stream, and `rtmpsink` publishes to the RTMP target.

## Requirements

- NVIDIA GPU, Pascal generation or newer.
- NVIDIA driver `>= 525.85.12`.
- Docker.
- NVIDIA Container Toolkit.

## Quick Start

1. Clone the repository.

```bash
git clone https://github.com/SeanWong17/deepstream-rtsp-rtmp-pipeline.git
cd deepstream-rtsp-rtmp-pipeline
```

2. Build the Docker image.

```bash
docker build -t deepstream-pipeline .
```

3. Update the stream configuration in `main.py`.

```python
if __name__ == "__main__":
    rtsp_url = "rtsp://your.rtsp.stream/url"
    rtmp_url = "rtmp://your.rtmp.server/live/stream_key"
    width = 1920
    height = 1080
    framerate = 30
```

`width` and `height` define the processing/output resolution between `appsink` and `appsrc`. The code reads the actual decoded frame dimensions from sample caps and derives row stride from the mapped buffer instead of blindly reshaping raw bytes.

4. Run the container.

```bash
docker run --gpus all -it --rm --net=host deepstream-pipeline
```

## Notes

- The queue between decode and processing is intentionally short to favor real-time behavior over buffering stale frames.
- RTSP and RTMP URLs are quoted before being inserted into `Gst.parse_launch()` to avoid breakage with credentials or query parameters.
- The demo processing step draws a rectangle and label; replace that section with your inference or business logic.

## Extension Ideas

- Add AI inference in `processing_loop()`.
- Split heavy processing into multiple worker stages.
- Tune encoder bitrate and keyframe interval for your target workload.
- Run multiple `DeepStreamProcessor` instances for multi-stream scenarios.

## License

Released under the [MIT License](LICENSE).
