# Python DeepStream RTSP-to-RTMP 高性能视频处理管线

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](https://www.docker.com/)

一个基于 Python 和 NVIDIA DeepStream 的高性能视频处理脚手架，演示了如何构建一个从 RTSP 拉流、经由 GPU 解码处理、再到 GPU 编码推向 RTMP 服务器的完整管线。

## 📝 项目简介

在视频监控、智能交通、安防等场景下，实时视频流分析的重要性不言而喻。传统的基于 CPU 的编解码方式在高分辨率或高帧率视频处理上往往会遇到性能瓶颈。

本项目利用 **NVIDIA DeepStream**，充分发挥 GPU 的强大并行计算能力，对视频流进行高效的硬件加速编解码和处理，大幅提升整体性能。结合 Python 的简洁易用性，为开发者提供了一个快速、高效且易于扩展的视频分析解决方案。

## ✨ 主要特性

- **🚀 高效的 GPU 编解码**: 利用 NVIDIA GPU (NVDEC/NVENC) 进行硬件加速，轻松处理高分辨率、高帧率视频，保证低延迟和实时性。
- **🐍 Python 驱动**: 使用 Python 及其丰富的生态系统进行开发，语法简洁，上手快，方便开发者专注于业务逻辑和算法实现。
- **🧩 极佳的扩展性**: 基于 GStreamer 的模块化管线架构，可轻松集成 OpenCV、PyTorch/TensorFlow 等 AI 推理框架，实现自定义的视频分析、目标检测和告警逻辑。
- **📦 Docker化部署**: 提供 `Dockerfile`，一键封装环境依赖，简化在服务器或边缘设备上的部署流程。
- **🔧 线程安全设计**: 解码、处理和编码在不同线程中进行，通过线程安全队列交换数据，确保流程稳定高效。

## ⚙️ 技术架构

本项目的核心流程分为三个主要环节：

1.  **解码 (Decode)**:
    * `rtspsrc` 从指定的 RTSP 地址拉取视频流。
    * `nvv4l2decoder` 利用 GPU 对 H.264 视频流进行硬件解码。
    * `nvvideoconvert` 将解码后的帧转换为 CPU 内存中的 BGR 格式。
    * `appsink` 将 BGR 帧推送给 Python 应用层进行处理。

2.  **处理 (Process)**:
    * 在独立的 Python 线程中，从队列中获取 BGR 帧。
    * 使用 OpenCV 等库进行图像处理（示例中为绘制矩形和文字）。
    * **此环节是集成 AI 推理（如目标检测、跟踪等）的核心位置**。

3.  **编码与推流 (Encode & Push)**:
    * `appsrc` 接收 Python 应用层处理后的 BGR 帧。
    * `nvvideoconvert` 将 BGR 帧转换回 GPU 内存格式。
    * `nvv4l2h264enc` 利用 GPU 将视频帧高效编码为 H.264 格式。
    * `flvmux` 将 H.264 码流封装为 FLV 格式。
    * `rtmpsink` 将最终的视频流推送到指定的 RTMP 服务器。

## 🛠️ 环境准备与运行

### 先决条件

-   NVIDIA 显卡 (Pascal 架构或更高)
-   [NVIDIA 驱动](https://www.nvidia.com/Download/index.aspx) >= 525.85.12
-   [Docker](https://docs.docker.com/engine/install/)
-   [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### 快速开始 (使用 Docker)

1.  **克隆项目**
    ```bash
    git clone https://github.com/SeanWong17/deepstream-rtsp-rtmp-pipeline.git
    cd deepstream-rtsp-rtmp-pipeline
    ```

2.  **构建 Docker 镜像**
    本项目基于 `deepstream:6.3-gc-triton-devel` 镜像，已包含大部分依赖。
    ```bash
    docker build -t deepstream-pipeline .
    ```

3.  **修改配置**
    打开 `main.py` 文件，修改底部的 `if __name__ == "__main__"` 部分，填入你自己的 RTSP 输入地址和 RTMP 输出地址。
    ```python
    if __name__ == "__main__":
        rtsp_url = "rtsp://your.rtsp.stream/url"  # 你的RTSP输入地址
        rtmp_url = "rtmp://your.rtmp.server/live/stream_key" # 你的RTMP输出地址
        width = 1920   # 输出到处理/推流链路的帧宽度
        height = 1080  # 输出到处理/推流链路的帧高度
        # ...
    ```
    `width` 和 `height` 用于约束 `appsink/appsrc` 之间处理链路的输出分辨率；程序会从实际解码出的 sample caps 中读取真实帧尺寸和 stride，避免直接按裸 buffer 强行 reshape。

4.  **运行容器**
    使用以下命令启动容器，它将自动运行处理程序。
    ```bash
    docker run --gpus all -it --rm --net=host deepstream-pipeline
    ```
    * `--gpus all`: 将主机的 NVIDIA GPU 挂载到容器中。
    * `--net=host`: 使用主机网络，方便访问 RTSP 和 RTMP 服务器。
    * `--rm`: 容器退出后自动删除。

    运行后，你将看到 "DeepStream Processor 正在运行..." 的日志。此时，可以从你的 RTMP 服务器上看到处理后的视频流。

## 🧩 代码解析

核心逻辑封装在 `main.py` 的 `DeepStreamProcessor` 类中。

-   `build_decode_pipeline()`: 构建从 `rtspsrc` 到 `appsink` 的解码管线。
-   `build_encode_pipeline()`: 构建从 `appsrc` 到 `rtmpsink` 的编码推流管线。
-   `on_new_sample()`: `appsink` 的回调函数，当有新帧解码完成时触发，将帧放入 `frame_queue`。
-   `processing_loop()`: 后台处理线程的核心。它从 `frame_queue` 中取出帧，进行处理（**可在此处添加AI推理**），然后调用 `push_frame_to_appsrc()`。
-   `push_frame_to_appsrc()`: 将处理后的帧和计算好的时间戳（PTS）推送到 `appsrc`，送入编码管线。
-   `start()` / `stop()`: 控制整个管线的启动和优雅停止。

## 🚀 扩展思路

-   **集成 AI 模型**: 在 `processing_loop` 方法中，加载一个目标检测模型（如 YOLO、Faster R-CNN）。对每一帧执行推理，并将检测框绘制在图像上。
-   **性能优化**:
    * 如果处理逻辑复杂，可将 `processing_loop` 拆分为多个线程，形成处理流水线。
    * 调整 `nvv4l2h264enc` 的 `bitrate` 和 `iframeinterval` 参数以优化码率和画质。
-   **多路视频处理**: 实例化多个 `DeepStreamProcessor` 对象，每个对象处理一路视频流。注意合理分配 GPU 资源。
-   **添加告警逻辑**: 当检测到特定事件（如有人闯入禁区），可通过独立的线程发送告警信息（如调用 API、发送消息到 Kafka/MQTT）。

## 📄 License

本项目采用 [MIT License](LICENSE) 开源。
