import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
import cv2
import numpy as np
import queue
import threading
import time
import logging

class DeepStreamProcessor:
    """
    深度流处理脚手架:
    ----------------
    1. RTSP解码 -> appsink (CPU 侧获取帧)
    2. (可插入推理/显式处理逻辑等)
    3. 推送处理后的帧到 appsrc -> 编码推流 (RTMP)

    可以在此基础上:
    - 添加推理模块 (模型加载、GPU推理、CPU后处理等)
    - 添加告警逻辑 (异步发送告警帧、报警消息等)
    - 修改编码参数 (码率、I帧间隔、插入SPS/PPS等)
    - 修正时间戳或者改用固定fps
    """

    def __init__(self, rtsp_url, rtmp_url, width, height, framerate=30, logger=None):
        Gst.init(None)
        self.rtsp_url = rtsp_url
        self.rtmp_url = rtmp_url
        self.width = width
        self.height = height
        self.framerate = framerate

        self.logger = logger or logging.getLogger("DeepStreamProcessor")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        self.stop_event = threading.Event()
        self.loop = None
        self.main_thread = None
        self.processing_thread = None

        self.decode_pipeline = None
        self.appsink = None

        self.encode_pipeline = None
        self.appsrc = None

        # 用于在解码线程与处理/推流之间交换帧
        self.frame_queue = queue.Queue(maxsize=100)
        self.frame_count = 0
        self._lock = threading.Lock()

    def build_decode_pipeline(self):
        """
        从RTSP拉流，解码并输出BGR格式CPU侧帧。
        注意加上queue，适当控制流量/缓冲。
        """
        decode_pipeline_desc = f"""
            rtspsrc location={self.rtsp_url} latency=100 do-retransmission=true !
            queue !
            rtph264depay ! h264parse !
            nvv4l2decoder !
            queue !
            nvvideoconvert !
            video/x-raw,format=BGR,width={self.width},height={self.height},framerate={self.framerate}/1 !
            appsink name=appsink0 emit-signals=true sync=false max-buffers=30 drop=true
        """
        self.decode_pipeline = Gst.parse_launch(decode_pipeline_desc)
        self.appsink = self.decode_pipeline.get_by_name("appsink0")
        self.appsink.set_property("emit-signals", True)
        self.appsink.connect("new-sample", self.on_new_sample)

    def build_encode_pipeline(self):
        """
        接收CPU侧帧，通过appsrc打包后，H.264编码并推送到RTMP。
        """
        encode_pipeline_desc = f"""
            appsrc name=appsrc0 emit-signals=true is-live=true do-timestamp=true block=true format=time !
            video/x-raw,format=BGR,width={self.width},height={self.height},framerate={self.framerate}/1 !
            queue !
            videoconvert !
            nvvideoconvert !
            video/x-raw(memory:NVMM),format=I420 !
            nvv4l2h264enc bitrate=4000000 iframeinterval=30 !
            h264parse !
            queue !
            flvmux streamable=true !
            rtmpsink location='{self.rtmp_url}' sync=false
        """
        self.encode_pipeline = Gst.parse_launch(encode_pipeline_desc)
        self.appsrc = self.encode_pipeline.get_by_name("appsrc0")

    def on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.ERROR

        buffer = sample.get_buffer()
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        try:
            frame_data = np.frombuffer(map_info.data, dtype=np.uint8).reshape(self.height, self.width, 3)
            if not self.frame_queue.full():
                self.frame_queue.put_nowait(frame_data)
        except queue.Full:
            self.logger.warning("Frame queue is full, dropping frame.")
        except Exception as e:
            self.logger.error(f"Failed to read decoded frame: {e}")
        finally:
            buffer.unmap(map_info)

        return Gst.FlowReturn.OK

    def processing_loop(self):
        """
        在后台线程中不断从frame_queue取帧，执行处理操作后推送到 appsrc。
        """
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            if frame is None:
                continue

            # --- 在这里插入你的处理逻辑 (AI推理, 图像增强等) ---
            # 简单演示：画一个矩形和文字
            processed_frame = frame.copy()
            cv2.rectangle(processed_frame, (50, 50), (200, 200), (0, 255, 0), 2)
            cv2.putText(processed_frame, "DeepStream Demo", (60, 45), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            # --- 处理逻辑结束 ---

            self.push_frame_to_appsrc(processed_frame)

    def push_frame_to_appsrc(self, frame):
        if not self.appsrc:
            return

        with self._lock:
            data = frame.tobytes()
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)

            # 手动计算时间戳，确保连续
            frame_duration = Gst.util_uint64_scale(1, Gst.SECOND, self.framerate)
            buf.pts = self.frame_count * frame_duration
            buf.duration = frame_duration
            self.frame_count += 1

            ret = self.appsrc.emit("push-buffer", buf)
            if ret != Gst.FlowReturn.OK:
                self.logger.error(f"Pushing frame to appsrc failed with status: {ret}")

    def add_bus_watch(self, pipeline, pipeline_name):
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        from functools import partial
        bus.connect("message", partial(self.bus_call, pipeline_name=pipeline_name))

    def bus_call(self, bus, message, pipeline_name="UnknownPipeline"):
        msg_type = message.type
        if msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            self.logger.error(f"[{pipeline_name}] Error: {err}, {debug}")
            self.stop()
        elif msg_type == Gst.MessageType.EOS:
            self.logger.info(f"[{pipeline_name}] End-Of-Stream reached.")
            self.stop()
        return True

    def start(self):
        self.stop_event.clear()
        self.build_decode_pipeline()
        self.build_encode_pipeline()

        self.add_bus_watch(self.decode_pipeline, "DecodePipeline")
        self.add_bus_watch(self.encode_pipeline, "EncodePipeline")

        self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)

        self.decode_pipeline.set_state(Gst.State.PLAYING)
        self.encode_pipeline.set_state(Gst.State.PLAYING)

        self.loop = GLib.MainLoop()
        self.main_thread = threading.Thread(target=self.loop.run, daemon=True)

        self.processing_thread.start()
        self.main_thread.start()

        self.logger.info("DeepStream Pipeline started...")

    def stop(self):
        if self.stop_event.is_set():
            return
        
        self.logger.info("Stopping DeepStream Pipeline...")
        self.stop_event.set()

        if self.loop and self.loop.is_running():
            self.loop.quit()
        
        if self.decode_pipeline:
            self.decode_pipeline.set_state(Gst.State.NULL)
        if self.encode_pipeline:
            self.encode_pipeline.set_state(Gst.State.NULL)

        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=5)
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)

        self.logger.info("DeepStream Pipeline has been stopped.")


if __name__ == "__main__":
    # --- 请修改为你的实际地址 ---
    rtsp_url = "rtsp://your_rtsp_stream_url"
    rtmp_url = "rtmp://your_rtmp_server/live/stream_key"
    
    # --- 视频流的原始分辨率 ---
    width = 1920
    height = 1080
    framerate = 30
    
    processor = DeepStreamProcessor(rtsp_url, rtmp_url, width, height, framerate)
    try:
        processor.start()
        print("DeepStream Processor is running. Press Ctrl+C to stop.")
        while not processor.stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping DeepStream Processor due to user interrupt...")
    finally:
        processor.stop()
