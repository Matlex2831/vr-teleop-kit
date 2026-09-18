import logging
import time
from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import CameraConfig, ColorMode
from lerobot.cameras.utils import get_cv2_rotation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

logger = logging.getLogger(__name__)


@dataclass
class ROS2CameraConfig(CameraConfig):
    """Configuration for ROS 2 Camera stream subscriber."""
    color_topic: str = "/camera/color/image_raw"
    depth_topic: str = "/camera/depth/image_raw"
    warmup_s: float = 2.0


class ROS2Camera(Camera):
    """
    Manages interactions with ROS 2 camera topics for frame and depth recording.

    Provides a 1:1 API match to `RealSenseCamera` while executing pure NumPy unpacking
    (compatible with NumPy 2.x and free of `cv_bridge` dependencies).
    """

    _MAX_CONNECT_ATTEMPTS = 3

    def __init__(self, config: ROS2CameraConfig):
        super().__init__(config)
        self.config: ROS2CameraConfig = config

        self.color_topic = config.color_topic
        self.depth_topic = config.depth_topic

        self.width: int | None = config.width
        self.height: int | None = config.height
        self.fps = config.fps
        self.color_mode = config.color_mode
        self.use_rgb = config.use_rgb
        self.use_depth = config.use_depth
        self.warmup_s = config.warmup_s

        self.node: Node | None = None
        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()

        self.latest_color_frame: NDArray[Any] | None = None
        self.latest_depth_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)
        self.capture_width: int | None = None
        self.capture_height: int | None = None
        self._reset_connection_settings()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(color={self.color_topic}, depth={self.depth_topic})"

    def _reset_connection_settings(self) -> None:
        """Restore settings that may have been auto-detected during a failed connection."""
        self.fps = self.config.fps
        self.width = self.config.width
        self.height = self.config.height
        self.warmup_s = self.config.warmup_s
        self.capture_width, self.capture_height = self.width, self.height
        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            self.capture_width, self.capture_height = self.height, self.width

    @property
    def is_connected(self) -> bool:
        """Checks if the ROS 2 node is initialized and the spin thread is active."""
        return self.node is not None and self.thread is not None and self.thread.is_alive()

    def _open_pipeline(self) -> None:
        """Initializes ROS 2 node, subscriptions, and background spin thread."""
        if not rclpy.ok():
            rclpy.init()

        node_name = f"lerobot_ros_camera_{abs(hash(self.color_topic)) % 100000}"
        self.node = Node(node_name)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        if self.use_rgb:
            self.node.create_subscription(
                Image, self.color_topic, self._color_callback, 1
            )
        if self.use_depth:
            self.node.create_subscription(
                Image, self.depth_topic, self._depth_callback, sensor_qos
            )

        self._start_read_thread()

    def _color_callback(self, msg: Image) -> None:
        """ROS Subscriber callback for color frames using pure NumPy binary parsing."""
        try:
            raw = np.frombuffer(msg.data, dtype=np.uint8)

            if msg.encoding in ["bgr8", "passthrough"]:
                img = raw.reshape((msg.height, msg.width, 3))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif msg.encoding == "rgb8":
                img_rgb = raw.reshape((msg.height, msg.width, 3)).copy()
            elif "yuv" in msg.encoding.lower() or "yuy2" in msg.encoding.lower():
                img_yuv = raw.reshape((msg.height, msg.width, 2))
                img_bgr = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR_YUY2)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            else:
                img = raw.reshape((msg.height, msg.width, 3))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            processed_color = self._postprocess_image(img_rgb)
            capture_time = time.perf_counter()

            with self.frame_lock:
                if self.stop_event and self.stop_event.is_set():
                    return
                self.latest_color_frame = processed_color
                self.latest_timestamp = capture_time
                if self.capture_width is None or self.capture_height is None:
                    self.capture_height, self.capture_width = img_rgb.shape[:2]

            self.new_frame_event.set()

        except Exception as e:
            logger.warning(f"{self} failed to process ROS color frame: {e}")

    def _depth_callback(self, msg: Image) -> None:
        """ROS Subscriber callback for 16-bit uint16 depth frames using pure NumPy parsing."""
        try:
            depth_mm = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
            processed_depth = self._postprocess_image(depth_mm, depth_frame=True)

            if processed_depth.ndim == 2:
                processed_depth = processed_depth[..., np.newaxis]

            capture_time = time.perf_counter()

            with self.frame_lock:
                if self.stop_event and self.stop_event.is_set():
                    return
                self.latest_depth_frame = processed_depth
                self.latest_timestamp = capture_time

            self.new_frame_event.set()

        except Exception as e:
            logger.warning(f"{self} failed to process ROS depth frame: {e}")

    def _run_warmup(self) -> None:
        """Blocks until at least one valid frame has been captured by the ROS background thread."""
        self.warmup_s = max(self.warmup_s, 1.0)
        warmup_read = self.async_read if self.use_rgb else self.async_read_depth
        start_time = time.time()

        while time.time() - start_time < self.warmup_s:
            try:
                warmup_read(timeout_ms=self.warmup_s * 1000)
            except TimeoutError:
                pass
            time.sleep(0.1)

        with self.frame_lock:
            if (self.use_rgb and self.latest_color_frame is None) or (
                self.use_depth and self.latest_depth_frame is None
            ):
                raise ConnectionError(f"{self} failed to capture frames during warmup.")

    def _release_after_failed_setup(self) -> None:
        """Releases resources and restores auto-detected settings after a failed attempt."""
        try:
            self._cleanup_resources()
        except Exception:
            logger.exception(f"Failed to fully clean up {self} after connect() failed.")
        self._reset_connection_settings()

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        """Connects to the ROS 2 camera topics and starts background frame reception."""
        if not warmup:
            self._open_pipeline()
            logger.info(f"{self} connected.")
            return

        last_error: Exception | None = None
        for attempt in range(1, self._MAX_CONNECT_ATTEMPTS + 1):
            self._open_pipeline()
            connected = False
            try:
                self._run_warmup()
                connected = True
            except (TimeoutError, ConnectionError) as e:
                last_error = e
            finally:
                if not connected:
                    self._release_after_failed_setup()

            if connected:
                logger.info(f"{self} connected.")
                return

            logger.warning(f"{self} warmup failed (attempt {attempt}/{self._MAX_CONNECT_ATTEMPTS}).")

        raise ConnectionError(
            f"{self} failed to capture frames after {self._MAX_CONNECT_ATTEMPTS} attempts."
        ) from last_error

    def _postprocess_image(self, image: NDArray[Any], depth_frame: bool = False) -> NDArray[Any]:
        """Applies color conversion, dimension validation, and rotation."""
        if self.color_mode and self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid requested color mode '{self.color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        if depth_frame:
            h, w = image.shape[:2]
        else:
            h, w, c = image.shape
            if c != 3:
                raise RuntimeError(f"{self} frame channels={c} do not match expected 3 channels.")

        processed_image = image
        if not depth_frame and self.color_mode == ColorMode.BGR:
            processed_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed_image = cv2.rotate(processed_image, self.rotation)

        return processed_image

    def _read_loop(self) -> None:
        """Background thread executing ROS 2 executor spinning."""
        if self.node is None or self.stop_event is None:
            return
        
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(self.node)

        while not self.stop_event.is_set() and rclpy.ok():
            executor.spin_once(timeout_sec=0.1)

        executor.remove_node(self.node)

    def _start_read_thread(self) -> None:
        """Starts the background ROS spin thread."""
        self._stop_read_thread()
        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, name=f"{self}_ros_spin_loop", daemon=True)
        self.thread.start()

    def _stop_read_thread(self) -> None:
        """Signals the background ROS spin thread to stop and joins it."""
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

        with self.frame_lock:
            self.latest_color_frame = None
            self.latest_depth_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

    def _cleanup_resources(self) -> None:
        """Stops background thread and destroys ROS 2 node."""
        try:
            self._stop_read_thread()
        finally:
            if self.node is not None:
                self.node.destroy_node()
                self.node = None

    def _read(self, read_depth: bool = False) -> NDArray[Any]:
        """Shared helper for synchronous frame reading."""
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        self.new_frame_event.clear()
        return self._async_read(timeout_ms=10000, read_depth=read_depth)

    def _async_read(self, timeout_ms: float, read_depth: bool = False) -> NDArray[Any]:
        """Shared helper for asynchronous frame reading."""
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"Timed out waiting for frame from ROS topic {self} after {timeout_ms} ms.")

        with self.frame_lock:
            frame = self.latest_depth_frame if read_depth else self.latest_color_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return frame

    def _read_latest(self, max_age_ms: int, read_depth: bool = False) -> NDArray[Any]:
        """Shared helper for peeking buffered frames."""
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        with self.frame_lock:
            frame = self.latest_depth_frame if read_depth else self.latest_color_frame
            timestamp = self.latest_timestamp

        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")

        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(f"{self} latest frame is too old: {age_ms:.1f} ms (max allowed: {max_age_ms} ms).")

        return frame

    @check_if_not_connected
    def read(self, color_mode: ColorMode | None = None, timeout_ms: int = 0) -> NDArray[Any]:
        """Reads a single color frame synchronously."""
        if not self.use_rgb:
            raise RuntimeError(f"{self}: cannot read color — camera configured with use_rgb=False.")
        return self._read(read_depth=False)

    @check_if_not_connected
    def read_depth(self, timeout_ms: int = 200) -> NDArray[Any]:
        """Reads a single depth frame synchronously in millimeters."""
        if not self.use_depth:
            raise RuntimeError(f"{self}: cannot read depth — camera configured with use_depth=False.")
        return self._read(read_depth=True)

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """Reads the latest color frame asynchronously."""
        if not self.use_rgb:
            raise RuntimeError(f"{self}: cannot read color — camera configured with use_rgb=False.")
        return self._async_read(timeout_ms=timeout_ms, read_depth=False)

    @check_if_not_connected
    def async_read_depth(self, timeout_ms: float = 200) -> NDArray[np.uint16]:
        """Reads the latest depth frame asynchronously in millimeters."""
        if not self.use_depth:
            raise RuntimeError(f"{self}: cannot read depth — camera configured with use_depth=False.")
        return self._async_read(timeout_ms=timeout_ms, read_depth=True)

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        """Peeks the most recent color frame."""
        if not self.use_rgb:
            raise RuntimeError(f"{self}: cannot read color — camera configured with use_rgb=False.")
        return self._read_latest(max_age_ms=max_age_ms, read_depth=False)

    @check_if_not_connected
    def read_latest_depth(self, max_age_ms: int = 500) -> NDArray[Any]:
        """Peeks the most recent depth frame."""
        if not self.use_depth:
            raise RuntimeError(f"{self}: cannot read depth — camera configured with use_depth=False.")
        return self._read_latest(max_age_ms=max_age_ms, read_depth=True)

    def disconnect(self) -> None:
        """Disconnects from the ROS topics and stops background thread."""
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"Attempted to disconnect {self}, but it appears already disconnected.")

        self._cleanup_resources()
        logger.info(f"{self} disconnected.")