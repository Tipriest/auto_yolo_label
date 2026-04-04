#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Optional, Tuple, List

import cv2
import numpy as np
import rospy
import rospkg
import tf
import tf2_ros
from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import CameraInfo, Image
from datetime import datetime


def _choose_fourcc(codec: str, ext: str) -> int:
    if codec:
        return cv2.VideoWriter_fourcc(*codec[:4])
    ext = ext.lower()
    if ext == ".avi":
        return cv2.VideoWriter_fourcc(*"XVID")
    if ext == ".mkv":
        return cv2.VideoWriter_fourcc(*"X264")
    return cv2.VideoWriter_fourcc(*"mp4v")


def _ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path)


def _make_output_path(output_dir: str, name: str, fmt: str) -> str:
    filename = f"{name}.{fmt.lstrip('.')}"
    return os.path.join(output_dir, filename)


def _get_image_topics() -> List[str]:
    topics = rospy.get_published_topics()
    return [name for name, typ in topics if typ == "sensor_msgs/Image"]


def _select_topic(topics: List[str], keywords: List[str]) -> str:
    if not topics:
        return ""
    for kw in keywords:
        for name in topics:
            if kw in name.lower():
                return name
    return topics[0]


def _infer_info_topic(image_topic: str) -> str:
    if not image_topic:
        return ""
    for suffix in ["/image_raw", "/image_rect", "/image", "/image_color", "/image_rect_color"]:
        if image_topic.endswith(suffix):
            return image_topic[: -len(suffix)] + "/camera_info"
    return image_topic + "/camera_info"


class VideoStreamWriter:
    def __init__(
        self,
        output_dir: str,
        name: str,
        fmt: str,
        codec: str,
        fps: float,
        infer_fps: bool,
        is_color: bool,
    ) -> None:
        self.output_dir = output_dir
        self.name = name
        self.fmt = fmt
        self.codec = codec
        self.default_fps = fps
        self.infer_fps = infer_fps
        self.is_color = is_color
        self.writer: Optional[cv2.VideoWriter] = None
        self.pending_frame: Optional[np.ndarray] = None
        self.pending_stamp: Optional[rospy.Time] = None

    def _init_writer(self, frame: np.ndarray, fps: float) -> None:
        path = _make_output_path(self.output_dir, self.name, self.fmt)
        ext = os.path.splitext(path)[1] or f".{self.fmt}"
        fourcc = _choose_fourcc(self.codec, ext)
        self.writer = cv2.VideoWriter(
            path, fourcc, fps, (frame.shape[1], frame.shape[0]), isColor=self.is_color
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for {path}")
        rospy.loginfo("Video -> %s (fps=%.2f)", path, fps)

    def write(self, frame: np.ndarray, stamp) -> None:
        if self.writer is None:
            if not self.infer_fps:
                self._init_writer(frame, self.default_fps)
            else:
                if self.pending_frame is None:
                    self.pending_frame = frame
                    self.pending_stamp = stamp
                    return
                dt = (stamp - self.pending_stamp).to_sec()
                if dt <= 0:
                    self.pending_frame = frame
                    self.pending_stamp = stamp
                    return
                fps = 1.0 / max(dt, 1e-6)
                fps = float(np.clip(fps, 1.0, 240.0))
                self._init_writer(self.pending_frame, fps)
                assert self.writer is not None
                self.writer.write(self.pending_frame)
                self.pending_frame = None
                self.pending_stamp = None
        if self.writer is not None:
            self.writer.write(frame)

    def close(self) -> None:
        if self.writer is None and self.pending_frame is not None:
            self._init_writer(self.pending_frame, self.default_fps)
            assert self.writer is not None
            self.writer.write(self.pending_frame)
            self.pending_frame = None
            self.pending_stamp = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None


class TopicToVideo:
    def __init__(self) -> None:
        self.bridge = CvBridge()
        self.rgb_writer: Optional[VideoStreamWriter] = None
        self.depth_writer: Optional[VideoStreamWriter] = None
        self.rgb_info: Optional[CameraInfo] = None
        self.depth_info: Optional[CameraInfo] = None
        self.rgb_model = PinholeCameraModel()
        self.depth_model = PinholeCameraModel()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.depth_raw_index = 0

        self.rgb_topic = rospy.get_param("~rgb_topic", "/rgb_camera")
        self.depth_topic = rospy.get_param("~depth_topic", "/depth_camera")
        self.auto_detect_topics = bool(rospy.get_param("~auto_detect_topics", True))
        self.rgb_info_topic = rospy.get_param("~rgb_info_topic", "auto")
        self.depth_info_topic = rospy.get_param("~depth_info_topic", "auto")
        self.output_dir = rospy.get_param("~output_dir", "")
        self.output_format = rospy.get_param("~output_format", "mp4")
        self.rgb_output_name = rospy.get_param("~rgb_output_name", "rgb")
        self.depth_output_name = rospy.get_param("~depth_output_name", "depth")
        time_suffix = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.rgb_output_name += f"_{time_suffix}"
        self.depth_output_name += f"_{time_suffix}"
        self.fps = float(rospy.get_param("~fps", 30))
        self.codec = rospy.get_param("~codec", "")
        self.infer_fps = bool(rospy.get_param("~infer_fps", True))
        self.sync_queue = int(rospy.get_param("~sync_queue", 10))
        self.sync_slop = float(rospy.get_param("~sync_slop", 0.05))
        self.align_depth_to_rgb = bool(rospy.get_param("~align_depth_to_rgb", True))
        self.use_tf = bool(rospy.get_param("~use_tf", True))
        self.tf_timeout = float(rospy.get_param("~tf_timeout", 0.1))
        self.depth_min = float(rospy.get_param("~depth_min", 0.0))
        self.depth_max = float(rospy.get_param("~depth_max", 5.0))
        self.depth_unit = rospy.get_param("~depth_unit", "mm")
        self.depth_raw_output = rospy.get_param("~depth_raw_output_selection", None)
        self.depth_raw_dir = rospy.get_param("~depth_raw_dir", "depth_raw")
        self.depth_raw_index_width = int(rospy.get_param("~depth_raw_index_width", 6))

        if self.auto_detect_topics:
            topics = _get_image_topics()
            if self.rgb_topic in ["", "auto"]:
                self.rgb_topic = _select_topic(topics, ["rgb", "color"])
            if self.depth_topic in ["", "auto"]:
                self.depth_topic = _select_topic(topics, ["depth"])
            rospy.loginfo("Auto topics: rgb=%s depth=%s", self.rgb_topic, self.depth_topic)

        if self.rgb_info_topic == "auto":
            self.rgb_info_topic = _infer_info_topic(self.rgb_topic)
        if self.depth_info_topic == "auto":
            self.depth_info_topic = _infer_info_topic(self.depth_topic)

        self.use_rgb = bool(self.rgb_topic)
        self.use_depth = bool(self.depth_topic)

        if not self.use_rgb and not self.use_depth:
            rospy.logerr("No topics configured: set ~rgb_topic and/or ~depth_topic.")
            raise RuntimeError("No topics configured")

        if not self.output_dir:
            pkg_path = rospkg.RosPack().get_path("rostopic_to_video")
            self.output_dir = os.path.join(pkg_path, "videos")
        _ensure_dir(self.output_dir)

        if self.rgb_info_topic:
            self.rgb_info_sub = rospy.Subscriber(self.rgb_info_topic, CameraInfo, self._on_rgb_info)
        if self.depth_info_topic:
            self.depth_info_sub = rospy.Subscriber(self.depth_info_topic, CameraInfo, self._on_depth_info)

        if self.use_rgb and self.use_depth:
            rgb_sub = Subscriber(self.rgb_topic, Image)
            depth_sub = Subscriber(self.depth_topic, Image)
            sync = ApproximateTimeSynchronizer(
                [rgb_sub, depth_sub], queue_size=self.sync_queue, slop=self.sync_slop
            )
            sync.registerCallback(self._on_rgb_depth)
            rospy.loginfo("Recording RGB+Depth (time sync) from %s and %s", self.rgb_topic, self.depth_topic)
        elif self.use_rgb:
            self.rgb_sub = rospy.Subscriber(self.rgb_topic, Image, self._on_rgb)
            rospy.loginfo("Recording RGB from %s", self.rgb_topic)
        else:
            self.depth_sub = rospy.Subscriber(self.depth_topic, Image, self._on_depth)
            rospy.loginfo("Recording Depth from %s", self.depth_topic)

    def _normalize_depth(self, depth: np.ndarray) -> np.ndarray:
        if depth.dtype == np.uint16:
            if self.depth_unit == "mm":
                depth = depth.astype(np.float32) * 0.001
            else:
                depth = depth.astype(np.float32)
        else:
            depth = depth.astype(np.float32)

        dmin = self.depth_min
        dmax = self.depth_max
        if dmax <= dmin:
            valid = np.isfinite(depth)
            if np.any(valid):
                dmin = float(np.min(depth[valid]))
                dmax = float(np.max(depth[valid]))
            else:
                dmin, dmax = 0.0, 1.0
        depth = np.clip(depth, dmin, dmax)
        depth = (depth - dmin) / (dmax - dmin + 1e-6)
        depth_u8 = (depth * 255.0).astype(np.uint8)
        return depth_u8

    def _depth_to_meters(self, depth: np.ndarray) -> np.ndarray:
        if depth.dtype == np.uint16:
            if self.depth_unit == "mm":
                return depth.astype(np.float32) * 0.001
            return depth.astype(np.float32)
        return depth.astype(np.float32)

    def _save_depth_raw(self, depth_m: np.ndarray) -> None:
        if self.depth_raw_output == None:
            return
        raw_dir = os.path.join(self.output_dir, self.depth_raw_dir)
        _ensure_dir(raw_dir)
        idx = str(self.depth_raw_index).zfill(self.depth_raw_index_width)
        self.depth_raw_index += 1
        if self.depth_raw_output == "npy":
            path = os.path.join(raw_dir, f"depth_{idx}.npy")
            np.save(path, depth_m)
            return
        depth_mm = np.clip(depth_m * 1000.0, 0.0, 65535.0).astype(np.uint16)
        path = os.path.join(raw_dir, f"depth_{idx}.png")
        cv2.imwrite(path, depth_mm)

    def _on_rgb_info(self, msg: CameraInfo) -> None:
        self.rgb_info = msg
        self.rgb_model.fromCameraInfo(msg)

    def _on_depth_info(self, msg: CameraInfo) -> None:
        self.depth_info = msg
        self.depth_model.fromCameraInfo(msg)

    def _lookup_transform(self, target_frame: str, source_frame: str, stamp) -> np.ndarray:
        transform = self.tf_buffer.lookup_transform(
            target_frame, source_frame, stamp, rospy.Duration.from_sec(self.tf_timeout)
        )
        t = transform.transform.translation
        q = transform.transform.rotation
        mat = tf.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
        mat[0, 3] = t.x
        mat[1, 3] = t.y
        mat[2, 3] = t.z
        return mat

    def _align_depth_to_rgb(
        self,
        depth_m: np.ndarray,
        depth_frame: str,
        rgb_frame: str,
        stamp,
        rgb_size: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        if self.rgb_info is None or self.depth_info is None:
            rospy.logwarn_throttle(2.0, "CameraInfo not ready; skip depth alignment")
            return None
        if self.align_depth_to_rgb is False:
            return depth_m

        if self.use_tf and depth_frame != rgb_frame:
            try:
                t_depth_to_rgb = self._lookup_transform(rgb_frame, depth_frame, stamp)
            except Exception as e:
                rospy.logwarn_throttle(1.0, "TF lookup failed: %s", e)
                return None
        else:
            t_depth_to_rgb = np.eye(4, dtype=np.float32)

        h, w = depth_m.shape
        u = np.arange(w)
        v = np.arange(h)
        uu, vv = np.meshgrid(u, v)
        z = depth_m
        valid = np.isfinite(z) & (z > 0.0)
        if not np.any(valid):
            return None

        fx_d = self.depth_model.fx()
        fy_d = self.depth_model.fy()
        cx_d = self.depth_model.cx()
        cy_d = self.depth_model.cy()

        x = (uu[valid] - cx_d) * z[valid] / fx_d
        y = (vv[valid] - cy_d) * z[valid] / fy_d
        z = z[valid]
        ones = np.ones_like(z)
        pts = np.vstack([x, y, z, ones])
        pts_rgb = t_depth_to_rgb.dot(pts)
        x_r, y_r, z_r = pts_rgb[0], pts_rgb[1], pts_rgb[2]

        fx_r = self.rgb_model.fx()
        fy_r = self.rgb_model.fy()
        cx_r = self.rgb_model.cx()
        cy_r = self.rgb_model.cy()

        u_r = (fx_r * x_r / z_r + cx_r)
        v_r = (fy_r * y_r / z_r + cy_r)
        u_r = np.round(u_r).astype(np.int32)
        v_r = np.round(v_r).astype(np.int32)

        rgb_w, rgb_h = rgb_size
        in_bounds = (z_r > 0.0) & (u_r >= 0) & (u_r < rgb_w) & (v_r >= 0) & (v_r < rgb_h)
        if not np.any(in_bounds):
            return None

        u_r = u_r[in_bounds]
        v_r = v_r[in_bounds]
        z_r = z_r[in_bounds]
        aligned = np.full((rgb_h, rgb_w), np.inf, dtype=np.float32)
        idx = v_r * rgb_w + u_r
        np.minimum.at(aligned.ravel(), idx, z_r)
        aligned[~np.isfinite(aligned)] = 0.0
        return aligned

    def _write_rgb(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if self.rgb_writer is None:
            self.rgb_writer = VideoStreamWriter(
                self.output_dir,
                self.rgb_output_name,
                self.output_format,
                self.codec,
                self.fps,
                self.infer_fps,
                True,
            )
        self.rgb_writer.write(frame, msg.header.stamp)

    def _write_depth(self, depth_m: np.ndarray, stamp) -> None:
        depth_u8 = self._normalize_depth(depth_m)
        depth_bgr = cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2BGR)
        if self.depth_writer is None:
            self.depth_writer = VideoStreamWriter(
                self.output_dir,
                self.depth_output_name,
                self.output_format,
                self.codec,
                self.fps,
                self.infer_fps,
                True,
            )
        self.depth_writer.write(depth_bgr, stamp)

    def _on_rgb(self, msg: Image) -> None:
        try:
            self._write_rgb(msg)
        except Exception as e:
            rospy.logerr_throttle(1.0, "RGB write error: %s", e)

    def _on_depth(self, msg: Image) -> None:
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            depth_m = self._depth_to_meters(depth)
            self._save_depth_raw(depth_m)
            self._write_depth(depth_m, msg.header.stamp)
        except Exception as e:
            rospy.logerr_throttle(1.0, "Depth write error: %s", e)

    def _on_rgb_depth(self, rgb_msg: Image, depth_msg: Image) -> None:
        try:
            self._write_rgb(rgb_msg)
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            depth_m = self._depth_to_meters(depth)
            rgb_frame = rgb_msg.header.frame_id
            depth_frame = depth_msg.header.frame_id
            rgb_size = (rgb_msg.width, rgb_msg.height)
            aligned = self._align_depth_to_rgb(
                depth_m, depth_frame, rgb_frame, depth_msg.header.stamp, rgb_size
            )
            if aligned is None:
                aligned = depth_m
            self._save_depth_raw(aligned)
            self._write_depth(aligned, depth_msg.header.stamp)
        except Exception as e:
            rospy.logerr_throttle(1.0, "RGB+Depth write error: %s", e)

    def close(self) -> None:
        if self.rgb_writer is not None:
            self.rgb_writer.close()
        if self.depth_writer is not None:
            self.depth_writer.close()


def main() -> None:
    rospy.init_node("rostopic_to_video")
    recorder = None
    try:
        recorder = TopicToVideo()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        if recorder is not None:
            recorder.close()


if __name__ == "__main__":
    main()
