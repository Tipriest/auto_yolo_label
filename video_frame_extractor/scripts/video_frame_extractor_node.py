#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Dict, Tuple

import cv2
import numpy as np
import rospy
import rospkg
import yaml


def _ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path)


def _resolve_path(path: str, base_dir: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def _load_config(path: str) -> Dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping")
    return data


def _format_name(prefix: str, index: int, zero_pad: int, ext: str) -> str:
    return f"{prefix}_{index:0{zero_pad}d}.{ext.lstrip('.')}"


def _prepare_frame(frame: np.ndarray, downscale: float) -> np.ndarray:
    if downscale <= 0 or downscale >= 1:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray
    h, w = frame.shape[:2]
    new_size = (max(1, int(w * downscale)), max(1, int(h * downscale)))
    resized = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return gray


def _diff_mse(prev: np.ndarray, curr: np.ndarray) -> float:
    diff = prev.astype(np.float32) - curr.astype(np.float32)
    return float(np.mean(diff * diff))


def _diff_hist_corr(prev: np.ndarray, curr: np.ndarray) -> float:
    hist_prev = cv2.calcHist([prev], [0], None, [32], [0, 256])
    hist_curr = cv2.calcHist([curr], [0], None, [32], [0, 256])
    cv2.normalize(hist_prev, hist_prev)
    cv2.normalize(hist_curr, hist_curr)
    return float(cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL))


def _save_frame(
    frame: np.ndarray,
    output_dir: str,
    prefix: str,
    index: int,
    zero_pad: int,
    fmt: str,
    jpg_quality: int,
    png_compression: int,
) -> str:
    filename = _format_name(prefix, index, zero_pad, fmt)
    path = os.path.join(output_dir, filename)
    params = []
    if fmt.lower() in ["jpg", "jpeg"]:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)]
    elif fmt.lower() == "png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), int(png_compression)]
    ok = cv2.imwrite(path, frame, params)
    if not ok:
        raise RuntimeError(f"Failed to write frame: {path}")
    return path


def _frame_interval(video_fps: float, target_fps: float) -> int:
    if video_fps > 0 and target_fps > 0:
        return max(int(round(video_fps / target_fps)), 1)
    return 1


def main() -> None:
    rospy.init_node("video_frame_extractor", anonymous=False)

    config_path = rospy.get_param("~config", "")
    if not config_path:
        pkg_path = rospkg.RosPack().get_path("video_frame_extractor")
        config_path = os.path.join(pkg_path, "config", "frame_extract.yaml")

    config_dir = os.path.dirname(config_path)
    config = _load_config(config_path)
    root_dir = os.path.join(config_dir, "..")

    input_video = _resolve_path(config.get("input_video", ""), config_dir)
    output_dir = _resolve_path(config.get("output_dir", "frames"), root_dir)
    mode = str(config.get("mode", "fps")).lower()

    if not input_video or not os.path.isfile(input_video):
        rospy.logerr("Input video not found: %s", input_video)
        return

    video_stem = os.path.splitext(os.path.basename(input_video))[0]
    output_dir = os.path.join(output_dir, video_stem)
    _ensure_dir(output_dir)

    target_fps = float(config.get("fps", 2.0))
    similarity_method = str(config.get("similarity_method", "mse")).lower()
    similarity_threshold = float(config.get("similarity_threshold", 12.0))
    downscale = float(config.get("downscale", 0.5))
    min_interval_sec = float(config.get("min_interval_sec", 0.0))

    output_format = str(config.get("output_format", "jpg")).lstrip(".")
    prefix = str(config.get("prefix", "frame"))
    zero_pad = int(config.get("zero_pad", 6))
    start_index = int(config.get("start_index", 0))
    jpg_quality = int(config.get("jpg_quality", 95))
    png_compression = int(config.get("png_compression", 3))

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        rospy.logerr("Failed to open video: %s", input_video)
        return

    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    interval = _frame_interval(video_fps, target_fps)

    saved_count = 0
    last_saved_time = -1.0
    prev_feat = None

    frame_index = 0
    output_index = start_index

    rospy.loginfo("Extracting frames -> %s", output_dir)
    rospy.loginfo("Mode=%s, video_fps=%.3f", mode, video_fps)

    while not rospy.is_shutdown():
        ok, frame = cap.read()
        if not ok:
            break

        keep = False
        if mode == "fps":
            if frame_index % interval == 0:
                keep = True
        elif mode == "similarity":
            if prev_feat is None:
                keep = True
            else:
                curr_feat = _prepare_frame(frame, downscale)
                if similarity_method == "hist":
                    diff = _diff_hist_corr(prev_feat, curr_feat)
                    keep = diff < similarity_threshold
                else:
                    diff = _diff_mse(prev_feat, curr_feat)
                    keep = diff >= similarity_threshold
            if keep:
                prev_feat = _prepare_frame(frame, downscale)
        else:
            rospy.logwarn_throttle(5.0, "Unknown mode '%s', fallback to fps", mode)
            if frame_index % interval == 0:
                keep = True

        if keep and min_interval_sec > 0 and video_fps > 0:
            curr_time = frame_index / video_fps
            if last_saved_time >= 0 and (curr_time - last_saved_time) < min_interval_sec:
                keep = False
            else:
                last_saved_time = curr_time

        if keep:
            _save_frame(
                frame,
                output_dir,
                prefix,
                output_index,
                zero_pad,
                output_format,
                jpg_quality,
                png_compression,
            )
            output_index += 1
            saved_count += 1

        frame_index += 1

    cap.release()
    rospy.loginfo("Done. Saved %d frames", saved_count)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        rospy.logerr("Error: %s", exc)
        raise
