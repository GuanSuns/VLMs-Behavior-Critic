import os
from typing import List
import base64
import io

import cv2
from PIL import Image
import numpy as np


def frame_preprocess(frames: List,
                     target_shape=None,
                     fit_black_background=None,
                     encode_format=None, is_rgb=True):
    """
    @param target_shape: (target_height, target_width)
    @param fit_black_background: (background_height, background_width)
    """
    processed_frames = list()
    for frame_i in range(len(frames)):
        frame = np.copy(frames[frame_i])
        if target_shape is not None:
            # cv2.resize takes (width, height)
            frame = cv2.resize(frame, (target_shape[1], target_shape[0]),
                               interpolation=cv2.INTER_LINEAR)

        if fit_black_background:
            background = np.zeros((*fit_black_background, 3), np.uint8)
            # place the frame in the middle of the background
            background_h, background_w = fit_black_background[0], fit_black_background[1]
            frame_h, frame_w = frame.shape[0], frame.shape[1]
            h_0, w_0 = (background_h - frame_h) // 2, (background_w - frame_w) // 2
            background[h_0:h_0 + frame_h, w_0:w_0 + frame_w, :] = frame
            frame = background

        if encode_format:
            frame = encode_img(frame, encode_format, is_rgb=is_rgb)

        processed_frames.append(frame)
    return processed_frames


def encode_img(img, img_format, is_rgb=True):
    if img_format == 'base64':
        if is_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", img)
        encoded_img = base64.b64encode(buffer).decode("utf-8")
    elif img_format == 'pil_jpg':
        if is_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        encoded_img = Image.open(io.BytesIO(cv2.imencode(".jpg", img)[1]))
    else:
        raise Exception(f'encode_img - unsupported format: {img_format}')
    return encoded_img


def preprocess_traj_img(traj_img, target_shape=None, fit_black_background=None,
                        key_frames=None, keep_frame_per=None, encode_format=None, is_rgb=True):
    """
    The param keep_frame_per is used to down-sample a video
    """
    if key_frames is not None:
        traj_img = [traj_img[_i] for _i in key_frames]
    elif keep_frame_per is not None:
        total_frames = len(traj_img)
        traj_img = [traj_img[_i] for _i in range(total_frames) if _i % keep_frame_per == 0 or _i == total_frames - 1]

    frames = frame_preprocess(traj_img, target_shape=target_shape,
                              fit_black_background=fit_black_background, encode_format=encode_format, is_rgb=is_rgb)
    return frames


def img2video(frames: List, fps: float, video_path: str, is_rgb=True):
    height, width, _ = frames[0].shape
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    # noinspection PyUnresolvedReferences
    out = cv2.VideoWriter(video_path,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (width, height))
    for i in range(len(frames)):
        frame = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR) if is_rgb else frames[i]
        out.write(frame)
    out.release()


def read_video_frames(video_fname, to_rgb=True):
    frames = list()
    video_cap = cv2.VideoCapture(video_fname)

    video_length = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        success, frame = video_cap.read()
        if not success:
            break
        if to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    if video_length != len(frames):
        print(f'[WARNING] read_video_frames - expected to read {video_length} frames but only got {len(frames)}')

    frames = np.array(frames)
    return frames

