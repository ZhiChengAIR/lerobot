#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Contains utilities to process raw data format of HDF5 files like in: https://github.com/tonyzhaozh/act
"""

import gc
import shutil
from pathlib import Path

import h5py
import numpy as np
import torch
import tqdm
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage
import glob
import torchvision
import subprocess

ZCAI_DATASET_VERSION = "2.0"

from lerobot.common.datasets.push_dataset_to_hub.utils import (
    concatenate_episodes,
    save_images_concurrently,
)
from lerobot.common.datasets.utils import (
    calculate_episode_data_index,
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import VideoFrame, encode_video_frames

from lerobot.common.rotation_transformer import RotationTransformer #added by yz

def get_cameras(raw_dir):
    f_dir = glob.glob(str(raw_dir / "episode_0_*.mp4"))
    cam_name = [name.split(".")[0].split("episode_0_")[-1] for name in f_dir]
    return cam_name


def check_format(raw_dir) -> bool:
    # only frames from simulation are uncompressed
    hdf5_paths = list(raw_dir.glob("episode_*.hdf5"))
    num_episodes = len(hdf5_paths)
    assert len(hdf5_paths) != 0
    for hdf5_path in hdf5_paths:
        ep_id = 0
        for i in range(num_episodes):
            if f"episode_{i}" in str(hdf5_path):
                ep_id = i
                break

        with h5py.File(hdf5_path, "r") as data:
            assert "/action" in data
            assert "/observations/qpos" in data

            assert data["/action"].ndim == 2
            assert data["/observations/qpos"].ndim == 2

            num_frames = data["/action"].shape[0]
            assert num_frames == data["/observations/qpos"].shape[0]

            for camera in get_cameras(raw_dir):
                img_shape = get_info_from_video(
                    str(raw_dir / f"episode_{ep_id}_{camera}.mp4")
                )
                assert len(img_shape) == 3
                c, h, w = img_shape
                assert (
                    c < h and c < w
                ), f"Expect (h,w,c) image format but ({h=},{w=},{c=}) provided."


def get_info_from_video(video_path):
    torchvision.set_video_backend("pyav")
    reader = torchvision.io.VideoReader(video_path, "video")
    img = next(reader)
    return img["data"].shape


def get_imgs_from_video(video_path):
    torchvision.set_video_backend("pyav")
    reader = torchvision.io.VideoReader(video_path, "video")
    imgs = list(reader)
    resize = torchvision.transforms.Resize((240, 320))
    imgs_array = [
        np.transpose(np.array(resize(img["data"])), (1, 2, 0)) for img in imgs
    ]
    return imgs_array


def load_from_raw(
    raw_dir: Path,
    videos_dir: Path,
    fps: int,
    video: bool,
    episodes: list[int] | None = None,
):
    # only frames from simulation are uncompressed
    compressed_images = "sim" not in raw_dir.name
    compressed_images = False

    hdf5_files = sorted(raw_dir.glob("episode_*.hdf5"))
    num_episodes = len(hdf5_files)

    # Initialize the RotationTransformer for converting from 'axis_angle' to 'rotation_6d'
    rotation_transformer = RotationTransformer(from_rep='axis_angle', to_rep='rotation_6d')

    #test
    rotation_transformer2 = RotationTransformer(from_rep='rotation_6d', to_rep='axis_angle')

    ep_dicts = []
    ep_ids = episodes if episodes else range(num_episodes)
    for ep_idx in tqdm.tqdm(ep_ids):
        ep_path = hdf5_files[ep_idx]
        with h5py.File(ep_path, "r") as ep:
            v = ep.attrs["version"]
            assert (
                v == "3.0"
            ), f"ZCAI_DATASET_VERSION version {ZCAI_DATASET_VERSION} is not fit for this code version {v},"
            f_fps = ep.attrs["fps"]
            down_sample_factor = f_fps // fps
            assert (
                f_fps >= fps
            ), f"fps in hdf5 {f_fps} cannot be smaller than para --fps {fps}"
            assert (
                f_fps % fps == 0
            ), f"fps in hdf5 {f_fps} cannot be divided evenly by para --fps {fps}"

            num_frames = ep["/action"].shape[0]
            down_sample_num_frames = (num_frames - 1) // down_sample_factor + 1

            # last step of demonstration is considered done
            done = torch.zeros(down_sample_num_frames, dtype=torch.bool)
            done[-1] = True

            state = torch.from_numpy(ep["/observations/qpos"][::down_sample_factor])
            qtor = torch.from_numpy(ep["/observations/qtor"][::down_sample_factor])
            qvel = torch.from_numpy(ep["/observations/qvel"][::down_sample_factor])
            qacc = torch.from_numpy(ep["/observations/qacc"][::down_sample_factor])
            tcppose = torch.from_numpy(
                ep["/observations/tcppose"][::down_sample_factor]
            )
            tcpvel = torch.from_numpy(ep["/observations/tcpvel"][::down_sample_factor])
            action = torch.from_numpy(ep["/action"][::down_sample_factor])
            action_tcp = torch.from_numpy(ep["/action_tcp"][::down_sample_factor])

            # --- added by yz: Convert specific parts of 'action_tcp' from 'axis_angle' to 'rotation_6d' ---

            # Convert 'action_tcp' to NumPy for processing
            action_tcp_np = action_tcp.numpy()  # Shape: [num_frames, action_dim]

            # Extract columns 3-5 and 10-12 (0-based indexing: 3:6 and 10:13)
            axis_angle_arm1 = action_tcp_np[:, 3:6]/180*np.pi   # Shape: [num_frames, 3]
            axis_angle_arm2 = action_tcp_np[:, 10:13]/180*np.pi # Shape: [num_frames, 3]

            # Apply the RotationTransformer to convert to 'rotation_6d'
            rotation_6d_arm1 = rotation_transformer.forward(axis_angle_arm1)  # Shape: [num_frames, 6]
            rotation_6d_arm2 = rotation_transformer.forward(axis_angle_arm2)

            #test
            axis_angle_arm1_back = rotation_transformer.inverse(rotation_6d_arm1)
            print("axis_angle_arm1_back: ", axis_angle_arm1_back)

            axis_angle_arm2_back = rotation_transformer2.forward(rotation_6d_arm2)
            print("axis_angle_arm2_back: ", axis_angle_arm2_back)

            # Convert the rotation_6d back to torch.Tensor
            #rotation_6d_tensor = torch.from_numpy(rotation_6d)  # Shape: [num_frames, 6]

            # Replace the original axis-angle entries with the 'rotation_6d' data
            # To accommodate the increased dimensionality, we'll expand the 'action_tcp' tensor
            # Insert 'rotation_6d' for arm1 at position 3, shifting existing entries
            # Similarly, insert 'rotation_6d' for arm2 at position 10 (adjusted for previous insertion)

            # Split the original 'action_tcp' into parts
            # Before arm1
            action_tcp_before_arm1 = action_tcp_np[:, :3]  # [num_frames, 3]
            # Between arm1 and arm2
            action_tcp_between_arms = action_tcp_np[:, 6:10]  # [num_frames, 4]
            # After arm2
            action_tcp_after_arm2 = action_tcp_np[:, 13:]  # Remaining columns

            # Create new 'action_tcp' with 'rotation_6d' for both arms
            # arm1: replace 3-5 with 6-dim rotation_6d
            # arm2: replace 10-12 with 6-dim rotation_6d
            # Total new columns: 3 (before) + 6 (arm1) + 4 (between) + 6 (arm2) + remaining

            action_tcp_6d_np = np.hstack((
                action_tcp_before_arm1,        # [num_frames, 3]
                rotation_6d_arm1,            # [num_frames, 6] for arm1
                action_tcp_between_arms,       # [num_frames, 4]
                rotation_6d_arm2,            # [num_frames, 6] for arm2
                action_tcp_after_arm2           # [num_frames, remaining]
            ))  # Final shape: [num_frames, original_dim + 6]

            # Convert back to torch.Tensor
            action_tcp_6d = torch.from_numpy(action_tcp_6d_np)  # [num_frames, new_action_dim]

            # --- End Modification ---

            ep_dict = {}
            for camera in get_cameras(raw_dir):
                img_key = f"observation.images.{camera}"

                if video:
                    fname = f"{img_key}_episode_{ep_idx:06d}.mp4"
                    video_path = videos_dir / fname
                    if down_sample_factor == 1:
                        video_path.parent.mkdir(parents=True, exist_ok=True)
                        ffmpeg_cmd = (
                            f"ffmpeg -loglevel error "
                            f"-i {str(raw_dir/f'episode_{ep_idx}_{camera}.mp4')} "
                            "-vcodec libx264 "
                            "-g 2 "
                            "-pix_fmt yuv444p "
                            f"-vf fps={fps},scale=320:240 "
                            f"{str(video_path)}"
                        )
                        subprocess.run(ffmpeg_cmd.split(" "), check=True)  
                    else:
                        # save png images in temporary directory
                        tmp_imgs_dir = videos_dir / "tmp_images"
                        imgs_array = get_imgs_from_video(
                            str(raw_dir / f"episode_{ep_idx}_{camera}.mp4")
                        )
                        imgs_array_down_sample = imgs_array[::down_sample_factor]
                        save_images_concurrently(imgs_array_down_sample, tmp_imgs_dir)
                        # encode images to a mp4 video
                        encode_video_frames(tmp_imgs_dir, video_path, fps)
                        # clean temporary images directory
                        shutil.rmtree(tmp_imgs_dir)

                    # store the reference to the video frame
                    ep_dict[img_key] = [
                        {"path": f"videos/{fname}", "timestamp": i / fps}
                        for i in range(down_sample_num_frames)
                    ]
                else:
                    imgs_array = get_imgs_from_video(
                        str(raw_dir / f"episode_{ep_idx}_{camera}.mp4")
                    )
                    ep_dict[img_key] = [PILImage.fromarray(x) for x in imgs_array][
                        ::down_sample_factor
                    ]

            ep_dict["observation.state"] = state
            ep_dict["observation.qtor"] = qtor
            ep_dict["observation.qvel"] = qvel
            ep_dict["observation.qacc"] = qacc
            ep_dict["observation.tcppose"] = tcppose
            ep_dict["observation.tcpvel"] = tcpvel
            ep_dict["action"] = action
            ep_dict["action_tcp"] = action_tcp_6d #action_tcp, modified by yz
            ep_dict["episode_index"] = torch.tensor([ep_idx] * down_sample_num_frames)
            ep_dict["frame_index"] = torch.arange(0, down_sample_num_frames, 1)
            ep_dict["timestamp"] = torch.arange(0, down_sample_num_frames, 1) / fps
            ep_dict["next.done"] = done
            # TODO(rcadene): add reward and success by computing them in sim

            assert isinstance(ep_idx, int)
            ep_dicts.append(ep_dict)

        gc.collect()

    data_dict = concatenate_episodes(ep_dicts)

    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)
    return data_dict


def to_hf_dataset(data_dict, video) -> Dataset:
    features = {}

    keys = [key for key in data_dict if "observation.images." in key]
    for key in keys:
        if video:
            features[key] = VideoFrame()
        else:
            features[key] = Image()

    features["observation.state"] = Sequence(
        length=data_dict["observation.state"].shape[1],
        feature=Value(dtype="float32", id=None),
    )
    features["observation.qtor"] = Sequence(
        length=data_dict["observation.qtor"].shape[1],
        feature=Value(dtype="float32", id=None),
    )
    features["observation.qvel"] = Sequence(
        length=data_dict["observation.qvel"].shape[1],
        feature=Value(dtype="float32", id=None),
    )
    features["observation.qacc"] = Sequence(
        length=data_dict["observation.qacc"].shape[1],
        feature=Value(dtype="float32", id=None),
    )
    features["observation.tcppose"] = Sequence(
        length=data_dict["observation.tcppose"].shape[1],
        feature=Value(dtype="float32", id=None),
    )
    features["observation.tcpvel"] = Sequence(
        length=data_dict["observation.tcpvel"].shape[1],
        feature=Value(dtype="float32", id=None),
    )

    features["action"] = Sequence(
        length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["action_tcp"] = Sequence(
        length=data_dict["action_tcp"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
    features["next.done"] = Value(dtype="bool", id=None)
    features["index"] = Value(dtype="int64", id=None)

    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset


def from_raw_to_lerobot_format(
    raw_dir: Path,
    videos_dir: Path,
    fps: int | None = None,
    video: bool = True,
    episodes: list[int] | None = None,
):
    # sanity check
    check_format(raw_dir)

    if fps is None:
        fps = 30

    data_dict = load_from_raw(raw_dir, videos_dir, fps, video, episodes)
    hf_dataset = to_hf_dataset(data_dict, video)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "fps": fps,
        "video": video,
    }
    return hf_dataset, episode_data_index, info
