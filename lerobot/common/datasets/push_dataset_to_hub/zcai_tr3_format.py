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


def get_cameras(raw_dir):
    f_dir = glob.glob(str(raw_dir/"episode_0_*.mp4"))
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
            assert "/observations/endpose" in data
            assert data["/action"].ndim == 2
            assert data["/observations/endpose"].ndim == 2
            num_frames = data["/action"].shape[0]
            assert num_frames == data["/observations/endpose"].shape[0]

            for camera in get_cameras(raw_dir):
                img_shape = get_info_from_video(str(raw_dir/f"episode_{ep_id}_{camera}.mp4"))
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
    imgs_array = [np.transpose(np.array(img["data"]),(1,2,0)) for img in imgs]
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

    ep_dicts = []
    ep_ids = episodes if episodes else range(num_episodes)
    for ep_idx in tqdm.tqdm(ep_ids):
        ep_path = hdf5_files[ep_idx]
        with h5py.File(ep_path, "r") as ep:
            v = ep.attrs["version"]
            assert v == "3.0",f"ZCAI_DATASET_VERSION version {ZCAI_DATASET_VERSION} is not fit for this code version {v},"
            f_fps = ep.attrs["fps"]
            assert f_fps == fps,f"fps {fps} is not equal to fps in HDF5 {f_fps}"
            num_frames = ep["/action"].shape[0]

            # last step of demonstration is considered done
            done = torch.zeros(num_frames, dtype=torch.bool)
            done[-1] = True

            # state = torch.from_numpy(ep["/observations/qpos"][:])
            # qtor = torch.from_numpy(ep["/observations/qtor"][:])
            # qvel = torch.from_numpy(ep["/observations/qvel"][:])
            # qacc = torch.from_numpy(ep["/observations/qacc"][:])
            # tcppose = torch.from_numpy(ep["/observations/tcppose"][:])
            # tcpvel = torch.from_numpy(ep["/observations/tcpvel"][:])
            # action = torch.from_numpy(ep["/action"][:])
            # action_tcp = torch.from_numpy(ep["/action_tcp"][:])

            action = torch.from_numpy(ep["/action"][:])
            endpose = torch.from_numpy(ep["/observations/endpose"][:])
            joint_state = torch.from_numpy(ep["/observations/joint_state"][:])

            ep_dict = {}
            for camera in get_cameras(raw_dir):
                img_key = f"observation.images.{camera}"

                if video:
                    # save png images in temporary directory
                    tmp_imgs_dir = videos_dir / "tmp_images"
                    # save_images_concurrently(imgs_array, tmp_imgs_dir)

                    # encode images to a mp4 video
                    fname = f"{img_key}_episode_{ep_idx:06d}.mp4"
                    video_path = videos_dir / fname
                    # encode_video_frames(tmp_imgs_dir, video_path, fps)
                    video_path.parent.mkdir(parents=True, exist_ok=True)
                    ffmpeg_cmd = (
                        f"ffmpeg -r {fps} "
                        "-loglevel error "
                        f"-i {str(raw_dir/f'episode_{ep_idx}_{camera}.mp4')} "
                        "-vcodec libx264 "
                        "-g 2 "
                        "-pix_fmt yuv444p "
                        f"{str(video_path)}"
                        )
                    subprocess.run(ffmpeg_cmd.split(" "), check=True)
                    

                    # clean temporary images directory
                    # shutil.rmtree(tmp_imgs_dir)

                    # store the reference to the video frame
                    ep_dict[img_key] = [
                        {"path": f"videos/{fname}", "timestamp": i / fps}
                        for i in range(num_frames)
                    ]
                else:
                    imgs_array = get_imgs_from_video(str(raw_dir/f"episode_{ep_idx}_{camera}.mp4"))
                    ep_dict[img_key] = [PILImage.fromarray(x) for x in imgs_array]

            # ep_dict["observation.state"] = state
            # ep_dict["observation.qtor"] = qtor
            # ep_dict["observation.qvel"] = qvel
            # ep_dict["observation.qacc"] = qacc
            # ep_dict["observation.tcppose"] = tcppose
            # ep_dict["observation.tcpvel"] = tcpvel
            # ep_dict["action_tcp"] = action_tcp
            ep_dict["observation.endpose"] = endpose
            ep_dict["observation.joint_state"] = joint_state
            ep_dict["action"] = action
            ep_dict["episode_index"] = torch.tensor([ep_idx] * num_frames)
            ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
            ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / fps
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

    # features["observation.state"] = Sequence(
    #     length=data_dict["observation.state"].shape[1],
    #     feature=Value(dtype="float32", id=None),
    # )
    # features["observation.qtor"] = Sequence(
    #     length=data_dict["observation.qtor"].shape[1],
    #     feature=Value(dtype="float32", id=None),
    # )
    # features["observation.qvel"] = Sequence(
    #     length=data_dict["observation.qvel"].shape[1],
    #     feature=Value(dtype="float32", id=None),
    # )
    # features["observation.qacc"] = Sequence(
    #     length=data_dict["observation.qacc"].shape[1],
    #     feature=Value(dtype="float32", id=None),
    # )
    # features["observation.tcppose"] = Sequence(
    #     length=data_dict["observation.tcppose"].shape[1],
    #     feature=Value(dtype="float32", id=None),
    # )
    # features["observation.tcpvel"] = Sequence(
    #     length=data_dict["observation.tcpvel"].shape[1],
    #     feature=Value(dtype="float32", id=None),
    # )

    # features["action"] = Sequence(
    #     length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)
    # )
    # features["action_tcp"] = Sequence(
    #     length=data_dict["action_tcp"].shape[1], feature=Value(dtype="float32", id=None)
    # )
    features["observation.endpose"] = Sequence(
        length=data_dict["observation.endpose"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["observation.joint_state"] = Sequence(
        length=data_dict["observation.joint_state"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["action"] = Sequence(
        length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)
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
