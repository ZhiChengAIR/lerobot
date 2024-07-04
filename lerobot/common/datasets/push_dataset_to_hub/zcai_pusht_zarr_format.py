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
"""Process zarr files formatted like in: https://github.com/real-stanford/diffusion_policy"""

import shutil
from pathlib import Path

import numpy as np
import torch
import tqdm
import zarr
import os
import ossaudiodev
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage

from lerobot.common.datasets.push_dataset_to_hub.utils import (
    concatenate_episodes,
    save_images_concurrently,
)
from lerobot.common.datasets.utils import (
    calculate_episode_data_index,
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import VideoFrame, encode_video_frames


def load_from_raw(
    raw_dir: Path,
    videos_dir: Path,
    fps: int,
    video: bool,
    episodes: list[int] | None = None,
):
    from lerobot.common.datasets.push_dataset_to_hub._diffusion_policy_replay_buffer import (
        ReplayBuffer as DiffusionPolicyReplayBuffer,
    )
    from numcodecs import register_codec
    from imagecodecs.numcodecs import Jpeg2k

    # 手动注册 JPEG 2000 编解码器
    register_codec(Jpeg2k)

    zarr_path = list(raw_dir.glob("*.zip"))[0]
    with zarr.ZipStore(zarr_path, mode="r") as zip_store:
        zarr_data = DiffusionPolicyReplayBuffer.copy_from_store(
            src_store=zip_store, store=zarr.MemoryStore()
        )

    episode_ids = torch.from_numpy(zarr_data.get_episode_idxs())
    assert len(
        {zarr_data[key].shape[0] for key in zarr_data.keys()}  # noqa: SIM118
    ), "Some data type dont have the same number of total frames."

    states = torch.from_numpy(zarr_data["robot_eef_pose"][:])
    actions = torch.from_numpy(zarr_data["action"][:])
    robot_gripper_qpos = torch.from_numpy(zarr_data["robot_gripper_qpos"][:])

    # load data indices from which each episode starts and ends
    from_ids, to_ids = [], []
    from_idx = 0
    for to_idx in zarr_data.meta["episode_ends"]:
        from_ids.append(from_idx)
        to_ids.append(to_idx)
        from_idx = to_idx

    num_episodes = len(from_ids)

    ep_dicts = []
    ep_ids = episodes if episodes else range(num_episodes)
    raw_video_dir = raw_dir / "videos"
    ## get camera names in subfolder of raw_video_dir randomly
    camera_names = [cam for cam in zarr_data.data.keys() if "camera" in cam]
    img_keys = [f"observation.image.{key}" for key in camera_names]
    video_path = Path(videos_dir)
    video_path.mkdir(parents=True, exist_ok=True)
    # for every episode
    with tqdm.tqdm(
        total=len(ep_ids), desc="Loading image data", mininterval=1.0
    ) as pbar:
        for ep_idx, selected_ep_idx in enumerate(ep_ids):
            from_idx = from_ids[selected_ep_idx]
            to_idx = to_ids[selected_ep_idx]
            num_frames = to_idx - from_idx

            # sanity check
            assert (episode_ids[from_idx:to_idx] == ep_idx).all()

            # construct ep_dict and appended to ep_dicts
            ep_dict = {}

            ep_dict["observation.state"] = states[from_idx:to_idx]
            ep_dict["action"] = actions[from_idx:to_idx]
            ep_dict["episode_index"] = torch.tensor(
                [ep_idx] * num_frames, dtype=torch.int64
            )
            ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
            ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / fps

            # constract label "observation.image" for ep_dict and save video to video_dir
            for key in camera_names:
                ## get image
                image = torch.from_numpy(zarr_data[key][from_idx:to_idx])
                assert image.min() >= 0.0
                assert image.max() <= 255.0
                image = image.type(torch.uint8)
                imgs_array = [x.numpy() for x in image]
                img_key = f"observation.image.{key}"
                if video:
                    # save png images in temporary directory
                    tmp_imgs_dir = videos_dir / "tmp_images"
                    save_images_concurrently(imgs_array, tmp_imgs_dir)

                    # encode images to a mp4 video
                    fname = f"{img_key}_episode_{ep_idx:06d}.mp4"
                    video_path = videos_dir / fname
                    encode_video_frames(tmp_imgs_dir, video_path, fps)

                    # clean temporary images directory
                    shutil.rmtree(tmp_imgs_dir)

                    # store the reference to the video frame
                    ep_dict[img_key] = [
                        {"path": f"videos/{fname}", "timestamp": i / fps}
                        for i in range(num_frames)
                    ]
                else:
                    ep_dict[img_key] = [PILImage.fromarray(x) for x in imgs_array]

            ep_dicts.append(ep_dict)
            pbar.update(1)

    data_dict = concatenate_episodes(ep_dicts)

    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)
    return data_dict


def to_hf_dataset(data_dict, video):
    features = {}

    for key in data_dict.keys():
        if "observation.image" in key:
            features[key] = VideoFrame()

    features["observation.state"] = Sequence(
        length=data_dict["observation.state"].shape[1],
        feature=Value(dtype="float32", id=None),
    )
    features["action"] = Sequence(
        length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
    # features["next.reward"] = Value(dtype="float32", id=None)
    # features["next.done"] = Value(dtype="bool", id=None)
    # features["next.success"] = Value(dtype="bool", id=None)
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
    if fps is None:
        fps = 10

    data_dict = load_from_raw(raw_dir, videos_dir, fps, video, episodes)
    hf_dataset = to_hf_dataset(data_dict, video)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "fps": fps,
        "video": video,
    }
    return hf_dataset, episode_data_index, info
