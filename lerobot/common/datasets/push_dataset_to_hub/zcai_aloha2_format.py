#!/usr/bin/env python

from pathlib import Path
import os
import numpy as np
import tqdm
import torch
from PIL import Image as PILImage
import subprocess
from datasets import Dataset, Features, Image, Sequence, Value

from lerobot.common.datasets.video_utils import VideoFrame
from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes
from lerobot.common.datasets.utils import (
    calculate_episode_data_index,
    hf_transform_to_torch,
)


def encode_video_frames(imgs_dir: Path, video_path: Path, fps: int):
    """More info on ffmpeg arguments tuning on `lerobot/common/datasets/_video_benchmark/README.md`"""
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    images_name = sorted(imgs_dir.glob("*.jpg"))
    images_name = images_name[:-1]
    images_name_dir = imgs_dir / "images.txt"
    with open(images_name_dir, "w") as f:
        for img in images_name:
            f.write(f"file '{imgs_dir/img}'\n")

    ffmpeg_cmd = (
        f"ffmpeg -r {fps} "
        "-f concat "
        "-safe 0 "
        "-loglevel error "
        f"-i {images_name_dir} "
        "-vcodec libx264 "
        "-g 2 "
        "-pix_fmt yuv444p "
        f"{str(video_path)}"
    )
    subprocess.run(ffmpeg_cmd.split(" "), check=True)
    os.remove(imgs_dir / "images.txt")


def check_data_sync(ep_raw_data, imgs_dir: Path, fps):
    ZCAI_SYNC_TIME_TOLERANCE = 2 / fps
    imgs_timestamps = [
        float(os.path.splitext(i)[0]) for i in sorted(os.listdir(imgs_dir))
    ]
    robot_timstamp = np.array([i["timestamp"] for i in ep_raw_data])

    # 处理imgs_timestamps中可能存在的时间间隙过大
    assert len(imgs_timestamps) > 2, "[SYNC] imgs_len <= 2"
    if imgs_timestamps[1] - imgs_timestamps[0] > ZCAI_SYNC_TIME_TOLERANCE:
        # imgs_timestamps.pop(0)
        print("[SYNC] del first img")

    new_imgs_time = []
    new_imgs_time.append(imgs_timestamps[0])
    for idx in range(1, len(imgs_timestamps)):
        if imgs_timestamps[idx] - imgs_timestamps[idx - 1] > ZCAI_SYNC_TIME_TOLERANCE:
            insert_num = np.floor(
                (imgs_timestamps[idx] - imgs_timestamps[idx - 1]) * fps
            )
            print(
                f"{imgs_dir}:idx {idx} the deltatime is larger than tolerance,and will insert {insert_num} imgs"
            )

        new_imgs_time.append(imgs_timestamps[idx])

    new_robot_timstamp_idx = []
    for idx, img_timestamp in enumerate(new_imgs_time):
        delta_time = np.abs(img_timestamp - robot_timstamp)
        min_idx = np.argmin(delta_time)
        new_robot_timstamp_idx.append(min_idx)
        if delta_time[min_idx] > ZCAI_SYNC_TIME_TOLERANCE:
            print(
                f"{imgs_dir}: idx {idx} corresponding img timestamp is over tolerance"
            )

    return new_robot_timstamp_idx


def dataset_alignment(raw_dir: Path):
    episode_folder = os.listdir(raw_dir)
    num_episodes = len(episode_folder)
    ep_ids = range(num_episodes)
    items = os.listdir(raw_dir / episode_folder[0])
    camera_names = [
        item
        for item in items
        if os.path.isdir(os.path.join(raw_dir, episode_folder[0], item))
    ]
    for ep_idx, selected_ep_idx in enumerate(ep_ids):
        ep_raw_data = np.load(
            raw_dir / episode_folder[ep_idx] / "robot_info.npy", allow_pickle=True
        )

        imgs_dir = raw_dir / episode_folder[ep_idx] / camera_names[0]
        imgs_timestamps = [
            float(os.path.splitext(i)[0]) for i in sorted(os.listdir(imgs_dir))
        ]
        robot_timstamp = np.array([i["timestamp"] for i in ep_raw_data])


def load_from_raw(
    raw_dir: Path,
    videos_dir: Path,
    fps: int,
    video: bool,
    episodes: list[int] | None = None,
):
    from numcodecs import register_codec
    from imagecodecs.numcodecs import Jpeg2k

    # dataset_alignment(raw_dir)

    # 手动注册 JPEG 2000 编解码器
    register_codec(Jpeg2k)

    episode_folder = os.listdir(raw_dir)
    num_episodes = len(episode_folder)
    ep_dicts = []
    ep_ids = episodes if episodes else range(num_episodes)
    items = os.listdir(raw_dir / episode_folder[0])
    # 筛选处items里的文件夹
    camera_names = [
        item
        for item in items
        if os.path.isdir(os.path.join(raw_dir, episode_folder[0], item))
    ]
    img_keys = [f"observation.images.{key}" for key in camera_names]

    with tqdm.tqdm(
        total=len(ep_ids), desc="Loading image data", mininterval=1.0
    ) as pbar:
        for ep_idx, selected_ep_idx in enumerate(ep_ids):
            # construct ep_dict and appended to ep_dicts
            ep_raw_data = np.load(
                raw_dir / episode_folder[ep_idx] / "robot_info.npy", allow_pickle=True
            )

            key = camera_names[0]
            imgs_dir = raw_dir / episode_folder[ep_idx] / key
            robot_timstamp_idx = check_data_sync(ep_raw_data, imgs_dir, fps)
            ep_raw_data = ep_raw_data[robot_timstamp_idx]
            ep_dict = {}

            origin_state = torch.tensor(
                [
                    item["arm_angles"] + [item["gripper_percentage"]]
                    for item in ep_raw_data
                ],
                dtype=torch.float32,
            )
            num_frames = origin_state.shape[0] - 1
            ep_dict["observation.state"] = origin_state[:-1]
            ep_dict["action"] = origin_state[1:]
            ep_dict["episode_index"] = torch.tensor(
                [ep_idx] * num_frames, dtype=torch.int64
            )
            ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
            ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / fps

            # constract label "observation.image" for ep_dict and save video to video_dir
            key = camera_names[0]
            imgs_dir = raw_dir / episode_folder[ep_idx] / key
            images_name = sorted(imgs_dir.glob("*.jpg"))[:-1]
            if len(images_name) != num_frames:
                print(
                    f"{episode_folder[ep_idx]} contains {len(images_name)} images,but num_frames is {num_frames}"
                )
                continue

            for key in camera_names:
                img_key = f"observation.images.{key}"
                imgs_dir = raw_dir / episode_folder[ep_idx] / key
                images_name = sorted(imgs_dir.glob("*.jpg"))[:-1]
                assert (
                    len(images_name) == num_frames
                ), f"{episode_folder[ep_idx]}_{key} contains {len(images_name)} images,but num_frames is {num_frames}"

                if video:
                    # encode images to a mp4 video
                    fname = f"{img_key}_episode_{ep_idx:06d}.mp4"
                    video_path = videos_dir / fname

                    encode_video_frames(imgs_dir, video_path, fps)
                    # store the reference to the video frame
                    ep_dict[img_key] = [
                        {"path": f"videos/{fname}", "timestamp": i / fps}
                        for i in range(num_frames)
                    ]
                else:
                    ep_dict[img_key] = [
                        PILImage.open(imgs_dir / img) for img in images_name
                    ]

            ep_dicts.append(ep_dict)
            pbar.update(1)

    data_dict = concatenate_episodes(ep_dicts)

    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)
    return data_dict


def to_hf_dataset(data_dict, video):
    features = {}

    if video:
        for key in data_dict.keys():
            if "observation.images" in key:
                features[key] = VideoFrame()
    else:
        for key in data_dict.keys():
            if "observation.images" in key:
                features[key] = Image()

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
        fps = 30

    data_dict = load_from_raw(raw_dir, videos_dir, fps, video, episodes)
    hf_dataset = to_hf_dataset(data_dict, video)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "fps": fps,
        "video": video,
    }
    return hf_dataset, episode_data_index, info
