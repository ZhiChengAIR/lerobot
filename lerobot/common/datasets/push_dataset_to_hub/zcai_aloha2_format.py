#!/usr/bin/env python

from pathlib import Path
import os


def load_from_raw(
    raw_dir: Path,
    videos_dir: Path,
    fps: int,
    video: bool,
    episodes: list[int] | None = None,
):
    from numcodecs import register_codec
    from imagecodecs.numcodecs import Jpeg2k

    # 手动注册 JPEG 2000 编解码器
    register_codec(Jpeg2k)

    episode_folder = os.listdir(raw_dir)
    num_episodes = len(episode_folder)
    ep_dicts = []
    ep_ids = episodes if episodes else range(num_episodes)
    items = os.listdir(os.path.join(raw_dir, episode_folder[0]))
    # 筛选处items里的文件夹
    camera_names = [
        item
        for item in items
        if os.path.isdir(os.path.join(raw_dir, episode_folder[0], item))
    ]
    img_keys = [f"observation.images.{key}" for key in camera_names]


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
    # hf_dataset = to_hf_dataset(data_dict, video)
    # episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "fps": fps,
        "video": video,
    }
    return hf_dataset, episode_data_index, info
