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

import argparse
import math
import json
import numpy as np
import torch
from pathlib import Path
from typing import Optional

# 新增导入
from lerobot.common.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_PARQUET_PATH,
    DEFAULT_VIDEO_PATH,
    EPISODES_PATH,
    INFO_PATH,
    STATS_PATH,
    TASKS_PATH,
    write_jsonlines,
    write_json,
    calculate_episode_data_index,
)
from lerobot.common.robot_devices.robots.utils import make_robot_config
from lerobot.common.datasets.v2.convert_dataset_v1_to_v2 import (
    split_parquet_by_episodes,
    convert_stats_to_json,
    parse_robot_config,
)

# 原test2内容保持不变...
# [保留原有的hdf5加载、视频处理等核心逻辑]
# 只展示关键修改部分

def from_raw_to_lerobot_format(
    raw_dir: Path,
    output_dir: Path,
    fps: int,
    video: bool = True,
    robot_config: Optional[dict] = None,
    single_task: Optional[str] = None,
    tasks_col: Optional[str] = None,
    tasks_path: Optional[Path] = None,
):
    """核心转换函数"""
    # 1. 加载原始数据
    data_dict, episode_data_index, info = load_raw_data(raw_dir, fps, video)
    
    # 2. 转换为HuggingFace Dataset
    hf_dataset = to_hf_dataset(data_dict, video)
    
    # 3. 生成v2.0目录结构
    v20_dir = output_dir / "v2.0" / "custom_dataset"
    v20_dir.mkdir(parents=True, exist_ok=True)

    # 4. 处理任务元数据
    if single_task:
        tasks = [{"task_index": 0, "task": single_task}]
        tasks_by_episodes = {ep_idx: [single_task] for ep_idx in range(len(episode_data_index))}
    elif tasks_col:
        tasks = list(hf_dataset.unique(tasks_col))
        tasks_by_episodes = hf_dataset.to_pandas().groupby("episode_index")[tasks_col].unique().to_dict()
    write_jsonlines(tasks, v20_dir / TASKS_PATH)

    # 5. 分块存储Parquet
    total_episodes = len(episode_data_index)
    total_chunks = math.ceil(total_episodes / DEFAULT_CHUNK_SIZE)
    episode_lengths = split_parquet_by_episodes(
        hf_dataset,
        total_episodes,
        total_chunks,
        v20_dir
    )

    # 6. 生成episodes.jsonl
    episodes = [
        {
            "episode_index": ep_idx,
            "tasks": tasks_by_episodes[ep_idx],
            "length": episode_lengths[ep_idx]
        }
        for ep_idx in range(total_episodes)
    ]
    write_jsonlines(episodes, v20_dir / EPISODES_PATH)

    # 7. 生成info.json
    features = {}
    if robot_config:
        parsed_config = parse_robot_config(robot_config)
        for key in ["observation.state", "action"]:
            features[key] = {
                "dtype": "float32",
                "shape": (len(parsed_config["names"][key]),),
                "names": parsed_config["names"][key]
            }
    # 添加其他特征字段...

    metadata = {
        "codebase_version": "2.0",
        "robot_type": robot_config.type if robot_config else "unknown",
        "total_episodes": total_episodes,
        "fps": fps,
        "data_path": DEFAULT_PARQUET_PATH,
        "video_path": DEFAULT_VIDEO_PATH if video else None,
        "features": features,
        # 其他元数据字段...
    }
    write_json(metadata, v20_dir / INFO_PATH)

    # 8. 处理视频分块
    if video:
        video_keys = [f"observation.images.{cam}" for cam in get_cameras(raw_dir)]
        for vid_key in video_keys:
            move_videos(
                repo_id="custom_dataset",
                video_keys=[vid_key],
                total_episodes=total_episodes,
                total_chunks=total_chunks,
                work_dir=v20_dir,
                clean_gitattributes=Path("path/to/gitattributes_reference"),
            )

    # 9. 生成统计信息
    convert_stats_to_json(raw_dir, v20_dir)

    return v20_dir

def main():
    parser = argparse.ArgumentParser(description='Convert raw data to LeRobot v2.0 format')
    parser.add_argument('--raw-dir', type=Path, required=True, help='Path to raw data directory')
    parser.add_argument('--output-dir', type=Path, required=True, help='Output directory')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--robot-config', type=str, help='Robot type (e.g. aloha, koch)')
    parser.add_argument('--single-task', type=str, help='Single task description')
    parser.add_argument('--tasks-col', type=str, help='Column name containing tasks')
    parser.add_argument('--no-video', action='store_false', dest='video', help='Disable video processing')
    
    args = parser.parse_args()

    # 处理机器人配置
    robot_config = make_robot_config(args.robot_config) if args.robot_config else None

    # 执行转换
    output_path = from_raw_to_lerobot_format(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        fps=args.fps,
        video=args.video,
        robot_config=robot_config,
        single_task=args.single_task,
        tasks_col=args.tasks_col,
    )

    print(f"Dataset converted to v2.0 format at: {output_path}")

if __name__ == "__main__":
    main()