#!/usr/bin/env python
# convert_lerobot_dataset.py
import argparse
import contextlib
import filecmp
import logging
import math
import subprocess
import tempfile

import json
import shutil
from pathlib import Path
import datasets
from datasets import Dataset
from datasets import Dataset, Features, Value, Image
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import pyarrow.parquet as pq
import re
import pyarrow as pa
import pyarrow.ipc as ipc
from tqdm import tqdm
import torch
from safetensors.torch import load_file
import os

from lerobot.common.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_PARQUET_PATH,
    DEFAULT_VIDEO_PATH,
    EPISODES_PATH,
    INFO_PATH,
    STATS_PATH,
    TASKS_PATH,
    create_branch,
    create_lerobot_dataset_card,
    flatten_dict,
    get_safe_version,
    load_json,
    unflatten_dict,
    write_json,
    write_jsonlines,
)

from lerobot.common.datasets.video_utils import (
    VideoFrame,  # noqa: F401
    get_image_pixel_channels,
    get_video_info,
)
from lerobot.common.robot_devices.robots.configs import RobotConfig
from lerobot.common.robot_devices.robots.utils import make_robot_config

# # 指定 Arrow 文件路径
arrow_file = "/home/h666/code/dataset/hf_dataset/zcai/aloha2/collect_dish_0126_merged_resized6d/train/data-00000-of-00001.arrow"
output_parquet = "/home/h666/code/dataset/hf_dataset/zcai/aloha2/collect_dish_0126_merged_resized6d_test/test.parquet"
single_task = "collect_dish"
episode_lengths = []
robot_type = "unknown"
dataset = Dataset.from_parquet(str(output_parquet))
# 打开 Arrow 文件并读取整个表
# with open(arrow_file, "rb") as f:
#     reader = ipc.open_stream(f)
#     table = reader.read_all()
# pq.write_table(table, output_parquet)
# 输出 Arrow Table 的基本信息
# print("Arrow Table 信息:")
# print(table)

# with open("schema.txt", "w") as f:
#     f.write(str(table.schema))
# print("已将 schema 写入 schema.txt，可使用编辑器查看")
# print("Arrow schema:", table.schema)
# # 可选：转换为 Pandas DataFrame 进行更直观的查看
# df = table.to_pandas()
# df.to_csv("merged_output.csv", index=False)
# print("数据预览:")
# print(df.head())

# 常量定义 (与LeRobot v2.0格式一致)
V16 = "v1.6"
V20 = "v2.0"
v1x_dir = Path("/home/h666/code/dataset/hf_dataset/zcai/aloha2/collect_dish_0126_merged_resized6d")
v20_dir = Path("/home/h666/code/dataset/hf_dataset/zcai/aloha2/collect_dish_0126_merged_resized6d_test_2")

GITATTRIBUTES_REF = "aliberts/gitattributes_reference"
V1_VIDEO_FILE = "{video_key}_episode_{episode_index:06d}.mp4"
V1_INFO_PATH = "meta_data/info.json"
V1_STATS_PATH = "meta_data/stats.safetensors"
DEFAULT_CHUNK_SIZE = 1000  # 每个chunk包含的episode数量
DEFAULT_PARQUET_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
DEFAULT_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

def convert_arrow_to_v2(input_arrow: Path, output_dir: Path):
    """将Arrow文件转换为v2.0格式"""
    print(f"Loading Arrow file: {input_arrow}")
    # dataset = Dataset.from_file(str(input_arrow))
    
    
    # 获取所有唯一的episode索引
    episode_indices = sorted(set(dataset["episode_index"]))
    total_episodes = len(episode_indices)
    print(f"Found {total_episodes} episodes")
    
    # 创建输出目录结构
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)
    
    # 提取视频键名 (从特征中获取)
    video_keys = [col for col in dataset.column_names if col.startswith("observation.images.")]
    print(f"Found video keys: {video_keys}")
    
    # 分割数据为Parquet文件
    for chunk_idx, start_idx in enumerate(range(0, total_episodes, DEFAULT_CHUNK_SIZE)):
        # chunk_dir = output_dir / f"data/chunk-{chunk_idx:03d}"
        # chunk_dir.mkdir(parents=True, exist_ok=True)
        
        end_idx = min(start_idx + DEFAULT_CHUNK_SIZE, total_episodes)
        # print(f"Processing chunk {chunk_idx}: episodes {start_idx}-{end_idx-1}")
        
        for ep_idx in tqdm(episode_indices[start_idx:end_idx], desc=f"Chunk {chunk_idx}"):
            # 提取单个episode数据
            episode_data = dataset.filter(lambda x: x["episode_index"] == ep_idx)
            episode_lengths.insert(ep_idx, len(episode_data))
            # 写入Parquet文件
            # output_path = chunk_dir / f"episode_{ep_idx:06d}.parquet"
            # pq.write_table(episode_data.data.table, output_path)



def reorganize_videos(videos_dir: Path, output_dir: Path):
    """重组视频文件结构"""
    print("Reorganizing video files...")
    video_files = list(videos_dir.glob("*.mp4"))
    
    for video_path in video_files:
        # 解析视频文件名 (格式: observation.images.cam_right_wrist_episode_000291.mp4)
        match = re.match(r"(.+?)_episode_(\d+)\.mp4", video_path.name)
        if not match:
            print(f"Skipping invalid video filename: {video_path.name}")
            continue
            
        video_key = match.group(1)  # 如 "observation.images.cam_right_wrist"
        ep_idx = int(match.group(2))
        chunk_idx = ep_idx // DEFAULT_CHUNK_SIZE
        
        # 创建目标目录
        target_dir = output_dir / f"videos/chunk-{chunk_idx:03d}/{video_key}"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制并重命名文件
        target_path = target_dir / f"episode_{ep_idx:06d}.mp4"
        shutil.copy2(video_path, target_path)
        print(f"Reorganized: {video_path.name} -> {target_path.relative_to(output_dir)}")

def convert_stats_to_json(v1_dir: Path, v2_dir: Path) -> None:
    safetensor_path = v1_dir / V1_STATS_PATH
    stats = load_file(safetensor_path)
    serialized_stats = {key: value.tolist() for key, value in stats.items()}
    serialized_stats = unflatten_dict(serialized_stats)

    json_path = v2_dir / STATS_PATH
    json_path.parent.mkdir(exist_ok=True, parents=True)
    with open(json_path, "w") as f:
        json.dump(serialized_stats, f, indent=4)

    # Sanity check
    with open(json_path) as f:
        stats_json = json.load(f)

    stats_json = flatten_dict(stats_json)
    stats_json = {key: torch.tensor(value) for key, value in stats_json.items()}
    for key in stats:
        torch.testing.assert_close(stats_json[key], stats[key])

def get_features_from_hf_dataset(
    dataset: Dataset, robot_config: RobotConfig | None = None
) -> dict[str, list]:
    robot_config = None
    features = {}
    for key, ft in dataset.features.items():
        if isinstance(ft, datasets.Value):
            dtype = ft.dtype
            shape = (1,)
            names = None
        if isinstance(ft, datasets.Sequence):
            assert isinstance(ft.feature, datasets.Value)
            dtype = ft.feature.dtype
            shape = (ft.length,)
            motor_names = (
                robot_config["names"][key] if robot_config else [f"motor_{i}" for i in range(ft.length)]
            )
            assert len(motor_names) == shape[0]
            names = {"motors": motor_names}
        elif isinstance(ft, datasets.Image):
            dtype = "image"
            image = dataset[0][key]  # Assuming first row
            channels = get_image_pixel_channels(image)
            shape = (image.height, image.width, channels)
            names = ["height", "width", "channels"]
        elif ft._type == "VideoFrame":
            dtype = "video"
            shape = None  # Add shape later
            names = ["height", "width", "channels"]

        features[key] = {
            "dtype": dtype,
            "shape": shape,
            "names": names,
        }

    return features

def add_task_index_by_episodes(dataset: Dataset, tasks_by_episodes: dict) -> tuple[Dataset, list[str]]:
    df = dataset.to_pandas()
    tasks = list(set(tasks_by_episodes.values()))
    tasks_to_task_index = {task: task_idx for task_idx, task in enumerate(tasks)}
    episodes_to_task_index = {ep_idx: tasks_to_task_index[task] for ep_idx, task in tasks_by_episodes.items()}
    df["task_index"] = df["episode_index"].map(episodes_to_task_index).astype(int)

    features = dataset.features
    features["task_index"] = datasets.Value(dtype="int64")
    dataset = Dataset.from_pandas(df, features=features, split="train")
    return dataset, tasks

def convert_metadata(meta_dir: Path, output_dir: Path):
    """转换元数据文件"""
    print("Converting metadata...")
    output_meta_dir = output_dir / "meta"
    output_meta_dir.mkdir(exist_ok=True)
    features = get_features_from_hf_dataset(dataset, None)
    video_keys = [key for key, ft in features.items() if ft["dtype"] == "video"]
    # Episodes & chunks
    episode_indices = sorted(dataset.unique("episode_index"))
    total_episodes = len(episode_indices)

    assert episode_indices == list(range(total_episodes))
    total_videos = total_episodes * len(video_keys)
    total_chunks = total_episodes // DEFAULT_CHUNK_SIZE
    if total_episodes % DEFAULT_CHUNK_SIZE != 0:
        total_chunks += 1

    tasks_by_episodes = {ep_idx: single_task for ep_idx in episode_indices}
    new_dataset, tasks = add_task_index_by_episodes(dataset, tasks_by_episodes)
    tasks_by_episodes = {ep_idx: [task] for ep_idx, task in tasks_by_episodes.items()}
    # Tasks
    # if single_task:
    #     tasks_by_episodes = {ep_idx: single_task for ep_idx in episode_indices}
    #     dataset, tasks = add_task_index_by_episodes(dataset, tasks_by_episodes)
    #     tasks_by_episodes = {ep_idx: [task] for ep_idx, task in tasks_by_episodes.items()}
    # elif tasks_path:
    #     tasks_by_episodes = load_json(tasks_path)
    #     tasks_by_episodes = {int(ep_idx): task for ep_idx, task in tasks_by_episodes.items()}
    #     dataset, tasks = add_task_index_by_episodes(dataset, tasks_by_episodes)
    #     tasks_by_episodes = {ep_idx: [task] for ep_idx, task in tasks_by_episodes.items()}
    # elif tasks_col:
    #     dataset, tasks, tasks_by_episodes = add_task_index_from_tasks_col(dataset, tasks_col)
    # else:
    #     raise ValueError

    assert set(tasks) == {task for ep_tasks in tasks_by_episodes.values() for task in ep_tasks}
    tasks = [{"task_index": task_idx, "task": task} for task_idx, task in enumerate(tasks)]
    write_jsonlines(tasks, v20_dir / TASKS_PATH)
    features["task_index"] = {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    }

    # Episodes
    episodes = [
        {"episode_index": ep_idx, "tasks": tasks_by_episodes[ep_idx], "length": episode_lengths[ep_idx]}
        for ep_idx in episode_indices
    ]
    write_jsonlines(episodes, v20_dir / EPISODES_PATH)

    metadata_v1 = load_json(v1x_dir / V1_INFO_PATH)
    # Assemble metadata v2.0
    metadata_v2_0 = {
        "codebase_version": V20,
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": len(new_dataset),
        "total_tasks": len(tasks),
        "total_videos": total_videos,
        "total_chunks": total_chunks,
        "chunks_size": DEFAULT_CHUNK_SIZE,
        "fps": metadata_v1["fps"],
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": DEFAULT_PARQUET_PATH,
        "video_path": DEFAULT_VIDEO_PATH if video_keys else None,
        "features": features,
    }
    write_json(metadata_v2_0, v20_dir / INFO_PATH)
    convert_stats_to_json(v1x_dir, v20_dir)
    
    # 转换info.json (示例，需根据实际内容调整)
    # with open(meta_dir / "info.json") as f:
    #     old_info = json.load(f)
    
    # new_info = {
    #     "codebase_version": "2.0",
    #     "fps": old_info.get("fps", 30),
    #     "video": True,
    #     # 其他需要保留的字段...
    # }
    # with open(output_meta_dir / "info.json", "w") as f:
    #     json.dump(new_info, f, indent=4)
    
    # # 创建空的元数据文件 (实际使用时需填充内容)
    # for empty_file in ["episodes.jsonl", "stats.json", "tasks.jsonl"]:
    #     (output_meta_dir / empty_file).touch()

def main(input_dir: Path, output_dir: Path):
    """主转换函数"""
    print(f"Converting dataset from {input_dir} to {output_dir}")
    # output_dir.mkdir(parents=True, exist_ok=True)
    
    # # 1. 处理arrow文件
    arrow_file = next(input_dir.glob("train/data-*.arrow"))
    # print(str(arrow_file))
    convert_arrow_to_v2(arrow_file, output_dir)
    
    # # 2. 处理视频文件
    # videos_dir = input_dir / "videos"
    # reorganize_videos(videos_dir, output_dir)
    
    # 3. 处理元数据
    meta_dir = input_dir / "meta_data"
    convert_metadata(meta_dir, output_dir)
    
    print(f"Conversion complete! Output saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True, 
                       help="Path to input dataset directory (containing train/, videos/, meta_data/)")
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="Path to output converted dataset")
    args = parser.parse_args()
    
    main(args.input_dir, args.output_dir)