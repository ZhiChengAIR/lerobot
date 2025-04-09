#!/usr/bin/env python
# convert_lerobot_dataset.py
import json
import shutil
from pathlib import Path
from datasets import Dataset
from datasets import Dataset, Features, Value, Image
import pyarrow.parquet as pq
import re
import pyarrow as pa
import pyarrow.ipc as ipc

# # 指定 Arrow 文件路径
# arrow_file = "/home/h666/code/dataset/hf_dataset/zcai/aloha2/collect_dish_0126_merged_resized6d/train/data-00000-of-00001.arrow"

# # 打开 Arrow 文件并读取整个表
# with open(arrow_file, "rb") as f:
#     reader = ipc.open_stream(f)
#     table = reader.read_all()

# # 输出 Arrow Table 的基本信息
# print("Arrow Table 信息:")
# print(table)

# # 可选：转换为 Pandas DataFrame 进行更直观的查看
# df = table.to_pandas()
# print("数据预览:")
# print(df.head())

# 常量定义 (与LeRobot v2.0格式一致)
DEFAULT_CHUNK_SIZE = 1000  # 每个chunk包含的episode数量
DEFAULT_PARQUET_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
DEFAULT_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

def convert_arrow_to_v2(input_arrow: Path, output_dir: Path):
    """将Arrow文件转换为v2.0格式"""
    print(f"Loading Arrow file: {input_arrow}")
    dataset = Dataset.from_file(str(input_arrow))
    
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
        chunk_dir = output_dir / f"data/chunk-{chunk_idx:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        
        end_idx = min(start_idx + DEFAULT_CHUNK_SIZE, total_episodes)
        print(f"Processing chunk {chunk_idx}: episodes {start_idx}-{end_idx-1}")
        
        for ep_idx in tqdm(episode_indices[start_idx:end_idx], desc=f"Chunk {chunk_idx}"):
            # 提取单个episode数据
            episode_data = dataset.filter(lambda x: x["episode_index"] == ep_idx)
            
            # 写入Parquet文件
            output_path = chunk_dir / f"episode_{ep_idx:06d}.parquet"
            pq.write_table(episode_data.data.table, output_path)

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

def convert_metadata(meta_dir: Path, output_dir: Path):
    """转换元数据文件"""
    print("Converting metadata...")
    output_meta_dir = output_dir / "meta"
    output_meta_dir.mkdir(exist_ok=True)
    
    # 转换info.json (示例，需根据实际内容调整)
    with open(meta_dir / "info.json") as f:
        old_info = json.load(f)
    
    new_info = {
        "codebase_version": "2.0",
        "fps": old_info.get("fps", 30),
        "video": True,
        # 其他需要保留的字段...
    }
    with open(output_meta_dir / "info.json", "w") as f:
        json.dump(new_info, f, indent=4)
    
    # 创建空的元数据文件 (实际使用时需填充内容)
    for empty_file in ["episodes.jsonl", "stats.json", "tasks.jsonl"]:
        (output_meta_dir / empty_file).touch()

def main(input_dir: Path, output_dir: Path):
    """主转换函数"""
    print(f"Converting dataset from {input_dir} to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 处理arrow文件
    arrow_file = next(input_dir.glob("train/data-*.arrow"))
    print(str(arrow_file))
    convert_arrow_to_v2(arrow_file, output_dir)
    
    # 2. 处理视频文件
    videos_dir = input_dir / "videos"
    reorganize_videos(videos_dir, output_dir)
    
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