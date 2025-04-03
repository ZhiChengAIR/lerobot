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
from scipy.spatial.transform import Rotation as R

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

from lerobot.common.rotation_transformer import RotationTransformer  # added by yz


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

# Initialize the RotationTransformer for converting from 'euler_angles' to 'rotation_6d'
rotation_transformer = RotationTransformer(from_rep='euler_angles', to_rep='rotation_6d', from_convention = "XYZ")

def load_from_raw(
    raw_dir: Path,
    videos_dir: Path,
    fps: int,
    video: bool,
    episodes: list[int] | None = None,
    rotation_transformer: RotationTransformer = None,  # Accept the transformer as a parameter
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
            assert (
                v == "4.0"
            ), f"ZCAI_DATASET_VERSION version {ZCAI_DATASET_VERSION} is not fit for this code version {v},"
            f_fps = ep.attrs["fps"]
            assert f_fps == fps, f"fps {fps} is not equal to fps in HDF5 {f_fps}"

            num_frames = ep["data"]["left_pose"]["pose"].shape[0]

            # last step of demonstration is considered done
            done = torch.zeros(num_frames, dtype=torch.bool)
            done[-1] = True

            try:
                left_action = np.concatenate((ep["data"]["left_pose_follow_cmd"]["pose"][:], ep["data"]["left_gripper_ctrl_cmd"]["pos"][:], ep["data"]["left_gripper_ctrl_cmd"]["vel"][:], ep["data"]["left_gripper_ctrl_cmd"]["eff"][:]), axis=1)
            except:
                print("shape of left_pose_follow_cmd is ", ep["data"]["left_pose_follow_cmd"]["pose"].shape)
                print("shape of left_gripper_ctrl_cmd is ", ep["data"]["left_gripper_ctrl_cmd"]["pos"].shape)
                print("shape of left_gripper_ctrl_cmd is ", ep["data"]["left_gripper_ctrl_cmd"]["vel"].shape)
                print("shape of left_gripper_ctrl_cmd is ", ep["data"]["left_gripper_ctrl_cmd"]["eff"].shape)
                # terminate the program
                exit(1)
            right_action = np.concatenate((ep["data"]["right_pose_follow_cmd"]["pose"][:], ep["data"]["right_gripper_ctrl_cmd"]["pos"][:], ep["data"]["right_gripper_ctrl_cmd"]["vel"][:], ep["data"]["right_gripper_ctrl_cmd"]["eff"][:]), axis=1)
            action = np.concatenate((left_action, right_action), axis=1)

            obs_gripper_left = np.concatenate((action[:,7:10][:1], action[:,7:10][:-1]), axis=0)
            obs_gripper_right = np.concatenate((action[:,17:20][:1], action[:,17:20][:-1]), axis=0)
            # state = np.concatenate((ep["data"]["puppet_left"]["qpos"][:],obs_gripper_left,ep["data"]["puppet_right"]["qpos"][:],obs_gripper_right), axis=1)
           # pose = np.concatenate((ep["data"]["puppet_left"]["tcp_pose"][:],obs_gripper_left,ep["data"]["puppet_right"]["tcp_pose"][:],obs_gripper_right), axis=1)

            left_arm_state = np.concatenate((ep["data"]["left_joint_state"]["pos"], ep["data"]["left_joint_state"]["vel"], ep["data"]["left_joint_state"]["eff"], obs_gripper_left), axis=1)
            right_arm_state = np.concatenate((ep["data"]["right_joint_state"]["pos"], ep["data"]["right_joint_state"]["vel"], ep["data"]["right_joint_state"]["eff"], obs_gripper_right), axis=1)
            state = np.concatenate((left_arm_state, right_arm_state), axis=1)

            pose = np.concatenate((ep["data"]["left_pose"]["pose"], obs_gripper_left, ep["data"]["right_pose"]["pose"], obs_gripper_right), axis=1)
            
            # qtor = np.concatenate((ep["data"]["puppet_left"]["qtor"][:], ep["data"]["puppet_right"]["qtor"][:]), axis=1)
            # qvel = np.concatenate((ep["data"]["puppet_left"]["qvel"][:], ep["data"]["puppet_right"]["qvel"][:]), axis=1)
            # qacc = np.concatenate((ep["data"]["puppet_left"]["qacc"][:], ep["data"]["puppet_right"]["qacc"][:]), axis=1)
            # tcpvel = np.concatenate((ep["data"]["puppet_left"]["tcp_vel"][:],ep["data"]["puppet_right"]["tcp_vel"][:]), axis=1)

            pos = np.concatenate((ep["data"]["left_joint_state"]["pos"], ep["data"]["right_joint_state"]["pos"]), axis=1)
            vel = np.concatenate((ep["data"]["left_joint_state"]["vel"], ep["data"]["right_joint_state"]["vel"]), axis=1)
            eff = np.concatenate((ep["data"]["left_joint_state"]["eff"], ep["data"]["right_joint_state"]["eff"]), axis=1)
            
            assert num_frames == action.shape[0]
            # assert num_frames == action_tcp.shape[0]
            assert num_frames == state.shape[0]
            assert num_frames == pose.shape[0]
            assert num_frames == pos.shape[0]
            assert num_frames == vel.shape[0]
            assert num_frames == eff.shape[0]
            # assert num_frames == tcpvel.shape[0]
            

            action = torch.from_numpy(action)
            # action_tcp = torch.from_numpy(action_tcp)
            # target_tcp_pos = torch.from_numpy(target_tcp_pos)
            state = torch.from_numpy(state)
            pose = torch.from_numpy(pose)
            pos = torch.from_numpy(pos)
            vel = torch.from_numpy(vel)
            eff = torch.from_numpy(eff)
            # tcpvel = torch.from_numpy(tcpvel)


            # --- added by yz: Convert specific parts of 'action_tcp' from 'euler_angles' to 'rotation_6d' ---

            # Convert 'action_tcp' to NumPy for processing
            # action_tcp_np = action_tcp.numpy()  # Shape: [num_frames, action_dim]

            # Extract columns 3-5 and 10-12 (0-based indexing: 3:6 and 10:13)
            # euler_angles_arm1 = action_tcp_np[:, 3:6]/180*np.pi   # Shape: [num_frames, 3]
            # euler_angles_arm2 = action_tcp_np[:, 10:13]/180*np.pi # Shape: [num_frames, 3]
            
            action = action.numpy()
            quat_left_arm = action[:, 3:7]
            quat_right_arm = action[:, 13:17]
            eular_left_arm = quat2euler(quat_left_arm)
            eular_right_arm = quat2euler(quat_right_arm)
            # Apply the RotationTransformer to convert to 'rotation_6d'
            # rotation_6d_arm1 = rotation_transformer.forward(euler_angles_arm1)  # Shape: [num_frames, 6]
            # rotation_6d_arm2 = rotation_transformer.forward(euler_angles_arm2)

            rotation_6d_left_arm = rotation_transformer.forward(eular_left_arm)
            rotation_6d_right_arm = rotation_transformer.forward(eular_right_arm)

            # # Replace the original euler_angles entries with the 'rotation_6d' data
            # # To accommodate the increased dimensionality, we'll expand the 'action_tcp' tensor
            # # Insert 'rotation_6d' for arm1 at position 3, shifting existing entries
            # # Similarly, insert 'rotation_6d' for arm2 at position 10 (adjusted for previous insertion)

            # # Split the original 'action_tcp' into parts
            # # Before arm1
            # action_tcp_before_arm1 = action_tcp_np[:, :3]  # [num_frames, 3]
            # # Between arm1 and arm2
            # action_tcp_between_arms = action_tcp_np[:, 6:10]  # [num_frames, 4]
            # # After arm2
            # action_tcp_after_arm2 = action_tcp_np[:, 13:]  # Remaining columns

            # # Create new 'action_tcp' with 'rotation_6d' for both arms
            # # arm1: replace 3-5 with 6-dim rotation_6d
            # # arm2: replace 10-12 with 6-dim rotation_6d
            # # Total new columns: 3 (before) + 6 (arm1) + 4 (between) + 6 (arm2) + remaining

            # action_tcp_6d_np = np.hstack((
            #     action_tcp_before_arm1,        # [num_frames, 3]
            #     rotation_6d_arm1,            # [num_frames, 6] for arm1
            #     action_tcp_between_arms,       # [num_frames, 4]
            #     rotation_6d_arm2,            # [num_frames, 6] for arm2
            #     action_tcp_after_arm2           # [num_frames, remaining]
            # ))  # Final shape: [num_frames, original_dim + 6]

            # Convert back to torch.Tensor
            # action_tcp_6d = torch.from_numpy(action_tcp_6d_np)  # [num_frames, new_action_dim]

            action_6d_np = np.hstack((
                action[:, :3],       
                rotation_6d_left_arm,            
                action[:, 7:8],
                action[:, 10:13],
                rotation_6d_right_arm,            
                action[:, 17:18]
            ))

            action_6d = torch.from_numpy(action_6d_np)

            # # --- added by yz: Convert specific parts of 'tcppose' from 'euler_angles' to 'rotation_6d' ---

            # # Convert 'tcppose' to NumPy for processing
            # tcppose_np = tcppose.numpy()  # Shape: [num_frames, action_dim]

            



            # # Extract columns 3-5 and 10-12 (0-based indexing: 3:6 and 10:13)
            # euler_angles_arm1 = tcppose_np[:, 3:6]/180*np.pi   # Shape: [num_frames, 3]
            # euler_angles_arm2 = tcppose_np[:, 10:13]/180*np.pi # Shape: [num_frames, 3]

            # # Apply the RotationTransformer to convert to 'rotation_6d'
            # rotation_6d_arm1 = rotation_transformer.forward(euler_angles_arm1)  # Shape: [num_frames, 6]
            # rotation_6d_arm2 = rotation_transformer.forward(euler_angles_arm2)


            # # Replace the original euler_angles entries with the 'rotation_6d' data
            # # To accommodate the increased dimensionality, we'll expand the 'tcppose' tensor
            # # Insert 'rotation_6d' for arm1 at position 3, shifting existing entries
            # # Similarly, insert 'rotation_6d' for arm2 at position 10 (adjusted for previous insertion)

            # # Split the original 'tcppose' into parts
            # # Before arm1
            # tcppose_before_arm1 = tcppose_np[:, :3]  # [num_frames, 3]
            # # Between arm1 and arm2
            # tcppose_between_arms = tcppose_np[:, 6:10]  # [num_frames, 4]
            # # After arm2
            # tcppose_after_arm2 = tcppose_np[:, 13:]  # Remaining columns

            # # Create new 'tcppose' with 'rotation_6d' for both arms
            # # arm1: replace 3-5 with 6-dim rotation_6d
            # # arm2: replace 10-12 with 6-dim rotation_6d
            # # Total new columns: 3 (before) + 6 (arm1) + 4 (between) + 6 (arm2) + remaining

            # tcppose_6d_np = np.hstack(
            #     (
            #         tcppose_before_arm1,  # [num_frames, 3]
            #         rotation_6d_arm1,  # [num_frames, 6] for arm1
            #         tcppose_between_arms,  # [num_frames, 4]
            #         rotation_6d_arm2,  # [num_frames, 6] for arm2
            #         tcppose_after_arm2,  # [num_frames, remaining]
            #     )
            # )  # Final shape: [num_frames, original_dim + 6]

            # # Convert back to torch.Tensor
            # tcppose_6d = torch.from_numpy(tcppose_6d_np)  # [num_frames, new_action_dim]

            pose = pose.numpy()
            
            quat_left_arm = pose[:, 3:7]
            quat_right_arm = pose[:, 13:17]
            eular_left_arm = quat2euler(quat_left_arm)
            eular_right_arm = quat2euler(quat_right_arm)

            rotation_6d_left_arm = rotation_transformer.forward(eular_left_arm)
            rotation_6d_right_arm = rotation_transformer.forward(eular_right_arm)

            pose_6d_np = np.hstack((
                pose[:, :3],       
                rotation_6d_left_arm,
                pose[:, 7:8],
                pose[:, 10:13],
                rotation_6d_right_arm,
                pose[:, 17:18]            
            ))

            pose_6d = torch.from_numpy(pose_6d_np)

            # --- End Modification ---

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
                        "-vf scale=320:240 "
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
                    imgs_array = get_imgs_from_video(
                        str(raw_dir / f"episode_{ep_idx}_{camera}.mp4")
                    )
                    ep_dict[img_key] = [PILImage.fromarray(x) for x in imgs_array]

            # ep_dict["observation.state"] = state
            # ep_dict["observation.pos"] = pos
            # ep_dict["observation.vel"] = vel
            # ep_dict["observation.eff"] = eff
            ep_dict["observation.state"] = pose_6d
            # ep_dict["observation.tcpvel"] = tcpvel
            ep_dict["action"] = action_6d
            # ep_dict["action_tcp"] = action_tcp_6d
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

    features["observation.state"] = Sequence(
        length=data_dict["observation.state"].shape[1],
        feature=Value(dtype="float32", id=None),
    )
    # features["observation.pos"] = Sequence(
    #     length=data_dict["observation.pos"].shape[1],
    #     feature=Value(dtype="float32", id=None),
    # )
    # features["observation.vel"] = Sequence(
    #     length=data_dict["observation.vel"].shape[1],
    #     feature=Value(dtype="float32", id=None),
    # )
    # features["observation.eff"] = Sequence(
    #     length=data_dict["observation.eff"].shape[1],
    #     feature=Value(dtype="float32", id=None),
    # )
    # features["observation.state"] = Sequence(
    #     length=data_dict["observation.state"].shape[1],
    #     feature=Value(dtype="float32", id=None),
    # )
    # features["observation.tcpvel"] = Sequence(
    #     length=data_dict["observation.tcpvel"].shape[1],
    #     feature=Value(dtype="float32", id=None),
    # )

    features["action"] = Sequence(
        length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)
    )
    # features["action_tcp"] = Sequence(
    #     length=data_dict["action_tcp"].shape[1], feature=Value(dtype="float32", id=None)
    # )
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
    # check_format(raw_dir)

    if fps is None:
        fps = 30

    data_dict = load_from_raw(raw_dir, videos_dir, fps, video, episodes, rotation_transformer)
    hf_dataset = to_hf_dataset(data_dict, video)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "fps": fps,
        "video": video,
    }
    return hf_dataset, episode_data_index, info

def quat2euler(q):
    # q = [x, y, z, w]
    r = R.from_quat(q)
    euler = r.as_euler('xyz', degrees=False)
    return euler