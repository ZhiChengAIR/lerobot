# 目录

- [目录](#目录)
- [1. aloha2-3090 服务器训练步骤](#1-aloha2-3090-服务器训练步骤)
  - [1.1. 数据集转换](#11-数据集转换)
  - [1.2. 训练模型](#12-训练模型)
    - [1.2.1. 初次训练配置](#121-初次训练配置)
    - [1.2.2. 恢复训练配置](#122-恢复训练配置)
    - [1.2.2. 启动训练](#122-启动训练)
- [2. 4090 服务器训练步骤](#2-4090-服务器训练步骤)
  - [2.1. 数据集转换](#21-数据集转换)
  - [2.2. 训练模型](#22-训练模型)
    - [2.2.1. 查看空闲 cpu 位置](#221-查看空闲-cpu-位置)
    - [2.2.2. 初次训练配置](#222-初次训练配置)
    - [2.2.3. 恢复训练配置](#223-恢复训练配置)
    - [2.2.2. 启动训练](#222-启动训练)
- [3. 结束训练](#3-结束训练)

# 1. aloha2-3090 服务器训练步骤

## 1.1. 数据集转换

双臂遥操采集的数据格式是 zcai_aloha2 数据格式，需要转换为 hf_dataset 才能被 lerobot 框架利用训练。执行以下指令转换数据集。

```
# 将zcai_aloha2采集的数据转化为hf_dataset,假设数据集文件夹为cloth_0909_0，一般将repo-id，raw-dir，local-dir修改为新数据集名字即可
/home/h666/miniforge3/envs/lerobot/bin/python ~/code/ZCAI-ROBOT/dependencies/lerobot/lerobot/scripts/push_dataset_to_hub.py \
--repo-id aloha2/stack_cup_1025_1 \
--raw-dir /home/h666/code/dataset/raw/stack_cup_1025_1 \
--local-dir /home/h666/code/dataset/hf_dataset/zcai/aloha2/stack_cup_1025_1 \
--raw-format zcai_aloha2 \
--push-to-hub 0 \
--force-override 0 \
--video 1 \
--fps 10
```

参数解释

```
--repo-id 训练时的数据集名字，需要与raw-dir中最后一项数据集名字保持一致
--local-dir zcai_aloha2格式数据集所在位置，也就是原始数据集所在位置，一般改最后一项即可
--local-dir 数据集转换后输出文件夹位置，一般改最后一项即可。需要与raw-dir中最后一项数据集名字保持一致
--raw-format 原始数据集格式名字，一般不用改
--push-to-hub 是否需要推送至huggingface，设置为0
--force-override local-dir文件夹如果存在的话是否需要覆写，不用改
--video 输出的数据集图片是以视频保存，还是打包放置在一起。这关系到训练的时候图片文件是否一次性加载到内存里。0，代表训练的时候会一次性加载训练所有图片。1代表，需要用到的时候再通过视频加载相应图片。这里不用改
--fps 采集数据时的频率，这里不用改
```

## 1.2. 训练模型

### 1.2.1. 初次训练配置

确认一下文件`~/code/ZCAI-ROBOT/dependencies/lerobot/lerobot/scripts/train.py`第 517 行是否为正确的配置名字。目前配置名字可选项有`zcai_aloha2_act，zcai_aloha2_dp_joints,zcai_aloha2_dp_tcp`。

```
@hydra.main(
    version_base="1.2", config_name="zcai_aloha2_act", config_path="../configs"
)
```

假设所选配置名字为`zcai_aloha2_act`,打开对应配置文件`~/code/ZCAI-ROBOT/dependencies/lerobot/lerobot/configs/zcai_aloha2_act.yaml`修改以下配置

```
# 更改对应repo_id
dataset_repo_id: aloha2/collect_dish_0909_0
# 初次训练时确保下面dir配置是根据时间自动设置的
hydra:
  run:
    dir: outputs/train/${dataset_repo_id}/${env.name}_${policy.name}/${now:%Y-%m-%d}_${now:%H-%M-%S}
    # dir: outputs/train/aloha2/collect_dish_0909_0/aloha2_act/2024-09-09_20-19-42
```

### 1.2.2. 恢复训练配置

如果训练中断需要再次恢复训练，需要更改以下配置.假设所选配置名字为 zcai_aloha2_act,打开对应配置文件~/code/ZCAI-ROBOT/dependencies/lerobot/lerobot/configs/zcai_aloha2_act.yaml 修改以下配置

```
# 将根据时间的dir配置注释掉，并将dir修改为你要恢复训练的文件夹地址
hydra:
  run:
    # dir: outputs/train/${dataset_repo_id}/${env.name}_${policy.name}/${now:%Y-%m-%d}_${now:%H-%M-%S}
    dir: outputs/train/aloha2/collect_dish_0909_0/aloha2_act/2024-09-09_20-19-42
```

### 1.2.2. 启动训练

正确修改完配置后，执行以下脚本启动训练

```
cd ~/code/ZCAI-ROBOT/dependencies/lerobot/scripts
./train_script.sh
```

# 2. 4090 服务器训练步骤

## 2.1. 数据集转换

执行章节 1.1 中的步骤将数据集转换，再拷贝至 4090 执行服务器。拷贝指令参考

```
# ssh到4090后执行下面指令将aloha2-3090中转换后的数据集拷贝到4090服务器
sudo cp -r ~/media/3090_aloah2/code/dataset/hf_dataset/zcai/aloha2/pick_and_place_0809_0/ ~/HUXIAN/dataset/hf_dataset/zcai/aloha2/
```

## 2.2. 训练模型

### 2.2.1. 查看空闲 cpu 位置

通过 nvidia-smi 查看空闲 gpu 位置.

```
# 打开4090服务器终端输入以下指令
nvidia-smi
# 得到以下结果
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A   2682142      C   python                                      19330MiB |
|    1   N/A  N/A   3683725      C   python                                      18240MiB |
|    2   N/A  N/A   2551793      C   python                                      14122MiB |
|    3   N/A  N/A   1926864      C   python                                      19412MiB |
+-----------------------------------------------------------------------------------------+
```

上面结果说明 4 个 gpu 都被占据。

### 2.2.2. 初次训练配置

确认一下文件`~/HUXIAN/ZCAI-ROBOT/dependencies/lerobot/lerobot/scripts/train.py`第 481 行是否为正确的配置名字。目前配置名字可选项有`zcai_aloha2_act，zcai_aloha2_diffusion`。

```
@hydra.main(
    version_base="1.2", config_name="zcai_aloha2_act", config_path="../configs"
)
```

假设所选配置名字为`zcai_aloha2_act`,打开对应配置文件`~/HUXIAN/ZCAI-ROBOT/dependencies/lerobot/lerobot/configs/zcai_aloha2_act.yaml`修改以下配置

```
# 更改对应repo_id
dataset_repo_id: aloha2/collect_dish_0909_0
# 一定要根据nvidia-smi提供空闲的gpu的id修改下面参数：
device: cuda:1
# 初次训练时确保下面dir配置是根据时间自动设置的
hydra:
  run:
    dir: outputs/train/${dataset_repo_id}/${env.name}_${policy.name}/${now:%Y-%m-%d}_${now:%H-%M-%S}
    # dir: outputs/train/aloha2/collect_dish_0909_0/aloha2_act/2024-09-09_20-19-42
```

### 2.2.3. 恢复训练配置

如果训练中断需要再次恢复训练，需要更改以下配置.假设所选配置名字为 zcai_aloha2_act,打开对应配置文件~/code/ZCAI-ROBOT/dependencies/lerobot/lerobot/configs/zcai_aloha2_act.yaml 修改以下配置

```
# 将根据时间的dir配置注释掉，并将dir修改为你要恢复训练的文件夹地址
hydra:
  run:
    # dir: outputs/train/${dataset_repo_id}/${env.name}_${policy.name}/${now:%Y-%m-%d}_${now:%H-%M-%S}
    dir: outputs/train/aloha2/collect_dish_0909_0/aloha2_act/2024-09-09_20-19-42
```

### 2.2.2. 启动训练

正确修改完配置后，执行以下脚本启动训练

```
cd ~/code/ZCAI-ROBOT/dependencies/lerobot/scripts
./train_script.sh
```

# 3. 结束训练

执行`nvidia-smi`得到以下结果

```
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A      1713      G   /usr/lib/xorg/Xorg                            156MiB |
|    0   N/A  N/A      2124      G   /usr/bin/gnome-shell                           26MiB |
|    0   N/A  N/A      3207      G   ...erProcess --variations-seed-version         47MiB |
|    0   N/A  N/A   1270857      C   python                                      18120MiB |
+-----------------------------------------------------------------------------------------+
```

由上图可以看到训练脚本产生的程序 PID 为`1270857`
执行以下指令即可终止训练

```
kill 1270857
```
