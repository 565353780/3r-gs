# 快速开始指南 - 使用自定义位姿估计

本指南帮助你快速开始使用自定义的相机位姿估计模块替代MASt3R。

## 🎯 核心概念

3r-gs现在使用**通用数据接口**,不再依赖特定的位姿估计方法。你可以使用:

- ✅ MASt3R (原有方式,完全兼容)
- ✅ COLMAP 
- ✅ 你自己的SLAM系统
- ✅ 任何其他方法

## 📋 需要准备的数据

无论使用什么方法,你需要提供:

### 1. 相机位姿数据 (必需)

```python
PoseEstimationData(
    camera_intrinsics,  # [N, 3, 3] 内参矩阵
    camera_poses,       # [N, 4, 4] 相机位姿 (camera-to-world)
    point_cloud,        # [M, 3] 初始点云
    point_cloud_rgb,    # [M, 3] 点云颜色 (可选)
)
```

### 2. 特征对应关系 (可选,用于对极约束)

```python
CorrespondenceData(
    ei, ej,             # 图像对索引
    corr_i, corr_j,     # 对应点坐标
    corr_mask,          # 有效性掩码
    corr_weight,        # 置信度权重
)
```

## 🚀 三种使用方式

### 方式1: 使用MASt3R (不需要修改代码)

```bash
# 你的数据结构:
# data/scene/
#   ├── images/
#   ├── mast3r/0/
#   │   ├── camera_intrinsics.npy
#   │   ├── camera_poses.npy
#   │   └── pointcloud.ply
#   ├── images_train.txt
#   └── images_test.txt

# 直接运行
bash run.sh
```

### 方式2: 使用COLMAP 

```bash
# 你的数据结构:
# data/scene/
#   ├── images/
#   ├── sparse/0/
#   │   ├── cameras.bin
#   │   ├── images.bin
#   │   └── points3D.bin
#   ├── images_train.txt
#   └── images_test.txt

# 运行训练
CUDA_VISIBLE_DEVICES=0 python train_custom_pose.py \
    --data_dir data/scene \
    --pose_source colmap \
    --data_factor 2 \
    --result_dir results/my_experiment
```

### 方式3: 使用你自己的方法

#### 步骤1: 准备数据

```python
# your_pose_estimator.py
import numpy as np

def run_my_pose_estimator(image_paths):
    """运行你的位姿估计算法"""
    
    # ... 你的代码 ...
    # 输出:
    # - intrinsics: [N, 3, 3] 内参
    # - poses: [N, 4, 4] 位姿矩阵
    # - points: [M, 3] 3D点
    # - colors: [M, 3] RGB颜色
    
    return intrinsics, poses, points, colors
```

#### 步骤2: 转换为标准格式

```python
# convert_data.py
from src.datasets.pose_data_interface import PoseEstimationData
from your_pose_estimator import run_my_pose_estimator

# 运行你的算法
intrinsics, poses, points, colors = run_my_pose_estimator(image_paths)

# 转换为标准格式
pose_data = PoseEstimationData(
    camera_intrinsics=intrinsics,
    camera_poses=poses,
    point_cloud=points,
    point_cloud_rgb=colors,
)

# 保存 (可选)
import pickle
with open('my_pose_data.pkl', 'wb') as f:
    pickle.dump(pose_data, f)
```

#### 步骤3: 训练

```python
# train_my_method.py
import pickle
from src.datasets.mast3r import Parser, Dataset
from src.trainer import Runner, Config

# 加载数据
with open('my_pose_data.pkl', 'rb') as f:
    pose_data = pickle.load(f)

# 创建配置
cfg = Config(
    data_dir="data/scene",
    data_factor=2,
    result_dir="results/my_exp",
)

# 创建Runner
runner = Runner(0, 0, 1, cfg)

# 使用自定义数据
runner.parser = Parser(
    data_dir=cfg.data_dir,
    factor=cfg.data_factor,
    pose_data=pose_data,  # 传入你的数据
)

# 重新创建datasets
runner.trainset = Dataset(runner.parser, split="train")
runner.valset = Dataset(runner.parser, split="val")

# 训练
runner.train()
```

## 📐 数据格式详解

### 相机内参矩阵 (3x3)

```
[[fx,  0, cx],
 [ 0, fy, cy],
 [ 0,  0,  1]]
```

- `fx, fy`: 焦距 (像素单位)
- `cx, cy`: 主点坐标

### 相机位姿矩阵 (4x4)

使用 **camera-to-world** 格式:

```
[[R11, R12, R13, tx],
 [R21, R22, R23, ty],
 [R31, R32, R33, tz],
 [  0,   0,   0,  1]]
```

- `R`: 3x3 旋转矩阵
- `t`: 3x1 平移向量

⚠️ **注意**: 如果你的方法输出world-to-camera,需要求逆:
```python
c2w = np.linalg.inv(w2c)
```

### 点云格式

- `point_cloud`: [M, 3] 形状,每行是 `[x, y, z]` 世界坐标
- `point_cloud_rgb`: [M, 3] 形状,每行是 `[r, g, b]`,范围 [0, 255]

## 🔍 常见问题

### Q1: 我的方法输出的是world-to-camera矩阵怎么办?

```python
# 转换为camera-to-world
camera_poses_c2w = np.linalg.inv(camera_poses_w2c)
```

### Q2: 我没有初始点云怎么办?

```python
# 可以生成随机点云
num_points = 10000
points = np.random.randn(num_points, 3) * 2  # 在原点附近
colors = np.random.rand(num_points, 3) * 255
```

### Q3: 我只有几个视角的相机,够吗?

一般建议至少10-20个视角。如果视角太少,3DGS可能难以收敛。

### Q4: 特征对应关系是必需的吗?

不是必需的!如果没有对应关系:
```python
cfg.use_corres_epipolar_loss = False  # 关闭对极约束损失
```

### Q5: 训练结果不好怎么办?

检查:
1. ✅ 位姿矩阵格式是否正确 (camera-to-world)
2. ✅ 内参是否准确
3. ✅ 点云是否合理分布
4. ✅ 图像是否按照相同顺序排列

## 📊 完整示例: 从零开始

```python
#!/usr/bin/env python3
"""完整示例: 使用自定义位姿估计"""

import os
import numpy as np
from src.datasets.pose_data_interface import PoseEstimationData
from src.datasets.mast3r import Parser, Dataset
from src.trainer import Config, Runner

# 1. 准备数据目录
data_dir = "data/my_scene"
os.makedirs(data_dir, exist_ok=True)

# 2. 准备图像列表文件
with open(f"{data_dir}/images_train.txt", 'w') as f:
    f.write('\n'.join([f"frame_{i:04d}" for i in range(50)]))
    
with open(f"{data_dir}/images_test.txt", 'w') as f:
    f.write('\n'.join([f"frame_{i:04d}" for i in range(50, 60)]))

# 3. 运行你的位姿估计
intrinsics, poses, points, colors = your_pose_estimator()

# 4. 创建数据对象
pose_data = PoseEstimationData(
    camera_intrinsics=intrinsics,  # [N, 3, 3]
    camera_poses=poses,            # [N, 4, 4]
    point_cloud=points,            # [M, 3]
    point_cloud_rgb=colors,        # [M, 3]
)

# 5. 训练配置
cfg = Config(
    data_dir=data_dir,
    data_factor=2,
    result_dir="results/my_scene",
    max_steps=30000,
    use_corres_epipolar_loss=False,  # 没有对应关系
)

# 6. 创建训练器
runner = Runner(0, 0, 1, cfg)

# 7. 使用自定义数据
runner.parser = Parser(
    data_dir=data_dir,
    factor=2,
    pose_data=pose_data,
)

# 8. 重新创建数据集
runner.trainset = Dataset(runner.parser, split="train")
runner.valset = Dataset(runner.parser, split="val")

# 9. 开始训练
runner.train()
```

## 🎓 进阶功能

### 使用特征对应关系

如果你有特征匹配结果,可以使用对极约束来提高位姿优化:

```python
from src.datasets.pose_data_interface import CorrespondenceData

corr_data = CorrespondenceData(
    ei=np.array([0, 1, 2]),           # 第一个图像索引
    ej=np.array([1, 2, 3]),           # 第二个图像索引
    corr_i=matches_i,                 # [P, K] 匹配点
    corr_j=matches_j,                 # [P, K]
    corr_mask=mask,                   # [P, K]
    corr_weight=confidence,           # [P, K]
)

# 在创建CorrespondenceDataset时传入
from src.datasets.mast3r import CorrespondenceDataset
corr_dataset = CorrespondenceDataset(
    parser=parser,
    split="train",
    corr_data=corr_data,
)
```

## 📚 更多资源

- 详细API文档: [CUSTOM_POSE_GUIDE.md](CUSTOM_POSE_GUIDE.md)
- 示例脚本: `train_custom_pose.py`
- 数据接口: `src/datasets/pose_data_interface.py`

## 💡 提示

1. **调试建议**: 先用少量图像 (5-10张) 测试你的数据格式是否正确
2. **可视化**: 可以用 `open3d` 或 `matplotlib` 可视化位姿和点云
3. **性能**: 初始点云建议1万到10万个点,太多会变慢
4. **质量**: 确保你的位姿估计足够准确,否则3DGS难以优化

祝你使用顺利! 🎉

