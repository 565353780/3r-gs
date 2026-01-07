# 使用自定义位姿估计模块指南

本指南说明如何在3r-gs中使用自定义的相机位姿估计和特征匹配模块,替代默认的MASt3R。

## 概述

3r-gs现在支持灵活的数据接口,允许你使用任何相机位姿估计方法(如COLMAP, NeRFStudio, 自定义SLAM等)和特征匹配方法(如SuperGlue, LoFTR等)。

## 数据接口

### 1. PoseEstimationData - 位姿估计数据

这个类封装了相机位姿估计模块的输出:

```python
from src.datasets.pose_data_interface import PoseEstimationData
import numpy as np

pose_data = PoseEstimationData(
    camera_intrinsics=intrinsics,  # [N, 3, 3] 相机内参矩阵
    camera_poses=poses,            # [N, 4, 4] 相机到世界的变换矩阵
    point_cloud=points,            # [M, 3] 初始点云坐标
    point_cloud_rgb=colors,        # [M, 3] 点云RGB颜色 (可选)
    point_cloud_errors=errors,     # [M,] 重投影误差 (可选)
)
```

**必需字段:**
- `camera_intrinsics`: [N, 3, 3] numpy数组
  ```
  [[fx, 0, cx],
   [0, fy, cy],
   [0, 0, 1]]
  ```
- `camera_poses`: [N, 4, 4] numpy数组, SE(3)变换矩阵 (camera-to-world)
- `point_cloud`: [M, 3] numpy数组, 3D点的世界坐标

**可选字段:**
- `point_cloud_rgb`: [M, 3] numpy数组, RGB值范围[0, 255]
- `point_cloud_errors`: [M,] numpy数组, 每个点的重投影误差

### 2. CorrespondenceData - 对应关系数据

这个类封装了特征匹配模块的输出:

```python
from src.datasets.pose_data_interface import CorrespondenceData

corr_data = CorrespondenceData(
    ei=image_pair_i,           # [P,] 第一个图像的索引
    ej=image_pair_j,           # [P,] 第二个图像的索引
    corr_i=corr_points_i,      # [P, K] 图像i中的对应点(扁平化索引)
    corr_j=corr_points_j,      # [P, K] 图像j中的对应点(扁平化索引)
    corr_mask=mask,            # [P, K] 对应点有效性
    corr_weight=weight,        # [P, K] 对应点置信度
    corr_batch_idx=batch_idx,  # [P, K] 批次索引 (可选)
    depthmaps=depths,          # [N, H, W] 深度图 (可选)
    original_image_size=(512, 512),  # 原始图像尺寸
)
```

**坐标格式说明:**
- `corr_i` 和 `corr_j` 使用扁平化索引: `index = y * width + x`
- 例如,对于512x512图像中的点(x=100, y=50): `index = 50 * 512 + 100 = 25700`

## 使用示例

### 示例1: 从MASt3R输出加载(向后兼容)

```python
from src.datasets.pose_data_interface import load_from_mast3r_directory

# 从MASt3R输出目录加载
pose_data, corr_data = load_from_mast3r_directory("data/scene/mast3r/0/")

# 使用加载的数据
from src.datasets.mast3r import Parser

parser = Parser(
    data_dir="data/scene",
    factor=2,
    pose_data=pose_data,  # 传入位姿数据
)
```

### 示例2: 使用COLMAP输出

```python
import numpy as np
from pycolmap import SceneManager
from src.datasets.pose_data_interface import PoseEstimationData

def load_colmap_data(colmap_dir):
    """从COLMAP输出加载数据"""
    manager = SceneManager(colmap_dir)
    manager.load_cameras()
    manager.load_images()
    manager.load_points3D()
    
    # 提取相机内参和位姿
    intrinsics_list = []
    poses_list = []
    
    for img_id in sorted(manager.images.keys()):
        image = manager.images[img_id]
        cam = manager.cameras[image.camera_id]
        
        # 内参
        K = np.array([
            [cam.fx, 0, cam.cx],
            [0, cam.fy, cam.cy],
            [0, 0, 1]
        ])
        intrinsics_list.append(K)
        
        # COLMAP使用world-to-camera,需要转换为camera-to-world
        from scipy.spatial.transform import Rotation
        R = Rotation.from_quat([*image.qvec[1:], image.qvec[0]]).as_matrix()
        t = image.tvec
        
        # world-to-camera
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = t
        
        # camera-to-world
        c2w = np.linalg.inv(w2c)
        poses_list.append(c2w)
    
    # 提取3D点
    points = []
    colors = []
    for pt_id, point in manager.points3D.items():
        points.append(point.xyz)
        colors.append(point.color)
    
    return PoseEstimationData(
        camera_intrinsics=np.array(intrinsics_list),
        camera_poses=np.array(poses_list),
        point_cloud=np.array(points),
        point_cloud_rgb=np.array(colors),
    )

# 使用COLMAP数据
pose_data = load_colmap_data("data/scene/sparse/0")

from src.datasets.mast3r import Parser
parser = Parser(
    data_dir="data/scene",
    factor=2,
    pose_data=pose_data,
)
```

### 示例3: 使用自定义位姿估计器

```python
import numpy as np
from src.datasets.pose_data_interface import PoseEstimationData

def run_custom_pose_estimator(image_paths):
    """
    运行你自己的位姿估计算法
    
    这可以是任何方法:
    - 自定义的SLAM系统
    - 神经网络位姿估计 (如PosNet, NeRF--)
    - 其他SfM系统
    """
    # 你的位姿估计代码
    # ...
    
    # 返回估计的内参、位姿和点云
    return intrinsics, poses, points, colors

# 运行你的算法
intrinsics, poses, points, colors = run_custom_pose_estimator(image_paths)

# 创建数据对象
pose_data = PoseEstimationData(
    camera_intrinsics=intrinsics,  # [N, 3, 3]
    camera_poses=poses,            # [N, 4, 4]
    point_cloud=points,            # [M, 3]
    point_cloud_rgb=colors,        # [M, 3]
)

# 使用数据
from src.datasets.mast3r import Parser
parser = Parser(
    data_dir="data/scene",
    factor=2,
    pose_data=pose_data,
)
```

### 示例4: 使用自定义特征匹配

```python
import numpy as np
from src.datasets.pose_data_interface import CorrespondenceData

def run_custom_feature_matching(images, image_pairs):
    """
    运行你自己的特征匹配算法
    
    这可以是:
    - SuperGlue
    - LoFTR
    - SIFT + RANSAC
    - 深度学习匹配器
    """
    ei_list = []
    ej_list = []
    corr_i_list = []
    corr_j_list = []
    mask_list = []
    weight_list = []
    
    for idx_i, idx_j in image_pairs:
        # 运行特征匹配
        matches, confidences = match_features(images[idx_i], images[idx_j])
        
        # 转换为扁平化索引
        # matches: [(x_i, y_i, x_j, y_j), ...]
        width, height = 512, 512  # 你的图像尺寸
        
        flat_i = [y_i * width + x_i for x_i, y_i, _, _ in matches]
        flat_j = [y_j * width + x_j for _, _, x_j, y_j in matches]
        
        ei_list.append(idx_i)
        ej_list.append(idx_j)
        corr_i_list.append(flat_i)
        corr_j_list.append(flat_j)
        mask_list.append([1] * len(matches))  # 全部有效
        weight_list.append(confidences)
    
    # 填充到相同长度
    max_len = max(len(x) for x in corr_i_list)
    for i in range(len(corr_i_list)):
        pad_len = max_len - len(corr_i_list[i])
        corr_i_list[i].extend([0] * pad_len)
        corr_j_list[i].extend([0] * pad_len)
        mask_list[i].extend([0] * pad_len)
        weight_list[i].extend([0] * pad_len)
    
    return CorrespondenceData(
        ei=np.array(ei_list),
        ej=np.array(ej_list),
        corr_i=np.array(corr_i_list),
        corr_j=np.array(corr_j_list),
        corr_mask=np.array(mask_list),
        corr_weight=np.array(weight_list),
        original_image_size=(width, height),
    )

# 使用自定义匹配
corr_data = run_custom_feature_matching(images, pairs)

# 在训练时使用
from src.datasets.mast3r import CorrespondenceDataset
corr_dataset = CorrespondenceDataset(
    parser=parser,
    split="train",
    corr_data=corr_data,  # 传入对应关系数据
)
```

### 示例5: 在训练脚本中使用

修改你的训练脚本 (例如 `run_custom.py`):

```python
import numpy as np
from src.datasets.pose_data_interface import PoseEstimationData, CorrespondenceData
from src.datasets.mast3r import Parser, Dataset, CorrespondenceDataset
from src.trainer import Runner, Config

# 1. 加载或生成你的位姿数据
pose_data = PoseEstimationData(
    camera_intrinsics=your_intrinsics,
    camera_poses=your_poses,
    point_cloud=your_points,
    point_cloud_rgb=your_colors,
)

# 2. (可选) 加载或生成对应关系数据
corr_data = CorrespondenceData(
    ei=your_ei,
    ej=your_ej,
    corr_i=your_corr_i,
    corr_j=your_corr_j,
    corr_mask=your_mask,
    corr_weight=your_weight,
)

# 3. 创建Parser
parser = Parser(
    data_dir="data/scene",
    factor=2,
    normalize=True,
    pose_data=pose_data,  # 使用自定义数据
)

# 4. 创建Dataset
trainset = Dataset(parser, split="train")

# 5. (可选) 如果使用对应关系损失
if use_correspondence_loss:
    corr_dataset = CorrespondenceDataset(
        parser=parser,
        split="train",
        corr_data=corr_data,  # 使用自定义对应关系
    )

# 6. 继续训练流程
# ...
```

## 完整训练示例

创建一个新的训练脚本 `train_custom.py`:

```python
#!/usr/bin/env python3
"""使用自定义位姿估计的训练脚本示例"""

import os
import numpy as np
import torch
from src.datasets.pose_data_interface import PoseEstimationData, load_from_mast3r_directory
from src.datasets.mast3r import Parser, Dataset
from src.trainer import Config, Runner, main
import tyro

def main_custom():
    """自定义训练入口"""
    
    # 配置
    data_dir = "data/your_scene"
    
    # 方式1: 从MASt3R加载 (向后兼容)
    # pose_data, corr_data = load_from_mast3r_directory(
    #     os.path.join(data_dir, "mast3r/0")
    # )
    
    # 方式2: 使用你自己的位姿估计
    from your_pose_estimator import estimate_poses
    intrinsics, poses, points, colors = estimate_poses(data_dir)
    
    pose_data = PoseEstimationData(
        camera_intrinsics=intrinsics,
        camera_poses=poses,
        point_cloud=points,
        point_cloud_rgb=colors,
    )
    
    # 创建配置
    cfg = Config(
        data_dir=data_dir,
        data_factor=2,
        result_dir="results/custom_experiment",
        use_corres_epipolar_loss=False,  # 如果没有对应关系数据
    )
    
    # 创建Runner并修改parser
    runner = Runner(
        local_rank=0,
        world_rank=0,
        world_size=1,
        cfg=cfg,
    )
    
    # 替换parser为使用自定义数据的版本
    runner.parser = Parser(
        data_dir=data_dir,
        factor=cfg.data_factor,
        normalize=cfg.normalize_world_space,
        test_every=cfg.test_every,
        pose_data=pose_data,  # 使用自定义数据
    )
    
    # 重新创建datasets
    runner.trainset = Dataset(runner.parser, split="train")
    runner.trainvalset = Dataset(runner.parser, split="train")
    runner.valset = Dataset(runner.parser, split="val")
    
    # 开始训练
    runner.train()

if __name__ == "__main__":
    main_custom()
```

## 注意事项

### 坐标系统
- **相机位姿**: 使用camera-to-world变换 (c2w)
- **点云**: 世界坐标系
- **图像坐标**: 左上角为原点 (0, 0)

### 数据格式
- 所有数组应该是numpy数组 (np.ndarray)
- 数据类型: float32 或 float64
- 确保数据已经正确对齐(按图像名称排序)

### 性能建议
- 初始点云建议包含10,000-100,000个点
- 对应关系数据:每对图像建议100-1000个对应点
- 过多的点会增加内存占用和训练时间

## 故障排查

### 问题: "camera_intrinsics应该是[N, 3, 3]"
**解决**: 检查内参矩阵的形状,确保每个相机都有3x3矩阵

### 问题: "ei和ej应该有相同的形状"
**解决**: 确保图像对索引数组长度相同

### 问题: 训练结果不好
**可能原因**:
1. 位姿估计不准确 - 检查初始位姿质量
2. 点云稀疏 - 增加初始点数
3. 坐标系不对齐 - 确认使用camera-to-world格式

## 更多资源

- 查看 `src/datasets/pose_data_interface.py` 了解接口详情
- 参考 `src/datasets/mast3r.py` 了解数据使用方式
- 阅读 MASt3R论文了解数据格式背景

## 贡献

如果你实现了新的位姿估计器或特征匹配器的转换函数,欢迎提交PR!

