# 3R-GS: 3D高斯喷溅重建系统

一个灵活的3D高斯喷溅训练系统,支持使用任意相机位姿估计和特征匹配模块。

## ✨ 新特性

**🔧 模块化位姿估计接口**
- ✅ 不再硬编码依赖MASt3R
- ✅ 支持任意相机位姿估计模块 (MASt3R, COLMAP, NeRFStudio, 自定义SLAM等)
- ✅ 支持任意特征匹配模块 (SuperGlue, LoFTR, SIFT等)
- ✅ 向后兼容原有MASt3R数据格式

## 📦 安装

```bash
# 克隆仓库
git clone <repository_url>
cd 3r-gs

# 安装依赖
bash setup.sh  # 或 bash dev_setup.sh
```

## 🚀 快速开始

### 方式1: 使用MASt3R (传统方式)

```bash
# 准备MASt3R输出数据
# data/scene/mast3r/0/
#   ├── camera_intrinsics.npy
#   ├── camera_poses.npy
#   ├── pointcloud.ply
#   └── ...

# 运行训练
bash run.sh
```

### 方式2: 使用COLMAP

```bash
# 使用COLMAP数据训练
CUDA_VISIBLE_DEVICES=0 python train_custom_pose.py \
    --data_dir data/scene \
    --pose_source colmap \
    --colmap_dir data/scene/sparse/0 \
    --data_factor 2 \
    --result_dir results/colmap_exp
```

### 方式3: 使用自定义位姿估计

```python
# 创建自定义训练脚本
from src.datasets.pose_data_interface import PoseEstimationData
import numpy as np

# 1. 运行你的位姿估计算法
intrinsics, poses, points, colors = your_pose_estimator(images)

# 2. 创建数据对象
pose_data = PoseEstimationData(
    camera_intrinsics=intrinsics,  # [N, 3, 3]
    camera_poses=poses,            # [N, 4, 4] camera-to-world
    point_cloud=points,            # [M, 3]
    point_cloud_rgb=colors,        # [M, 3]
)

# 3. 使用自定义数据训练
from src.datasets.mast3r import Parser
parser = Parser(
    data_dir="data/scene",
    factor=2,
    pose_data=pose_data,  # 传入自定义数据
)

# 4. 继续训练流程
# ...
```

## 📚 文档

### 详细使用指南

查看 [CUSTOM_POSE_GUIDE.md](CUSTOM_POSE_GUIDE.md) 获取:
- 完整的数据接口说明
- 多种位姿估计方法的集成示例
- 自定义特征匹配的实现方法
- 故障排查指南

### 数据格式

#### PoseEstimationData (位姿估计数据)

```python
PoseEstimationData(
    camera_intrinsics: np.ndarray,  # [N, 3, 3] 相机内参矩阵
    camera_poses: np.ndarray,       # [N, 4, 4] camera-to-world变换
    point_cloud: np.ndarray,        # [M, 3] 3D点坐标
    point_cloud_rgb: np.ndarray,    # [M, 3] RGB颜色 (可选)
    point_cloud_errors: np.ndarray, # [M,] 重投影误差 (可选)
)
```

#### CorrespondenceData (特征对应关系数据)

```python
CorrespondenceData(
    ei: np.ndarray,              # [P,] 图像对索引i
    ej: np.ndarray,              # [P,] 图像对索引j
    corr_i: np.ndarray,          # [P, K] 图像i中的匹配点
    corr_j: np.ndarray,          # [P, K] 图像j中的匹配点
    corr_mask: np.ndarray,       # [P, K] 有效性掩码
    corr_weight: np.ndarray,     # [P, K] 置信度权重
    depthmaps: np.ndarray,       # [N, H, W] 深度图 (可选)
)
```

## 🔧 支持的位姿估计方法

| 方法 | 支持状态 | 说明 |
|------|---------|------|
| MASt3R | ✅ 完全支持 | 默认方法,向后兼容 |
| COLMAP | ✅ 完全支持 | 提供转换函数 |
| NeRFStudio | 🔄 可集成 | 参考文档实现 |
| 自定义SLAM | 🔄 可集成 | 实现数据接口即可 |
| 神经网络位姿估计 | 🔄 可集成 | 如PosNet, NeRF-- |

## 🎯 主要改进

### 之前 (硬编码MASt3R)
```python
# 只能从固定目录读取MASt3R输出
dust_dir = os.path.join(data_dir, "mast3r/0/")
intrinsics = np.load(os.path.join(dust_dir, 'camera_intrinsics.npy'))
poses = np.load(os.path.join(dust_dir, 'camera_poses.npy'))
```

### 现在 (灵活的数据接口)
```python
# 可以使用任何位姿估计方法
pose_data = PoseEstimationData(
    camera_intrinsics=your_intrinsics,
    camera_poses=your_poses,
    point_cloud=your_points,
)

parser = Parser(data_dir="...", pose_data=pose_data)
```

## 📝 使用示例

### 示例1: 从现有MASt3R数据加载

```python
from src.datasets.pose_data_interface import load_from_mast3r_directory

# 加载MASt3R输出
pose_data, corr_data = load_from_mast3r_directory("data/scene/mast3r/0/")

# 使用加载的数据
parser = Parser(data_dir="data/scene", pose_data=pose_data)
```

### 示例2: 集成COLMAP

```bash
# 直接使用COLMAP数据训练
python train_custom_pose.py \
    --data_dir data/scene \
    --pose_source colmap \
    --data_factor 2
```

### 示例3: 集成自定义算法

参考 `train_custom_pose.py` 中的 `load_pose_data_custom()` 函数,实现你自己的数据加载逻辑。

## 🏗️ 项目结构

```
3r-gs/
├── src/
│   ├── datasets/
│   │   ├── pose_data_interface.py  # 通用数据接口 (新)
│   │   ├── mast3r.py               # Parser和Dataset类 (已更新)
│   │   └── ...
│   ├── trainer.py                  # 训练器
│   └── utils/
├── train_custom_pose.py            # 自定义位姿训练脚本 (新)
├── run.sh                          # 默认训练脚本
├── CUSTOM_POSE_GUIDE.md           # 详细使用指南 (新)
└── README.md                       # 本文件
```

## 🤝 贡献

欢迎贡献代码!如果你实现了新的位姿估计器或特征匹配器的集成,请提交PR。

### 添加新的位姿估计方法

1. 在 `pose_data_interface.py` 中添加转换函数
2. 更新文档说明数据格式
3. 提供示例代码

## 📄 许可证

见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- 原始3R-GS实现
- [MASt3R](https://github.com/naver/mast3r) 用于位姿估计
- [gsplat](https://github.com/nerfstudio-project/gsplat) 3D高斯喷溅库
- [COLMAP](https://colmap.github.io/) SfM系统

## 📮 联系方式

如有问题或建议,请提交Issue。
