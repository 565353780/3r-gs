# 更新日志 - MASt3R依赖解耦

## 版本: 2.0.0

**发布日期**: 2026-01-06

**主要改进**: 将3r-gs从硬编码的MASt3R依赖解耦,支持使用任意相机位姿估计和特征匹配模块。

---

## 🎉 新增功能

### 1. 通用数据接口

创建了灵活的数据接口,允许使用任何位姿估计方法:

- ✅ `PoseEstimationData`: 标准化的位姿数据格式
- ✅ `CorrespondenceData`: 标准化的特征对应关系格式  
- ✅ 自动数据验证和格式检查
- ✅ 向后兼容MASt3R格式

**新文件**: `src/datasets/pose_data_interface.py`

### 2. 支持的位姿估计方法

现在可以轻松集成:

| 方法 | 状态 | 说明 |
|------|-----|------|
| MASt3R | ✅ 完全支持 | 默认方法,无需修改 |
| COLMAP | ✅ 提供转换工具 | `examples/convert_colmap_to_interface.py` |
| 自定义方法 | ✅ 提供模板 | `examples/custom_pose_estimator_template.py` |
| 其他SfM/SLAM | 🔄 可集成 | 参考文档实现 |

### 3. 新增训练脚本

**`train_custom_pose.py`** - 灵活的训练脚本

支持三种数据来源:
```bash
# MASt3R
python train_custom_pose.py --pose_source mast3r --data_dir data/scene

# COLMAP
python train_custom_pose.py --pose_source colmap --data_dir data/scene

# 自定义
python train_custom_pose.py --pose_source custom --data_dir data/scene
```

### 4. 示例和工具

#### 示例脚本

- **`examples/convert_colmap_to_interface.py`**
  - COLMAP输出转换工具
  - 支持可视化验证
  - 支持保存为标准格式

- **`examples/custom_pose_estimator_template.py`**
  - 自定义方法模板
  - 包含完整的训练流程
  - 易于修改和扩展

#### 验证工具

- **`tools/validate_pose_data.py`**
  - 数据格式验证
  - 自动检测常见错误
  - 生成可视化报告
  
```bash
# 验证数据
python tools/validate_pose_data.py --pickle pose_data.pkl --visualize
```

### 5. 完善的文档

| 文档 | 说明 |
|------|------|
| `README.md` | 项目概览,快速开始 |
| `CUSTOM_POSE_GUIDE.md` | 详细API文档,多个实例 (英文) |
| `QUICKSTART_CN.md` | 快速开始指南 (中文) |
| `MIGRATION_SUMMARY.md` | 迁移指南和变更总结 |
| `CHANGES.md` | 本文档 |

---

## 🔧 核心修改

### Parser类 (src/datasets/mast3r.py)

**之前**:
```python
def __init__(self, data_dir: str, factor: int = 1, ...):
    # 硬编码从MASt3R目录读取
    dust_dir = os.path.join(data_dir, "mast3r/0/")
    intrinsics = np.load(os.path.join(dust_dir, 'camera_intrinsics.npy'))
```

**现在**:
```python
def __init__(
    self, 
    data_dir: str, 
    factor: int = 1,
    pose_data: Optional[PoseEstimationData] = None,  # 新增
    ...
):
    if pose_data is not None:
        # 使用提供的数据
        intrinsics = pose_data.camera_intrinsics
    else:
        # 向后兼容: 从MASt3R目录加载
        dust_dir = os.path.join(data_dir, "mast3r/0/")
        intrinsics = np.load(os.path.join(dust_dir, 'camera_intrinsics.npy'))
```

### CorrespondenceDataset类 (src/datasets/mast3r.py)

**新增参数**: `corr_data: Optional[CorrespondenceData] = None`

支持直接传入特征对应关系,而不是从文件读取。

---

## 📊 使用对比

### 传统方式 (仍然支持)

```python
# 需要固定的目录结构
data/scene/
  ├── images/
  ├── mast3r/0/
  │   ├── camera_intrinsics.npy
  │   ├── camera_poses.npy
  │   └── pointcloud.ply
  ...

# 代码
parser = Parser(data_dir="data/scene")
```

### 新方式 (灵活)

```python
# 任意位姿估计方法
intrinsics, poses, points, colors = your_method()

# 转换为标准格式
pose_data = PoseEstimationData(
    camera_intrinsics=intrinsics,
    camera_poses=poses,
    point_cloud=points,
    point_cloud_rgb=colors,
)

# 使用
parser = Parser(data_dir="data/scene", pose_data=pose_data)
```

---

## 🎯 使用场景

### 场景1: 从MASt3R迁移到COLMAP

```bash
# 1. 运行COLMAP
colmap feature_extractor ...
colmap mapper ...

# 2. 转换数据
python examples/convert_colmap_to_interface.py \
    --colmap_dir data/scene/sparse/0 \
    --output_dir data/scene/pose_data \
    --visualize

# 3. 训练
python train_custom_pose.py \
    --data_dir data/scene \
    --pose_source colmap
```

### 场景2: 集成自定义SLAM

```python
# 1. 实现你的转换函数
from your_slam import run_slam
from src.datasets.pose_data_interface import PoseEstimationData

intrinsics, poses, points, colors = run_slam(images)

pose_data = PoseEstimationData(
    camera_intrinsics=intrinsics,
    camera_poses=poses,
    point_cloud=points,
    point_cloud_rgb=colors,
)

# 2. 训练
from src.datasets.mast3r import Parser
parser = Parser(data_dir="data/scene", pose_data=pose_data)
# ... 继续训练流程
```

### 场景3: 在线SLAM + 3DGS

```python
# 实时从SLAM获取位姿,增量式训练3DGS
while True:
    # 获取新的位姿
    new_intrinsics, new_poses = slam.get_latest_poses()
    
    # 更新数据
    pose_data = PoseEstimationData(...)
    
    # 重新初始化parser
    parser = Parser(data_dir="...", pose_data=pose_data)
    
    # 继续训练
    ...
```

---

## ✅ 向后兼容性

### 完全兼容

现有代码**无需任何修改**即可继续工作:

```bash
# 这些命令仍然完全有效
bash run.sh
python src/trainer.py ...
```

### 数据格式兼容

MASt3R输出目录结构保持不变:
```
data/scene/mast3r/0/
  ├── camera_intrinsics.npy
  ├── camera_poses.npy
  ├── pointcloud.ply
  ├── corr_i.npy
  └── ...
```

---

## 📋 迁移指南

### 如果你正在使用MASt3R

**不需要任何改动!** 代码会自动从MASt3R目录加载。

### 如果你想切换到其他方法

1. ✅ 准备位姿数据 (内参、位姿、点云)
2. ✅ 转换为`PoseEstimationData`格式
3. ✅ 传入`pose_data`参数
4. ✅ 开始训练

**详见**: `MIGRATION_SUMMARY.md`

---

## 🐛 已知问题

### 无

目前没有已知的重大问题。如有问题请提交Issue。

---

## 📚 API文档

### PoseEstimationData

```python
@dataclass
class PoseEstimationData:
    """相机位姿估计数据"""
    camera_intrinsics: np.ndarray  # [N, 3, 3]
    camera_poses: np.ndarray       # [N, 4, 4] camera-to-world
    point_cloud: np.ndarray        # [M, 3]
    point_cloud_rgb: np.ndarray    # [M, 3] 可选
    point_cloud_errors: np.ndarray # [M,] 可选
```

### CorrespondenceData

```python
@dataclass
class CorrespondenceData:
    """特征对应关系数据"""
    ei: np.ndarray              # [P,] 图像对索引i
    ej: np.ndarray              # [P,] 图像对索引j
    corr_i: np.ndarray          # [P, K] 对应点i
    corr_j: np.ndarray          # [P, K] 对应点j
    corr_mask: np.ndarray       # [P, K] 有效性
    corr_weight: np.ndarray     # [P, K] 权重
    depthmaps: np.ndarray       # [N, H, W] 可选
```

### 辅助函数

```python
# 从MASt3R目录加载
pose_data, corr_data = load_from_mast3r_directory(mast3r_dir)

# 创建Parser (新方式)
parser = Parser(data_dir, pose_data=pose_data)

# 创建CorrespondenceDataset (新方式)
corr_dataset = CorrespondenceDataset(parser, corr_data=corr_data)
```

---

## 🎓 学习资源

### 快速开始

1. 阅读 `QUICKSTART_CN.md` (中文)
2. 运行示例: `examples/custom_pose_estimator_template.py`
3. 验证数据: `tools/validate_pose_data.py`

### 深入学习

1. 完整指南: `CUSTOM_POSE_GUIDE.md`
2. API文档: `src/datasets/pose_data_interface.py`
3. 实际案例: `examples/` 目录

---

## 🤝 贡献

欢迎贡献!

### 如何贡献新的位姿估计方法集成

1. 在`examples/`创建转换脚本
2. 添加文档说明
3. 提供测试数据/场景
4. 提交PR

### 当前需要

- [ ] NeRFStudio集成示例
- [ ] RealityCapture转换工具
- [ ] 其他SLAM系统集成
- [ ] 更多测试场景

---

## 📞 支持

- **文档**: 查看`docs/`目录下的所有文档
- **示例**: `examples/`目录有多个实用示例
- **问题**: 提交GitHub Issue
- **讨论**: GitHub Discussions

---

## 🙏 致谢

感谢所有使用和测试这个项目的用户!

特别感谢:
- MASt3R团队提供的原始位姿估计方法
- COLMAP开发者
- gsplat库维护者

---

## 📄 许可证

MIT License - 见 `LICENSE` 文件

---

## 🔮 未来计划

- [ ] 支持更多位姿估计方法的一键转换
- [ ] 提供预训练的位姿估计模型
- [ ] 在线文档和教程
- [ ] 交互式数据验证工具
- [ ] Docker镜像支持

---

**享受使用灵活的3r-gs! 🎉**

