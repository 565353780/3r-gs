# 迁移总结 - MASt3R依赖解耦

## 🎯 改进目标

将3r-gs从硬编码的MASt3R依赖解耦,允许使用任意相机位姿估计和特征匹配模块。

## 📝 主要变化

### 1. 新增文件

| 文件 | 说明 |
|------|------|
| `src/datasets/pose_data_interface.py` | 通用数据接口定义 |
| `train_custom_pose.py` | 使用自定义位姿的训练脚本 |
| `CUSTOM_POSE_GUIDE.md` | 详细使用指南 |
| `QUICKSTART_CN.md` | 中文快速开始 |
| `examples/convert_colmap_to_interface.py` | COLMAP转换示例 |
| `examples/custom_pose_estimator_template.py` | 自定义方法模板 |

### 2. 修改文件

| 文件 | 修改内容 |
|------|---------|
| `src/datasets/mast3r.py` | Parser和CorrespondenceDataset类支持传入数据 |
| `src/trainer.py` | 导入新的数据接口 |
| `README.md` | 更新说明文档 |

### 3. 核心接口

#### PoseEstimationData (位姿数据)

```python
@dataclass
class PoseEstimationData:
    camera_intrinsics: np.ndarray  # [N, 3, 3]
    camera_poses: np.ndarray       # [N, 4, 4]
    point_cloud: np.ndarray        # [M, 3]
    point_cloud_rgb: np.ndarray    # [M, 3] (可选)
    point_cloud_errors: np.ndarray # [M,] (可选)
```

#### CorrespondenceData (对应关系)

```python
@dataclass
class CorrespondenceData:
    ei: np.ndarray              # [P,]
    ej: np.ndarray              # [P,]
    corr_i: np.ndarray          # [P, K]
    corr_j: np.ndarray          # [P, K]
    corr_mask: np.ndarray       # [P, K]
    corr_weight: np.ndarray     # [P, K]
    depthmaps: np.ndarray       # [N, H, W] (可选)
```

## 🔄 向后兼容性

### ✅ 完全向后兼容

现有使用MASt3R的代码**无需任何修改**即可继续运行:

```bash
# 原有方式仍然有效
bash run.sh
```

当`pose_data=None`时,Parser会自动从MASt3R目录加载数据。

## 🚀 使用新接口

### 方式1: 从MASt3R加载 (显式)

```python
from src.datasets.pose_data_interface import load_from_mast3r_directory

pose_data, corr_data = load_from_mast3r_directory("data/scene/mast3r/0/")
parser = Parser(data_dir="data/scene", pose_data=pose_data)
```

### 方式2: 使用COLMAP

```bash
python train_custom_pose.py \
    --data_dir data/scene \
    --pose_source colmap
```

### 方式3: 自定义方法

```python
# 1. 运行你的位姿估计
intrinsics, poses, points, colors = your_estimator()

# 2. 创建数据对象
pose_data = PoseEstimationData(
    camera_intrinsics=intrinsics,
    camera_poses=poses,
    point_cloud=points,
    point_cloud_rgb=colors,
)

# 3. 训练
parser = Parser(data_dir="...", pose_data=pose_data)
```

## 📊 代码变化对比

### Parser类 - 之前

```python
class Parser:
    def __init__(self, data_dir: str, factor: int = 1, ...):
        dust_dir = os.path.join(data_dir, "mast3r/0/")
        intrinsics = np.load(os.path.join(dust_dir, 'camera_intrinsics.npy'))
        poses = np.load(os.path.join(dust_dir, 'camera_poses.npy'))
        # 硬编码从固定目录读取
```

### Parser类 - 现在

```python
class Parser:
    def __init__(
        self, 
        data_dir: str, 
        factor: int = 1,
        pose_data: Optional[PoseEstimationData] = None,  # 新增参数
        ...
    ):
        if pose_data is not None:
            # 使用提供的数据
            intrinsics = pose_data.camera_intrinsics
            poses = pose_data.camera_poses
        else:
            # 向后兼容: 从MASt3R目录加载
            dust_dir = os.path.join(data_dir, "mast3r/0/")
            intrinsics = np.load(os.path.join(dust_dir, 'camera_intrinsics.npy'))
            poses = np.load(os.path.join(dust_dir, 'camera_poses.npy'))
```

## ⚙️ 数据格式要求

### 坐标系统

| 项目 | 格式 | 说明 |
|------|------|------|
| 相机位姿 | camera-to-world (c2w) | 如果是w2c需要求逆 |
| 点云 | 世界坐标系 | - |
| 图像坐标 | 左上角为原点(0,0) | - |

### 矩阵格式

```python
# 内参矩阵 (3x3)
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]

# 位姿矩阵 (4x4, camera-to-world)
c2w = [[R11, R12, R13, tx],
       [R21, R22, R23, ty],
       [R31, R32, R33, tz],
       [  0,   0,   0,  1]]
```

## 🔧 迁移检查清单

如果你想从MASt3R迁移到其他方法:

- [ ] 确认你的位姿格式是camera-to-world
- [ ] 验证内参矩阵格式正确
- [ ] 检查点云在合理范围内
- [ ] 准备图像split文件 (images_train.txt, images_test.txt)
- [ ] 确保图像和位姿按照相同顺序排列
- [ ] (可选) 准备特征对应关系数据

## 📚 文档资源

| 文档 | 内容 |
|------|------|
| [README.md](README.md) | 项目概览和快速开始 |
| [CUSTOM_POSE_GUIDE.md](CUSTOM_POSE_GUIDE.md) | 详细API文档和示例 |
| [QUICKSTART_CN.md](QUICKSTART_CN.md) | 中文快速指南 |
| `examples/` | 实用示例脚本 |

## 🐛 故障排查

### 问题: 位姿格式错误

**症状**: 训练结果很差,相机位置看起来不对

**解决**:
```python
# 检查是否需要转换
if your_poses_are_w2c:
    poses = np.linalg.inv(poses)  # 转换为c2w
```

### 问题: 数据未对齐

**症状**: 图像和位姿不匹配

**解决**: 确保图像名称和位姿按照相同顺序排序

```python
# 按文件名排序
image_names = sorted(os.listdir(image_dir))
# 确保位姿也按照相同顺序
```

### 问题: 点云太少或太多

**建议**:
- 最少: 1,000 点
- 推荐: 10,000 - 100,000 点
- 最多: 1,000,000 点 (会变慢)

## 💡 最佳实践

1. **先测试小规模**: 用5-10张图像验证数据格式
2. **可视化验证**: 使用matplotlib或open3d检查位姿和点云
3. **保存中间结果**: 转换后的数据保存为.npy文件,避免重复计算
4. **渐进式迁移**: 先用MASt3R验证流程,再替换为自定义方法

## 🤝 贡献

如果你实现了新的位姿估计方法集成,欢迎贡献到`examples/`目录!

## ✉️ 支持

遇到问题请查看文档或提交Issue。

