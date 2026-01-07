#!/usr/bin/env python3
"""
自定义位姿估计器模板

这是一个模板文件,展示如何将你自己的位姿估计方法集成到3r-gs中。
复制这个文件并根据你的需求修改。
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Tuple

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets.pose_data_interface import PoseEstimationData, CorrespondenceData


def your_pose_estimation_method(
    image_dir: str,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    你的位姿估计方法
    
    在这里实现你自己的位姿估计算法,可以是:
    - 自定义SLAM系统
    - 神经网络位姿估计 (如PosNet)
    - 从其他软件的输出读取 (如Metashape, RealityCapture)
    - 其他SfM系统
    
    Args:
        image_dir: 图像目录路径
        **kwargs: 其他参数
        
    Returns:
        intrinsics: [N, 3, 3] 相机内参矩阵
        poses: [N, 4, 4] 相机位姿矩阵 (camera-to-world)
        points: [M, 3] 3D点坐标
        colors: [M, 3] RGB颜色 (0-255)
    """
    
    # ============================================================
    # TODO: 在这里实现你的位姿估计逻辑
    # ============================================================
    
    # 示例: 读取你的输出文件
    # intrinsics = np.load("path/to/your/intrinsics.npy")
    # poses = np.load("path/to/your/poses.npy")
    # points = np.load("path/to/your/points.npy")
    # colors = np.load("path/to/your/colors.npy")
    
    # 示例: 调用你的API
    # from your_slam_system import run_slam
    # result = run_slam(image_dir)
    # intrinsics = result.intrinsics
    # poses = result.poses
    # points = result.points
    # colors = result.colors
    
    # 临时示例: 生成随机数据
    print("警告: 使用随机数据,请替换为你的实际实现!")
    
    # 假设有50个图像
    num_images = 50
    
    # 生成内参 (假设所有相机使用相同内参)
    K = np.array([
        [800, 0, 400],
        [0, 800, 300],
        [0, 0, 1]
    ], dtype=np.float32)
    intrinsics = np.tile(K, (num_images, 1, 1))
    
    # 生成位姿 (环形排列的相机)
    poses = []
    radius = 5.0
    for i in range(num_images):
        angle = 2 * np.pi * i / num_images
        
        # 相机位置
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 1.0 + np.sin(angle * 4) * 0.5  # 添加一些高度变化
        
        # 计算朝向原点的旋转矩阵
        # 前向: 从相机指向原点
        forward = -np.array([x, y, z])
        forward = forward / np.linalg.norm(forward)
        
        # 右向: 前向叉乘世界上方向
        right = np.cross(forward, np.array([0, 0, 1]))
        right = right / np.linalg.norm(right)
        
        # 上向: 右向叉乘前向
        up = np.cross(right, forward)
        
        # 构建旋转矩阵 [right, up, -forward]
        R = np.stack([right, up, -forward], axis=1)
        
        # 构建camera-to-world矩阵
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R
        c2w[:3, 3] = [x, y, z]
        
        poses.append(c2w)
    
    poses = np.array(poses, dtype=np.float32)
    
    # 生成点云
    num_points = 5000
    points = np.random.randn(num_points, 3).astype(np.float32) * 2
    colors = np.random.rand(num_points, 3).astype(np.float32) * 255
    
    # ============================================================
    # 数据验证 (建议保留)
    # ============================================================
    
    assert intrinsics.shape[1:] == (3, 3), f"内参应该是 [N, 3, 3], 得到 {intrinsics.shape}"
    assert poses.shape[1:] == (4, 4), f"位姿应该是 [N, 4, 4], 得到 {poses.shape}"
    assert points.shape[1] == 3, f"点云应该是 [M, 3], 得到 {points.shape}"
    assert colors.shape == points.shape, "颜色和点云形状应该相同"
    
    # 检查位姿矩阵是否是有效的SE(3)变换
    for i, pose in enumerate(poses[:5]):  # 检查前几个
        R = pose[:3, :3]
        det = np.linalg.det(R)
        if not np.isclose(det, 1.0, atol=1e-3):
            print(f"警告: 位姿 {i} 的旋转矩阵行列式不为1: {det}")
    
    print(f"位姿估计完成:")
    print(f"  - 相机数量: {len(intrinsics)}")
    print(f"  - 点云数量: {len(points)}")
    
    return intrinsics, poses, points, colors


def your_feature_matching_method(
    images: list,
    image_pairs: list,
) -> CorrespondenceData:
    """
    你的特征匹配方法 (可选)
    
    如果你有特征匹配结果,可以在这里实现转换。
    这是可选的,如果没有可以返回None或不使用对极约束。
    
    Args:
        images: 图像列表
        image_pairs: 图像对列表 [(i, j), ...]
        
    Returns:
        CorrespondenceData对象
    """
    
    # ============================================================
    # TODO: 在这里实现你的特征匹配逻辑
    # ============================================================
    
    # 示例: 使用SuperGlue
    # from your_matcher import match_image_pair
    # for i, j in image_pairs:
    #     matches = match_image_pair(images[i], images[j])
    #     ...
    
    print("特征匹配方法未实现,返回None")
    return None


def convert_to_interface(
    image_dir: str,
    output_dir: str = None,
    **kwargs
) -> PoseEstimationData:
    """
    转换为标准接口格式
    
    Args:
        image_dir: 图像目录
        output_dir: 输出目录 (可选,用于保存numpy文件)
        **kwargs: 传递给位姿估计方法的参数
        
    Returns:
        PoseEstimationData对象
    """
    
    print(f"处理图像目录: {image_dir}")
    
    # 运行位姿估计
    intrinsics, poses, points, colors = your_pose_estimation_method(
        image_dir, **kwargs
    )
    
    # 创建数据对象
    pose_data = PoseEstimationData(
        camera_intrinsics=intrinsics,
        camera_poses=poses,
        point_cloud=points,
        point_cloud_rgb=colors,
    )
    
    # 可选: 保存为numpy文件
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "camera_intrinsics.npy"), intrinsics)
        np.save(os.path.join(output_dir, "camera_poses.npy"), poses)
        np.save(os.path.join(output_dir, "point_cloud.npy"), points)
        np.save(os.path.join(output_dir, "point_cloud_rgb.npy"), colors)
        print(f"已保存数据到: {output_dir}")
    
    return pose_data


def train_with_custom_pose(
    data_dir: str,
    pose_data: PoseEstimationData,
    corr_data: CorrespondenceData = None,
    result_dir: str = "results/custom",
    **train_kwargs
):
    """
    使用自定义位姿数据训练
    
    Args:
        data_dir: 数据目录 (应包含images/和split文件)
        pose_data: 位姿数据
        corr_data: 对应关系数据 (可选)
        result_dir: 结果保存目录
        **train_kwargs: 传递给训练器的参数
    """
    
    from datasets.mast3r import Parser, Dataset, CorrespondenceDataset
    from trainer import Config, Runner
    
    # 创建配置
    cfg = Config(
        data_dir=data_dir,
        result_dir=result_dir,
        use_corres_epipolar_loss=(corr_data is not None),
        **train_kwargs
    )
    
    # 创建训练器
    runner = Runner(
        local_rank=0,
        world_rank=0,
        world_size=1,
        cfg=cfg
    )
    
    # 使用自定义位姿数据
    runner.parser = Parser(
        data_dir=data_dir,
        factor=cfg.data_factor,
        normalize=cfg.normalize_world_space,
        test_every=cfg.test_every,
        pose_data=pose_data,
    )
    
    # 重新创建数据集
    runner.trainset = Dataset(runner.parser, split="train")
    runner.trainvalset = Dataset(runner.parser, split="train")
    runner.valset = Dataset(runner.parser, split="val")
    runner.scene_scale = runner.parser.scene_scale * 1.1 * cfg.global_scale
    
    print(f"\n训练配置:")
    print(f"  - 训练图像: {len(runner.trainset)}")
    print(f"  - 验证图像: {len(runner.valset)}")
    print(f"  - 场景规模: {runner.scene_scale:.2f}")
    
    # 开始训练
    print("\n开始训练...")
    runner.train()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="使用自定义位姿估计器训练3r-gs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用步骤:
  1. 修改 your_pose_estimation_method() 函数,实现你的位姿估计
  2. (可选) 修改 your_feature_matching_method() 实现特征匹配
  3. 运行此脚本进行训练

示例:
  python custom_pose_estimator_template.py \
      --data_dir data/scene \
      --result_dir results/my_method
        """
    )
    
    parser.add_argument("--data_dir", type=str, required=True,
                       help="数据目录 (应包含images/目录)")
    parser.add_argument("--result_dir", type=str, default="results/custom",
                       help="结果保存目录")
    parser.add_argument("--save_pose_data", type=str, default=None,
                       help="保存位姿数据到指定目录")
    parser.add_argument("--skip_train", action="store_true",
                       help="只转换数据,不训练")
    
    args = parser.parse_args()
    
    # 检查数据目录
    image_dir = os.path.join(args.data_dir, "images")
    if not os.path.exists(image_dir):
        print(f"错误: 图像目录不存在: {image_dir}")
        sys.exit(1)
    
    # 转换数据
    print("=" * 60)
    print("步骤1: 运行位姿估计")
    print("=" * 60)
    pose_data = convert_to_interface(
        image_dir,
        output_dir=args.save_pose_data
    )
    
    # 是否跳过训练
    if args.skip_train:
        print("\n跳过训练 (--skip_train)")
        sys.exit(0)
    
    # 训练
    print("\n" + "=" * 60)
    print("步骤2: 开始训练")
    print("=" * 60)
    train_with_custom_pose(
        data_dir=args.data_dir,
        pose_data=pose_data,
        result_dir=args.result_dir,
        data_factor=2,
        max_steps=30000,
    )
    
    print("\n完成!")

