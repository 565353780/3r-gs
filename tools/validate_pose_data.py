#!/usr/bin/env python3
"""
位姿数据验证工具

这个工具帮助你验证PoseEstimationData的格式是否正确,
并提供可视化来检查数据质量。
"""

import os
import sys
import numpy as np
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets.pose_data_interface import PoseEstimationData


def validate_intrinsics(intrinsics: np.ndarray) -> list:
    """验证内参矩阵"""
    issues = []
    
    # 检查形状
    if intrinsics.shape[1:] != (3, 3):
        issues.append(f"❌ 内参形状错误: {intrinsics.shape}, 应该是 [N, 3, 3]")
        return issues
    
    # 检查每个内参矩阵
    for i, K in enumerate(intrinsics[:5]):  # 检查前5个
        # 检查对角线值
        fx, fy = K[0, 0], K[1, 1]
        if fx <= 0 or fy <= 0:
            issues.append(f"❌ 内参[{i}] 焦距非正: fx={fx}, fy={fy}")
        
        # 检查主点
        cx, cy = K[0, 2], K[1, 2]
        if cx < 0 or cy < 0:
            issues.append(f"⚠️  内参[{i}] 主点异常: cx={cx}, cy={cy}")
        
        # 检查格式
        if not np.isclose(K[2, 2], 1.0):
            issues.append(f"❌ 内参[{i}] 第(2,2)元素应该是1: {K[2, 2]}")
        
        if not np.allclose(K[[0, 1, 2], [1, 0, 0]], 0):
            issues.append(f"⚠️  内参[{i}] 非零的非对角元素: {K}")
    
    if not issues:
        issues.append(f"✅ 内参矩阵格式正确 (共{len(intrinsics)}个)")
        print(f"   示例内参[0]:\n{intrinsics[0]}")
    
    return issues


def validate_poses(poses: np.ndarray) -> list:
    """验证位姿矩阵"""
    issues = []
    
    # 检查形状
    if poses.shape[1:] != (4, 4):
        issues.append(f"❌ 位姿形状错误: {poses.shape}, 应该是 [N, 4, 4]")
        return issues
    
    # 检查每个位姿矩阵
    for i, pose in enumerate(poses[:5]):  # 检查前5个
        # 检查最后一行
        if not np.allclose(pose[3, :], [0, 0, 0, 1]):
            issues.append(f"❌ 位姿[{i}] 最后一行应该是 [0,0,0,1]: {pose[3,:]}")
        
        # 检查旋转矩阵
        R = pose[:3, :3]
        
        # 行列式应该接近1
        det = np.linalg.det(R)
        if not np.isclose(det, 1.0, atol=1e-2):
            issues.append(f"❌ 位姿[{i}] 旋转矩阵行列式={det:.4f}, 应该接近1")
        
        # 应该是正交矩阵
        orthogonality = np.max(np.abs(R @ R.T - np.eye(3)))
        if orthogonality > 1e-2:
            issues.append(f"❌ 位姿[{i}] 旋转矩阵不正交, 误差={orthogonality:.4f}")
        
        # 检查平移
        t = pose[:3, 3]
        if np.linalg.norm(t) > 1000:
            issues.append(f"⚠️  位姿[{i}] 平移向量很大: {t}")
    
    if not issues:
        issues.append(f"✅ 位姿矩阵格式正确 (共{len(poses)}个)")
        
        # 统计位姿分布
        positions = poses[:, :3, 3]
        center = positions.mean(axis=0)
        distances = np.linalg.norm(positions - center, axis=1)
        
        print(f"   相机位置统计:")
        print(f"   - 中心: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
        print(f"   - 到中心距离: {distances.min():.2f} ~ {distances.max():.2f}")
    
    return issues


def validate_point_cloud(points: np.ndarray, colors: np.ndarray) -> list:
    """验证点云"""
    issues = []
    
    # 检查形状
    if points.shape[1] != 3:
        issues.append(f"❌ 点云形状错误: {points.shape}, 应该是 [M, 3]")
        return issues
    
    if colors.shape != points.shape:
        issues.append(f"❌ 颜色形状与点云不匹配: {colors.shape} vs {points.shape}")
    
    # 检查点云范围
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    ranges = maxs - mins
    
    if np.any(ranges == 0):
        issues.append(f"⚠️  点云在某些维度上是平的: {ranges}")
    
    # 检查颜色范围
    if colors.min() < 0 or colors.max() > 255:
        issues.append(f"⚠️  颜色值超出[0,255]范围: [{colors.min()}, {colors.max()}]")
    
    # 检查是否有无效值
    if np.any(np.isnan(points)):
        issues.append(f"❌ 点云包含NaN值")
    if np.any(np.isinf(points)):
        issues.append(f"❌ 点云包含Inf值")
    
    if not issues:
        issues.append(f"✅ 点云格式正确 (共{len(points)}个点)")
        print(f"   点云范围:")
        print(f"   - X: [{mins[0]:.2f}, {maxs[0]:.2f}]")
        print(f"   - Y: [{mins[1]:.2f}, {maxs[1]:.2f}]")
        print(f"   - Z: [{mins[2]:.2f}, {maxs[2]:.2f}]")
    
    return issues


def validate_pose_data(pose_data: PoseEstimationData) -> bool:
    """完整验证PoseEstimationData"""
    
    print("=" * 60)
    print("开始验证PoseEstimationData")
    print("=" * 60)
    
    all_issues = []
    
    # 验证内参
    print("\n📷 验证相机内参...")
    issues = validate_intrinsics(pose_data.camera_intrinsics)
    for issue in issues:
        print(f"  {issue}")
    all_issues.extend([i for i in issues if i.startswith("❌")])
    
    # 验证位姿
    print("\n🎯 验证相机位姿...")
    issues = validate_poses(pose_data.camera_poses)
    for issue in issues:
        print(f"  {issue}")
    all_issues.extend([i for i in issues if i.startswith("❌")])
    
    # 验证点云
    print("\n☁️  验证点云...")
    issues = validate_point_cloud(pose_data.point_cloud, pose_data.point_cloud_rgb)
    for issue in issues:
        print(f"  {issue}")
    all_issues.extend([i for i in issues if i.startswith("❌")])
    
    # 检查数量匹配
    print("\n🔢 验证数量...")
    num_cameras = len(pose_data.camera_intrinsics)
    num_poses = len(pose_data.camera_poses)
    
    if num_cameras != num_poses:
        print(f"  ❌ 内参数量({num_cameras})与位姿数量({num_poses})不匹配")
        all_issues.append("数量不匹配")
    else:
        print(f"  ✅ 相机数量一致: {num_cameras}")
    
    # 总结
    print("\n" + "=" * 60)
    if all_issues:
        print(f"❌ 发现 {len(all_issues)} 个错误!")
        print("请修复这些问题后再使用数据训练。")
        return False
    else:
        print("✅ 所有验证通过!")
        print("数据格式正确,可以用于训练。")
        return True


def visualize_pose_data(pose_data: PoseEstimationData, save_path: str = None):
    """可视化位姿数据"""
    
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("需要matplotlib进行可视化: pip install matplotlib")
        return
    
    print("\n🎨 生成可视化...")
    
    fig = plt.figure(figsize=(15, 10))
    
    # 子图1: 3D位姿和点云
    ax1 = fig.add_subplot(221, projection='3d')
    
    # 采样点云
    num_points = len(pose_data.point_cloud)
    sample_size = min(2000, num_points)
    indices = np.random.choice(num_points, sample_size, replace=False)
    sampled_points = pose_data.point_cloud[indices]
    sampled_colors = pose_data.point_cloud_rgb[indices] / 255.0
    
    ax1.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2],
               c=sampled_colors, s=1, alpha=0.3, label='点云')
    
    # 相机位置
    camera_positions = pose_data.camera_poses[:, :3, 3]
    ax1.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2],
               c='red', s=100, marker='^', label='相机', edgecolors='black')
    
    # 相机朝向
    for i in range(0, len(pose_data.camera_poses), max(1, len(pose_data.camera_poses)//20)):
        pose = pose_data.camera_poses[i]
        pos = pose[:3, 3]
        # 相机朝向是-Z方向
        forward = pose[:3, :3] @ np.array([0, 0, 1])
        ax1.quiver(pos[0], pos[1], pos[2],
                  forward[0], forward[1], forward[2],
                  length=0.3, color='blue', alpha=0.6, arrow_length_ratio=0.3)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('相机位姿和点云')
    ax1.legend()
    
    # 子图2: 相机轨迹俯视图
    ax2 = fig.add_subplot(222)
    ax2.plot(camera_positions[:, 0], camera_positions[:, 1], 'r-', alpha=0.5)
    ax2.scatter(camera_positions[:, 0], camera_positions[:, 1], c='red', s=50, marker='^')
    ax2.scatter(sampled_points[:, 0], sampled_points[:, 1], c=sampled_colors, s=0.5, alpha=0.3)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('俯视图 (X-Y平面)')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 焦距分布
    ax3 = fig.add_subplot(223)
    focal_lengths = []
    for K in pose_data.camera_intrinsics:
        fx, fy = K[0, 0], K[1, 1]
        focal_lengths.append([fx, fy])
    focal_lengths = np.array(focal_lengths)
    
    ax3.plot(focal_lengths[:, 0], label='fx', marker='o')
    ax3.plot(focal_lengths[:, 1], label='fy', marker='s')
    ax3.set_xlabel('相机索引')
    ax3.set_ylabel('焦距 (像素)')
    ax3.set_title('相机焦距变化')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 相机间距离分布
    ax4 = fig.add_subplot(224)
    distances = []
    for i in range(len(camera_positions) - 1):
        dist = np.linalg.norm(camera_positions[i+1] - camera_positions[i])
        distances.append(dist)
    
    ax4.hist(distances, bins=30, edgecolor='black')
    ax4.set_xlabel('相邻相机距离')
    ax4.set_ylabel('频数')
    ax4.set_title('相机间距分布')
    ax4.axvline(np.mean(distances), color='red', linestyle='--', 
                label=f'均值: {np.mean(distances):.2f}')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ 保存可视化到: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse
    import pickle
    
    parser = argparse.ArgumentParser(
        description="验证和可视化PoseEstimationData",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

1. 验证numpy文件:
   python validate_pose_data.py \\
       --intrinsics data/camera_intrinsics.npy \\
       --poses data/camera_poses.npy \\
       --points data/point_cloud.npy \\
       --colors data/point_cloud_rgb.npy

2. 验证pickle文件:
   python validate_pose_data.py --pickle pose_data.pkl

3. 验证并可视化:
   python validate_pose_data.py --pickle pose_data.pkl --visualize

4. 保存可视化图片:
   python validate_pose_data.py --pickle pose_data.pkl \\
       --visualize --save_viz validation.png
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pickle", type=str,
                      help="PoseEstimationData pickle文件")
    group.add_argument("--intrinsics", type=str,
                      help="内参numpy文件")
    
    parser.add_argument("--poses", type=str,
                       help="位姿numpy文件")
    parser.add_argument("--points", type=str,
                       help="点云numpy文件")
    parser.add_argument("--colors", type=str,
                       help="颜色numpy文件")
    parser.add_argument("--visualize", action="store_true",
                       help="生成可视化")
    parser.add_argument("--save_viz", type=str,
                       help="保存可视化图片路径")
    
    args = parser.parse_args()
    
    # 加载数据
    if args.pickle:
        print(f"从pickle文件加载: {args.pickle}")
        with open(args.pickle, 'rb') as f:
            pose_data = pickle.load(f)
    else:
        print(f"从numpy文件加载...")
        intrinsics = np.load(args.intrinsics)
        poses = np.load(args.poses)
        points = np.load(args.points)
        colors = np.load(args.colors) if args.colors else np.random.rand(*points.shape) * 255
        
        pose_data = PoseEstimationData(
            camera_intrinsics=intrinsics,
            camera_poses=poses,
            point_cloud=points,
            point_cloud_rgb=colors,
        )
    
    # 验证
    is_valid = validate_pose_data(pose_data)
    
    # 可视化
    if args.visualize or args.save_viz:
        if is_valid:
            visualize_pose_data(pose_data, save_path=args.save_viz)
        else:
            print("\n⚠️  数据验证失败,跳过可视化")
    
    sys.exit(0 if is_valid else 1)

