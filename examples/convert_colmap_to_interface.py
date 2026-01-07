#!/usr/bin/env python3
"""
示例: 将COLMAP输出转换为通用数据接口

这个脚本展示如何从COLMAP稀疏重建结果转换为3r-gs的通用数据格式。
你可以参考这个脚本实现其他位姿估计方法的转换。
"""

import os
import sys
import numpy as np
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets.pose_data_interface import PoseEstimationData


def colmap_to_pose_data(colmap_sparse_dir: str, verbose: bool = True) -> PoseEstimationData:
    """
    从COLMAP稀疏重建目录转换为PoseEstimationData
    
    Args:
        colmap_sparse_dir: COLMAP sparse目录路径 (例如: "data/scene/sparse/0")
        verbose: 是否打印详细信息
        
    Returns:
        PoseEstimationData对象
    """
    try:
        from pycolmap import SceneManager
        from scipy.spatial.transform import Rotation
    except ImportError as e:
        print(f"错误: 缺少依赖库: {e}")
        print("请安装: pip install pycolmap scipy")
        sys.exit(1)
    
    if verbose:
        print(f"从COLMAP目录加载: {colmap_sparse_dir}")
    
    # 加载COLMAP数据
    manager = SceneManager(colmap_sparse_dir)
    manager.load_cameras()
    manager.load_images()
    manager.load_points3D()
    
    if verbose:
        print(f"  - 相机数量: {len(manager.cameras)}")
        print(f"  - 图像数量: {len(manager.images)}")
        print(f"  - 3D点数量: {len(manager.points3D)}")
    
    # 按图像名称排序,确保顺序一致
    image_ids = sorted(manager.images.keys(), 
                      key=lambda x: manager.images[x].name)
    
    # 提取相机内参和位姿
    intrinsics_list = []
    poses_list = []
    image_names = []
    
    for img_id in image_ids:
        image = manager.images[img_id]
        cam = manager.cameras[image.camera_id]
        
        # 提取内参
        K = np.array([
            [cam.fx, 0, cam.cx],
            [0, cam.fy, cam.cy],
            [0, 0, 1]
        ], dtype=np.float32)
        intrinsics_list.append(K)
        
        # COLMAP使用world-to-camera格式,需要转换为camera-to-world
        # qvec格式: [qw, qx, qy, qz]
        # 转换为 [qx, qy, qz, qw] 供scipy使用
        R = Rotation.from_quat([
            image.qvec[1], image.qvec[2], image.qvec[3], image.qvec[0]
        ]).as_matrix()
        t = image.tvec
        
        # 构建world-to-camera矩阵
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R
        w2c[:3, 3] = t
        
        # 转换为camera-to-world
        c2w = np.linalg.inv(w2c)
        poses_list.append(c2w)
        image_names.append(image.name)
    
    # 提取3D点云
    points = []
    colors = []
    errors = []
    
    for pt_id, point in manager.points3D.items():
        points.append(point.xyz)
        colors.append(point.color)
        errors.append(point.error)
    
    if verbose:
        print(f"\n转换结果:")
        print(f"  - 内参矩阵: {len(intrinsics_list)} 个")
        print(f"  - 位姿矩阵: {len(poses_list)} 个")
        print(f"  - 点云: {len(points)} 个点")
        print(f"\n示例内参矩阵:")
        print(intrinsics_list[0])
        print(f"\n示例位姿矩阵 (camera-to-world):")
        print(poses_list[0])
    
    # 创建PoseEstimationData对象
    pose_data = PoseEstimationData(
        camera_intrinsics=np.array(intrinsics_list, dtype=np.float32),
        camera_poses=np.array(poses_list, dtype=np.float32),
        point_cloud=np.array(points, dtype=np.float32),
        point_cloud_rgb=np.array(colors, dtype=np.float32),
        point_cloud_errors=np.array(errors, dtype=np.float32),
    )
    
    if verbose:
        print("\n✓ 转换成功!")
    
    return pose_data


def save_pose_data(pose_data: PoseEstimationData, output_dir: str):
    """
    保存PoseEstimationData到文件
    
    Args:
        pose_data: PoseEstimationData对象
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, "camera_intrinsics.npy"), 
            pose_data.camera_intrinsics)
    np.save(os.path.join(output_dir, "camera_poses.npy"), 
            pose_data.camera_poses)
    np.save(os.path.join(output_dir, "point_cloud.npy"), 
            pose_data.point_cloud)
    np.save(os.path.join(output_dir, "point_cloud_rgb.npy"), 
            pose_data.point_cloud_rgb)
    np.save(os.path.join(output_dir, "point_cloud_errors.npy"), 
            pose_data.point_cloud_errors)
    
    print(f"已保存数据到: {output_dir}")


def visualize_poses_and_points(pose_data: PoseEstimationData):
    """
    可视化相机位姿和点云
    
    Args:
        pose_data: PoseEstimationData对象
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("需要matplotlib进行可视化: pip install matplotlib")
        return
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制点云 (采样显示)
    num_points = len(pose_data.point_cloud)
    sample_indices = np.random.choice(num_points, min(1000, num_points), replace=False)
    sampled_points = pose_data.point_cloud[sample_indices]
    sampled_colors = pose_data.point_cloud_rgb[sample_indices] / 255.0
    
    ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2],
              c=sampled_colors, s=1, alpha=0.5, label='点云')
    
    # 绘制相机位姿
    camera_positions = pose_data.camera_poses[:, :3, 3]
    ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2],
              c='red', s=100, marker='^', label='相机')
    
    # 绘制相机方向
    for i, pose in enumerate(pose_data.camera_poses[::5]):  # 每5个绘制一个
        pos = pose[:3, 3]
        forward = pose[:3, :3] @ np.array([0, 0, 1])
        ax.quiver(pos[0], pos[1], pos[2],
                 forward[0], forward[1], forward[2],
                 length=0.5, color='blue', alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('相机位姿和点云可视化')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="将COLMAP输出转换为3r-gs通用数据格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本转换
  python convert_colmap_to_interface.py --colmap_dir data/scene/sparse/0
  
  # 转换并保存
  python convert_colmap_to_interface.py \
      --colmap_dir data/scene/sparse/0 \
      --output_dir data/scene/pose_data
  
  # 转换、保存并可视化
  python convert_colmap_to_interface.py \
      --colmap_dir data/scene/sparse/0 \
      --output_dir data/scene/pose_data \
      --visualize
        """
    )
    
    parser.add_argument("--colmap_dir", type=str, required=True,
                       help="COLMAP稀疏重建目录 (例如: data/scene/sparse/0)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="保存转换后数据的目录 (可选)")
    parser.add_argument("--visualize", action="store_true",
                       help="可视化相机位姿和点云")
    parser.add_argument("--quiet", action="store_true",
                       help="静默模式,不打印详细信息")
    
    args = parser.parse_args()
    
    # 检查COLMAP目录是否存在
    if not os.path.exists(args.colmap_dir):
        print(f"错误: COLMAP目录不存在: {args.colmap_dir}")
        sys.exit(1)
    
    # 转换数据
    pose_data = colmap_to_pose_data(args.colmap_dir, verbose=not args.quiet)
    
    # 保存数据
    if args.output_dir:
        save_pose_data(pose_data, args.output_dir)
    
    # 可视化
    if args.visualize:
        visualize_poses_and_points(pose_data)
    
    print("\n完成!")

