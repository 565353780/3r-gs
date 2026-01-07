#!/usr/bin/env python3
"""
使用自定义位姿估计模块的训练脚本

这个脚本展示了如何使用自定义的相机位姿估计和特征匹配模块来训练3D高斯喷溅。
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import torch
import tyro
from dataclasses import dataclass
from typing import Optional

from datasets.pose_data_interface import (
    PoseEstimationData, 
    CorrespondenceData, 
    load_from_mast3r_directory
)
from datasets.mast3r import Parser, Dataset, CorrespondenceDataset
from trainer import Config, Runner


@dataclass
class CustomConfig(Config):
    """扩展配置,支持自定义位姿数据"""
    
    # 自定义位姿数据来源
    # 选项: "mast3r" (从mast3r目录加载), "colmap" (从colmap加载), "custom" (自定义实现)
    pose_source: str = "mast3r"
    
    # 如果使用colmap,指定colmap目录
    colmap_dir: Optional[str] = None


def load_pose_data_from_colmap(colmap_dir: str, image_dir: str) -> PoseEstimationData:
    """
    从COLMAP输出加载位姿数据
    
    Args:
        colmap_dir: COLMAP sparse目录 (例如: "data/scene/sparse/0")
        image_dir: 图像目录,用于确定图像顺序
        
    Returns:
        PoseEstimationData对象
    """
    from pycolmap import SceneManager
    from scipy.spatial.transform import Rotation
    
    print(f"从COLMAP目录加载: {colmap_dir}")
    
    manager = SceneManager(colmap_dir)
    manager.load_cameras()
    manager.load_images()
    manager.load_points3D()
    
    # 按图像名称排序,确保顺序一致
    image_ids = sorted(manager.images.keys(), 
                      key=lambda x: manager.images[x].name)
    
    intrinsics_list = []
    poses_list = []
    
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
    
    # 提取3D点云
    points = []
    colors = []
    errors = []
    
    for pt_id, point in manager.points3D.items():
        points.append(point.xyz)
        colors.append(point.color)
        errors.append(point.error)
    
    print(f"加载了 {len(poses_list)} 个相机位姿和 {len(points)} 个3D点")
    
    return PoseEstimationData(
        camera_intrinsics=np.array(intrinsics_list, dtype=np.float32),
        camera_poses=np.array(poses_list, dtype=np.float32),
        point_cloud=np.array(points, dtype=np.float32),
        point_cloud_rgb=np.array(colors, dtype=np.float32),
        point_cloud_errors=np.array(errors, dtype=np.float32),
    )


def load_pose_data_custom() -> PoseEstimationData:
    """
    使用自定义方法加载位姿数据
    
    在这里实现你自己的位姿估计逻辑:
    - 调用自定义SLAM系统
    - 使用神经网络位姿估计
    - 从其他格式转换
    
    Returns:
        PoseEstimationData对象
    """
    # TODO: 实现你自己的位姿加载逻辑
    
    # 示例: 生成随机数据 (仅用于演示)
    num_cameras = 10
    num_points = 1000
    
    # 随机生成相机内参
    intrinsics = np.tile(
        np.array([[800, 0, 400], [0, 800, 300], [0, 0, 1]], dtype=np.float32),
        (num_cameras, 1, 1)
    )
    
    # 随机生成相机位姿 (环形排列)
    poses = []
    for i in range(num_cameras):
        angle = 2 * np.pi * i / num_cameras
        radius = 5.0
        
        # 相机位置
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 1.0
        
        # 看向原点的旋转
        forward = -np.array([x, y, z])
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, np.array([0, 0, 1]))
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        R = np.stack([right, up, -forward], axis=1)
        t = np.array([x, y, z])
        
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R
        c2w[:3, 3] = t
        poses.append(c2w)
    
    # 随机生成点云
    points = np.random.randn(num_points, 3).astype(np.float32) * 2
    colors = np.random.rand(num_points, 3).astype(np.float32) * 255
    
    print(f"生成了 {num_cameras} 个相机和 {num_points} 个点")
    print("警告: 这是随机数据,仅用于演示!")
    
    return PoseEstimationData(
        camera_intrinsics=intrinsics,
        camera_poses=np.array(poses, dtype=np.float32),
        point_cloud=points,
        point_cloud_rgb=colors,
    )


def main(cfg: CustomConfig):
    """主训练函数"""
    
    # 根据配置加载位姿数据
    pose_data = None
    corr_data = None
    
    if cfg.pose_source == "mast3r":
        print("使用MASt3R数据...")
        # 从MASt3R目录加载
        mast3r_dir = os.path.join(cfg.data_dir, "mast3r/0")
        if not os.path.exists(mast3r_dir):
            mast3r_dir = os.path.join(cfg.data_dir, "mast3r")
        
        if os.path.exists(mast3r_dir):
            pose_data, corr_data = load_from_mast3r_directory(mast3r_dir)
        else:
            print(f"MASt3R目录不存在: {mast3r_dir}")
            print("将使用传统方式从文件加载")
            pose_data = None  # 使用传统加载方式
            
    elif cfg.pose_source == "colmap":
        print("使用COLMAP数据...")
        if cfg.colmap_dir is None:
            cfg.colmap_dir = os.path.join(cfg.data_dir, "sparse/0")
        
        pose_data = load_pose_data_from_colmap(
            cfg.colmap_dir,
            os.path.join(cfg.data_dir, "images")
        )
        # COLMAP通常不提供对应关系,设为None
        corr_data = None
        # 禁用对应关系损失
        cfg.use_corres_epipolar_loss = False
        
    elif cfg.pose_source == "custom":
        print("使用自定义位姿数据...")
        pose_data = load_pose_data_custom()
        corr_data = None
        cfg.use_corres_epipolar_loss = False
        
    else:
        raise ValueError(f"未知的pose_source: {cfg.pose_source}")
    
    # 创建Runner
    print("\n初始化训练器...")
    runner = Runner(
        local_rank=0,
        world_rank=0,
        world_size=1,
        cfg=cfg,
    )
    
    # 如果使用自定义位姿数据,替换parser
    if pose_data is not None:
        print("使用自定义位姿数据初始化Parser...")
        runner.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
            pose_data=pose_data,
        )
        
        # 重新创建datasets
        print("重新创建训练数据集...")
        runner.trainset = Dataset(
            runner.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )
        runner.trainvalset = Dataset(runner.parser, split="train")
        runner.valset = Dataset(runner.parser, split="val")
        runner.scene_scale = runner.parser.scene_scale * 1.1 * cfg.global_scale
        
        print(f"场景规模: {runner.scene_scale}")
        print(f"训练图像数: {len(runner.trainset)}")
        print(f"验证图像数: {len(runner.valset)}")
    
    # 开始训练
    print("\n开始训练...\n")
    runner.train()
    
    print("\n训练完成!")


if __name__ == "__main__":
    """
    使用示例:
    
    # 1. 使用MASt3R数据 (向后兼容)
    python train_custom_pose.py --data_dir data/scene --pose_source mast3r
    
    # 2. 使用COLMAP数据
    python train_custom_pose.py --data_dir data/scene --pose_source colmap --colmap_dir data/scene/sparse/0
    
    # 3. 使用自定义数据
    python train_custom_pose.py --data_dir data/scene --pose_source custom
    
    # 4. 完整配置示例
    CUDA_VISIBLE_DEVICES=0 python train_custom_pose.py \
        --data_dir data/scene \
        --pose_source colmap \
        --colmap_dir data/scene/sparse/0 \
        --data_factor 2 \
        --result_dir results/custom_exp \
        --max_steps 30000
    """
    
    # 配置可选项
    configs = {
        "default": (
            "使用默认策略训练",
            CustomConfig(
                strategy=tyro.conf.FlagConversionOff[DefaultStrategy](verbose=True),
            ),
        ),
        "mcmc": (
            "使用MCMC策略训练",
            CustomConfig(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=tyro.conf.FlagConversionOff[MCMCStrategy](verbose=True),
            ),
        ),
    }
    
    # 解析命令行参数
    from gsplat.strategy import DefaultStrategy, MCMCStrategy
    cfg = tyro.cli(CustomConfig)
    cfg.adjust_steps(cfg.steps_scaler)
    
    # 运行训练
    main(cfg)

