"""
通用相机位姿估计和特征匹配数据接口

这个模块定义了一个通用接口,用于替代MASt3R的硬编码依赖。
用户可以使用任何相机位姿估计模块(如MASt3R, COLMAP, NeRFStudio等)
和任何特征匹配模块(如SuperGlue, LoFTR等)生成数据。
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch


@dataclass
class PoseEstimationData:
    """
    相机位姿估计数据接口
    
    这个类封装了相机位姿估计模块的输出数据。
    无论使用MASt3R、COLMAP还是其他方法,都应该转换为这个格式。
    
    Attributes:
        camera_intrinsics: 相机内参矩阵 
            - Shape: [N, 3, 3] 
            - 格式: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            
        camera_poses: 相机到世界的变换矩阵 (camera-to-world)
            - Shape: [N, 4, 4]
            - 格式: SE(3)变换矩阵,包含旋转和平移
            
        point_cloud: 初始化点云的3D坐标
            - Shape: [M, 3]
            - 格式: [x, y, z]世界坐标系下的点
            
        point_cloud_rgb: 点云的RGB颜色 (可选)
            - Shape: [M, 3]
            - 格式: RGB值,范围[0, 255]
            - 如果为None,将使用随机颜色
            
        point_cloud_errors: 每个点的重投影误差 (可选)
            - Shape: [M,]
            - 如果为None,将使用零误差
    """
    camera_intrinsics: np.ndarray  # [N, 3, 3]
    camera_poses: np.ndarray  # [N, 4, 4]
    point_cloud: np.ndarray  # [M, 3]
    point_cloud_rgb: Optional[np.ndarray] = None  # [M, 3]
    point_cloud_errors: Optional[np.ndarray] = None  # [M,]
    
    def __post_init__(self):
        """验证数据格式"""
        assert self.camera_intrinsics.shape[1:] == (3, 3), \
            f"camera_intrinsics应该是[N, 3, 3], 但得到 {self.camera_intrinsics.shape}"
        assert self.camera_poses.shape[1:] == (4, 4), \
            f"camera_poses应该是[N, 4, 4], 但得到 {self.camera_poses.shape}"
        assert self.point_cloud.shape[1] == 3, \
            f"point_cloud应该是[M, 3], 但得到 {self.point_cloud.shape}"
        
        # 设置默认值
        if self.point_cloud_rgb is None:
            self.point_cloud_rgb = np.random.rand(*self.point_cloud.shape) * 255
        if self.point_cloud_errors is None:
            self.point_cloud_errors = np.zeros(len(self.point_cloud))
            
        assert self.point_cloud_rgb.shape == self.point_cloud.shape, \
            "point_cloud_rgb应该和point_cloud形状相同"
        assert len(self.point_cloud_errors) == len(self.point_cloud), \
            "point_cloud_errors应该和point_cloud点数相同"


@dataclass 
class CorrespondenceData:
    """
    图像对应关系数据接口
    
    这个类封装了特征匹配模块的输出数据。
    无论使用SuperGlue、LoFTR还是其他方法,都应该转换为这个格式。
    
    Attributes:
        ei: 图像对的第一个图像索引
            - Shape: [P,]
            - 每个元素是图像在数据集中的索引
            
        ej: 图像对的第二个图像索引
            - Shape: [P,]
            - 每个元素是图像在数据集中的索引
            
        corr_i: 第一个图像中的对应点位置(扁平化索引)
            - Shape: [P, K]
            - 对于512x512图像,索引 = y * 512 + x
            - K是每对图像的对应点数量
            
        corr_j: 第二个图像中的对应点位置(扁平化索引)
            - Shape: [P, K]
            - 格式同corr_i
            
        corr_mask: 对应点的有效性掩码
            - Shape: [P, K]
            - 1表示有效对应,0表示无效
            
        corr_weight: 对应点的置信度权重
            - Shape: [P, K]
            - 范围[0, 1],表示匹配的置信度
            
        corr_batch_idx: 批次索引 (可选)
            - Shape: [P, K]
            - 用于批处理,通常填充0
            
        depthmaps: 深度图 (可选)
            - Shape: [N, H, W]
            - N是图像数量,H和W是图像高度和宽度
            - 如果为None,将使用零深度图
            
        original_image_size: 原始图像尺寸,用于坐标转换
            - Tuple[int, int]: (width, height)
            - 默认为512x512
    """
    ei: np.ndarray  # [P,]
    ej: np.ndarray  # [P,]
    corr_i: np.ndarray  # [P, K]
    corr_j: np.ndarray  # [P, K]
    corr_mask: np.ndarray  # [P, K]
    corr_weight: np.ndarray  # [P, K]
    corr_batch_idx: Optional[np.ndarray] = None  # [P, K]
    depthmaps: Optional[np.ndarray] = None  # [N, H, W]
    original_image_size: tuple = (512, 512)  # (width, height)
    
    def __post_init__(self):
        """验证数据格式"""
        assert self.ei.shape == self.ej.shape, \
            "ei和ej应该有相同的形状"
        assert self.corr_i.shape == self.corr_j.shape, \
            "corr_i和corr_j应该有相同的形状"
        assert self.corr_i.shape == self.corr_mask.shape, \
            "corr_i和corr_mask应该有相同的形状"
        assert self.corr_i.shape == self.corr_weight.shape, \
            "corr_i和corr_weight应该有相同的形状"
        assert len(self.ei) == len(self.corr_i), \
            "ei的长度应该等于corr_i的第一维"
        
        # 设置默认值
        if self.corr_batch_idx is None:
            self.corr_batch_idx = np.zeros_like(self.corr_i)


def load_from_mast3r_directory(data_dir: str) -> tuple[PoseEstimationData, Optional[CorrespondenceData]]:
    """
    从MASt3R输出目录加载数据
    
    这是一个辅助函数,用于从现有的MASt3R输出格式转换为通用接口。
    
    Args:
        data_dir: MASt3R输出目录路径 (例如: "data/mast3r/0/")
        
    Returns:
        pose_data: PoseEstimationData对象
        corr_data: CorrespondenceData对象(如果对应关系文件存在)
    """
    import os
    from plyfile import PlyData
    
    # 加载位姿数据
    intrinsics = np.load(os.path.join(data_dir, 'camera_intrinsics.npy'))
    poses = np.load(os.path.join(data_dir, 'camera_poses.npy'))
    
    # 加载点云
    ply_data = PlyData.read(os.path.join(data_dir, 'pointcloud.ply'))
    point_cloud = ply_data['vertex']
    points = np.stack([point_cloud['x'], point_cloud['y'], point_cloud['z']], axis=-1)
    points_rgb = np.stack([point_cloud['red'], point_cloud['green'], point_cloud['blue']], axis=-1)
    
    pose_data = PoseEstimationData(
        camera_intrinsics=intrinsics,
        camera_poses=poses,
        point_cloud=points,
        point_cloud_rgb=points_rgb,
    )
    
    # 尝试加载对应关系数据
    corr_data = None
    try:
        corr_i = np.load(os.path.join(data_dir, 'corr_i.npy'))
        corr_j = np.load(os.path.join(data_dir, 'corr_j.npy'))
        corr_batch_idx = np.load(os.path.join(data_dir, 'corr_batch_idx.npy'))
        corr_mask = np.load(os.path.join(data_dir, 'corr_mask.npy'))
        corr_weight = np.load(os.path.join(data_dir, 'corr_weight.npy'))
        ei = np.load(os.path.join(data_dir, 'ei.npy'))
        ej = np.load(os.path.join(data_dir, 'ej.npy'))
        depthmaps = np.load(os.path.join(data_dir, 'depthmaps.npy'))
        
        corr_data = CorrespondenceData(
            ei=ei,
            ej=ej,
            corr_i=corr_i,
            corr_j=corr_j,
            corr_mask=corr_mask,
            corr_weight=corr_weight,
            corr_batch_idx=corr_batch_idx,
            depthmaps=depthmaps,
        )
    except FileNotFoundError:
        print("对应关系文件未找到,跳过加载")
    
    return pose_data, corr_data


def example_colmap_to_pose_data(colmap_dir: str) -> PoseEstimationData:
    """
    示例:从COLMAP输出转换为PoseEstimationData格式
    
    这是一个示例函数,展示如何从其他位姿估计方法转换数据。
    用户可以根据自己使用的方法编写类似的转换函数。
    
    Args:
        colmap_dir: COLMAP输出目录
        
    Returns:
        PoseEstimationData对象
    """
    from pycolmap import SceneManager
    
    # 这只是一个框架示例,需要根据实际情况完善
    manager = SceneManager(colmap_dir)
    manager.load_cameras()
    manager.load_images()
    manager.load_points3D()
    
    # 提取相机内参
    camera_intrinsics = []
    camera_poses = []
    
    for img_id, image in manager.images.items():
        cam = manager.cameras[image.camera_id]
        K = np.array([
            [cam.fx, 0, cam.cx],
            [0, cam.fy, cam.cy],
            [0, 0, 1]
        ])
        camera_intrinsics.append(K)
        
        # COLMAP使用world-to-camera,需要转换为camera-to-world
        # qvec是四元数,tvec是平移
        # 这里需要实现四元数到旋转矩阵的转换
        # pose = ...
        # camera_poses.append(pose)
    
    # 提取3D点
    points = []
    points_rgb = []
    for pt_id, point in manager.points3D.items():
        points.append(point.xyz)
        points_rgb.append(point.color)
    
    return PoseEstimationData(
        camera_intrinsics=np.array(camera_intrinsics),
        camera_poses=np.array(camera_poses),
        point_cloud=np.array(points),
        point_cloud_rgb=np.array(points_rgb),
    )

