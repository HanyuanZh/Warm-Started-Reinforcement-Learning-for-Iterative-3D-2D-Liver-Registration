import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import glob
import os
import re
import random
from typing import Optional, Tuple, Union


class EnhancedDepthAugmenter:
    """
    Enhanced depth augmenter with customizable aspect ratio rectangles,
    depth normalization, and scale perturbation
    """

    def __init__(self,
                 # Original occlusion parameters
                 occluder_probability: float = 0.4,
                 random_erasing_probability: float = 0.4,
                 erasing_ratio: tuple = (0.05, 0.25),
                 # Rectangle parameters
                 length_range: tuple = (100, 400),  # Length range
                 width_range: tuple = (8, 25),  # Width range
                 angle_range: tuple = (-45, 45),  # Angle range
                 # New depth perturbation parameters
                 depth_normalization_probability: float = 0.5,
                 scale_perturbation_probability: float = 0.6,
                 # Depth normalization parameters
                 norm_min_range: tuple = (0.0, 0.2),  # Minimum normalized value range
                 norm_max_range: tuple = (0.8, 1.0),  # Maximum normalized value range
                 # Scale perturbation parameters
                 scale_factor_range: tuple = (0.7, 1.3),  # Scale factor range
                 depth_shift_range: tuple = (-30, 30),  # Depth shift range
                 noise_std_range: tuple = (0.01, 0.05)):  # Gaussian noise std range

        # Original parameters
        self.occluder_prob = occluder_probability
        self.random_erasing_prob = random_erasing_probability
        self.erasing_ratio = erasing_ratio
        self.length_range = length_range
        self.width_range = width_range
        self.angle_range = angle_range

        # New depth perturbation parameters
        self.depth_norm_prob = depth_normalization_probability
        self.scale_pert_prob = scale_perturbation_probability
        self.norm_min_range = norm_min_range
        self.norm_max_range = norm_max_range
        self.scale_factor_range = scale_factor_range
        self.depth_shift_range = depth_shift_range
        self.noise_std_range = noise_std_range

    def create_rotated_rectangle_mask(self, height: int, width: int) -> np.ndarray:
        """Create rotated rectangle occlusion mask"""
        mask = np.zeros((height, width), dtype=np.uint8)

        # Random rectangle parameters, ensure not exceeding image size
        max_length = min(self.length_range[1], min(height, width) - 20)
        min_length = min(self.length_range[0], max_length)

        if min_length >= max_length:
            min_length = max_length - 1

        length = random.randint(min_length, max_length)

        # Safe width calculation with proper bounds checking
        max_allowed_width = min(height, width) // 3
        max_width = min(self.width_range[1], max_allowed_width)
        min_width = min(self.width_range[0], max_width)

        if min_width >= max_width:
            min_width = max_width - 1

        if min_width < 1:
            min_width = 1
        if max_width < min_width:
            max_width = min_width + 1

        rect_width = random.randint(min_width, max_width)
        angle = random.uniform(self.angle_range[0], self.angle_range[1])

        # Safe random position calculation
        margin = max(length // 2, rect_width // 2) + 20  # Increased margin

        center_x_min = max(margin, length // 2)
        center_x_max = max(center_x_min + 1, width - margin)
        center_y_min = max(margin, length // 2)
        center_y_max = max(center_y_min + 1, height - margin)

        # Ensure valid ranges
        if center_x_min >= center_x_max:
            center_x_max = center_x_min + 1
        if center_y_min >= center_y_max:
            center_y_max = center_y_min + 1

        center_x = random.randint(center_x_min, center_x_max)
        center_y = random.randint(center_y_min, center_y_max)

        # Draw rotated rectangle
        self._draw_rotated_rectangle(mask, center_x, center_y, length, rect_width, angle)

        return mask

    def _draw_rotated_rectangle(self, mask: np.ndarray, center_x: int, center_y: int,
                                length: int, width: int, angle: float):
        """Draw a rotated rectangle"""
        # Convert angle to radians
        angle_rad = np.radians(angle)

        # Create four vertices of the rectangle
        half_length = length // 2
        half_width = width // 2

        # Original vertices (centered at origin)
        points = np.array([
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width]
        ])

        # Rotation matrix
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle]
        ])

        # Apply rotation
        rotated_points = points @ rotation_matrix.T

        # Translate to specified center position
        rotated_points[:, 0] += center_x
        rotated_points[:, 1] += center_y

        # Convert to integers and ensure within image bounds
        rotated_points = np.round(rotated_points).astype(np.int32)

        # Boundary check
        h, w = mask.shape
        valid_points = []
        for point in rotated_points:
            x, y = point
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            valid_points.append([x, y])

        valid_points = np.array(valid_points, dtype=np.int32)

        # Draw filled polygon
        if len(valid_points) >= 3:  # Need at least 3 points to draw polygon
            cv2.fillPoly(mask, [valid_points], 255)

    def apply_rectangle_occluder(self, depth_image: np.ndarray) -> np.ndarray:
        """Apply rectangle occlusion with 0 value replacement"""
        if depth_image.ndim == 3:
            depth_image = depth_image.squeeze()

        height, width = depth_image.shape
        result = depth_image.copy()

        # Generate 1-2 rectangle occlusions
        num_rectangles = random.randint(1, 2)

        for _ in range(num_rectangles):
            # Create rectangle occlusion mask
            rect_mask = self.create_rotated_rectangle_mask(height, width)

            # Replace occluded areas with 0 value
            result[rect_mask > 128] = 0

        return result

    def apply_random_erasing(self, depth_image: np.ndarray) -> np.ndarray:
        """Random erasing"""
        if depth_image.ndim == 3:
            depth_image = depth_image.squeeze()

        result = depth_image.copy()
        height, width = depth_image.shape

        erase_ratio = random.uniform(self.erasing_ratio[0], self.erasing_ratio[1])

        # Create erasing regions
        num_patches = random.randint(0, 2)

        for _ in range(num_patches):
            # Rectangle erasing
            patch_w = random.randint(width // 20, width // 8)
            patch_h = random.randint(height // 20, height // 8)
            x = random.randint(0, max(1, width - patch_w))
            y = random.randint(0, max(1, height - patch_h))
            result[y:y + patch_h, x:x + patch_w] = 0

        return result

    def apply_depth_normalization(self, depth_image: np.ndarray) -> np.ndarray:
        """Apply random depth normalization"""
        if depth_image.ndim == 3:
            depth_image = depth_image.squeeze()

        result = depth_image.copy().astype(np.float32)

        # 找出非零像素(忽略遮挡的0值)
        valid_mask = result > 0

        if not np.any(valid_mask):
            return result.astype(np.uint8)

        # 获取有效深度值的范围
        valid_depths = result[valid_mask]
        depth_min = valid_depths.min()
        depth_max = valid_depths.max()

        # 避免除零
        if depth_max == depth_min:
            return result.astype(np.uint8)

        # 随机选择归一化的目标范围
        target_min = random.uniform(self.norm_min_range[0], self.norm_min_range[1])
        target_max = random.uniform(self.norm_max_range[0], self.norm_max_range[1])

        # 确保target_max > target_min
        if target_max <= target_min:
            target_max = target_min + 0.1

        # 归一化到[0, 1]
        normalized = (result - depth_min) / (depth_max - depth_min)

        # 重新缩放到目标范围
        normalized = normalized * (target_max - target_min) + target_min

        # 转换回0-255范围
        result = normalized * 255

        # 保持遮挡区域为0
        result[~valid_mask] = 0

        return np.clip(result, 0, 255).astype(np.uint8)

    def apply_scale_perturbation(self, depth_image: np.ndarray) -> np.ndarray:
        """Apply random scale perturbation and depth shift"""
        if depth_image.ndim == 3:
            depth_image = depth_image.squeeze()

        result = depth_image.copy().astype(np.float32)

        # 找出非零像素
        valid_mask = result > 0

        if not np.any(valid_mask):
            return result.astype(np.uint8)

        # 随机缩放因子
        scale_factor = random.uniform(self.scale_factor_range[0], self.scale_factor_range[1])

        # 随机深度偏移
        depth_shift = random.uniform(self.depth_shift_range[0], self.depth_shift_range[1])

        # 随机噪声标准差
        noise_std = random.uniform(self.noise_std_range[0], self.noise_std_range[1])

        # 应用缩放和偏移
        result[valid_mask] = result[valid_mask] * scale_factor + depth_shift

        # 添加高斯噪声
        noise = np.random.normal(0, noise_std * 255, result.shape)
        result[valid_mask] += noise[valid_mask]

        # 限制在有效范围内
        result = np.clip(result, 0, 255)

        # 保持遮挡区域为0
        result[~valid_mask] = 0

        return result.astype(np.uint8)

    def __call__(self, depth_image: np.ndarray) -> np.ndarray:
        """Main augmentation function with all enhancements"""
        result = depth_image.copy()

        # Step 1: Randomly apply rectangle occlusion
        if random.random() < self.occluder_prob:
            result = self.apply_rectangle_occluder(result)

        # Step 2: Randomly apply random erasing
        if random.random() < self.random_erasing_prob:
            result = self.apply_random_erasing(result)

        # Step 3: Randomly apply depth normalization
        if random.random() < self.depth_norm_prob:
            result = self.apply_depth_normalization(result)

        # Step 4: Randomly apply scale perturbation
        if random.random() < self.scale_pert_prob:
            result = self.apply_scale_perturbation(result)

        return result


# 使用示例
if __name__ == "__main__":
    # 创建数据集实例，启用深度增强
    dataset = PairedImageMatrixDataset(
        mask_dir="path/to/masks",
        contour_dir="path/to/contours",
        depth_dir="path/to/depths",
        matrix_path="path/to/matrices.npy",
        num_pairs=2,
        apply_depth_augmentation=True,
        depth_aug_config={
            'occluder_probability': 0.5,
            'depth_normalization_probability': 0.6,
            'scale_perturbation_probability': 0.5,
            'length_range': (40, 120),  # 适应128x128尺寸
            'width_range': (3, 8),
        }
    )

    # 获取一个样本
    sample1, sample2 = dataset[0]
    print(f"Sample 1 combined shape: {sample1['combined'].shape}")
    print(f"Sample 1 matrix shape: {sample1['matrix'].shape}")
    print(f"Sample 2 combined shape: {sample2['combined'].shape}")
    print(f"Sample 2 matrix shape: {sample2['matrix'].shape}")