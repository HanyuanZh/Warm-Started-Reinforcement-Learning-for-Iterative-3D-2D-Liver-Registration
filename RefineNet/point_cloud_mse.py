import vtk
import numpy as np
import torch
from typing import Union
import time


# ---------- 工具 1: 读取 VTK 点 ----------
def load_vtk_points(
        filename: str,
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cuda",
) -> torch.Tensor:
    """
    读取 VTK polydata，返回 (N, 3) Torch Tensor
    """
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()

    vtk_points = reader.GetOutput().GetPoints()
    n = vtk_points.GetNumberOfPoints()
    pts_np = np.array([vtk_points.GetPoint(i) for i in range(n)], dtype=np.float32)
    return torch.as_tensor(pts_np, dtype=dtype, device=device)


# ---------- 工具 2: 统一转 Tensor ----------
def as_tensor(
        arr: Union[np.ndarray, torch.Tensor],
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cuda",
) -> torch.Tensor:
    """确保输入变成指定 device/dtype 的 Torch Tensor（保留梯度）"""
    if isinstance(arr, torch.Tensor):
        return arr.to(device=device, dtype=dtype)
    elif isinstance(arr, np.ndarray):
        return torch.as_tensor(arr, dtype=dtype, device=device)
    else:
        raise TypeError(
            f"Expected torch.Tensor or np.ndarray, got {type(arr).__name__}"
        )


# ---------- 工具 3: 刚性 4×4 变换（原版本，保持向后兼容）----------
def apply_transform(
        points: Union[np.ndarray, torch.Tensor],
        transform_4x4: Union[np.ndarray, torch.Tensor],
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cuda",
) -> torch.Tensor:
    """
    points        : (N, 3)
    transform_4x4 : (4, 4)
    return        : (N, 3)  —— torch Tensor，可参与反向传播
    """
    pts = as_tensor(points, dtype=dtype, device=device)  # (N, 3)
    T = as_tensor(transform_4x4, dtype=dtype, device=device)  # (4, 4)

    ones = torch.ones((pts.shape[0], 1), dtype=pts.dtype, device=pts.device)
    homog = torch.cat([pts, ones], dim=1)  # (N, 4)
    transformed = (T @ homog.T).T  # (N, 4)
    return transformed[:, :3]  # (N, 3)


def apply_transform_batch(
        points: torch.Tensor,  # (B, N, 3)
        transforms: torch.Tensor,  # (B, 4, 4)
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cuda",
) -> torch.Tensor:
    """
    批量刚性变换：
      points     : (B, N, 3)
      transforms : (B, 4, 4)
    返回：
      (B, N, 3)，可参与反向传播
    """
    # 强制到同样的 device/dtype
    pts = points.to(dtype=dtype, device=device)
    Ts = transforms.to(dtype=dtype, device=device)

    # 检查形状
    assert pts.ndim == 3 and pts.shape[2] == 3, \
        f"Expected points (B, N, 3), got {tuple(pts.shape)}"
    assert Ts.ndim == 3 and Ts.shape[1:] == (4, 4), \
        f"Expected transforms (B, 4, 4), got {tuple(Ts.shape)}"

    B, N, _ = pts.shape
    # 构造齐次坐标
    ones = torch.ones((B, N, 1), device=pts.device, dtype=pts.dtype)
    homog = torch.cat([pts, ones], dim=2)  # (B, N, 4)
    homog_t = homog.permute(0, 2, 1)  # (B, 4, N)
    # 批量矩阵乘法
    transformed = Ts @ homog_t  # (B, 4, N)
    transformed = transformed.permute(0, 2, 1)  # (B, N, 4)
    return transformed[..., :3]  # (B, N, 3)


# ---------- 工具 3-FAST: 高效版刚性变换 ----------
def apply_transform_fast(
        points: torch.Tensor,  # (B, N, 3) 或 (N, 3)
        transforms: torch.Tensor,  # (B, 4, 4) 或 (4, 4)
) -> torch.Tensor:
    """
    高效版刚性变换，避免重复创建ones tensor和数据转换
    假设输入已经在正确的device和dtype上
    """
    if points.ndim == 2:
        # 单个变换 (N, 3) × (4, 4)
        N = points.shape[0]
        # 预分配齐次坐标
        homog = torch.empty((N, 4), device=points.device, dtype=points.dtype)
        homog[:, :3] = points
        homog[:, 3] = 1.0

        # 变换
        transformed = (transforms @ homog.T).T
        return transformed[:, :3]

    elif points.ndim == 3:
        # 批量变换 (B, N, 3) × (B, 4, 4)
        B, N = points.shape[:2]
        # 预分配齐次坐标
        homog = torch.empty((B, N, 4), device=points.device, dtype=points.dtype)
        homog[..., :3] = points
        homog[..., 3] = 1.0

        # 批量矩阵乘法
        transformed = torch.bmm(transforms, homog.transpose(-2, -1))
        return transformed.transpose(-2, -1)[..., :3]
    else:
        raise ValueError(f"Unsupported shape: {points.shape}")


# ---------- 工具 4: 3-D Surface MSE（原版本，保持向后兼容）----------
def compute_surface_mse(
        points1: Union[np.ndarray, torch.Tensor],
        points2: Union[np.ndarray, torch.Tensor],
        reduction: str = "mean",
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cuda",
) -> torch.Tensor:
    """
    Computes surface MSE between two point sets, supporting optional batching:
      points1, points2: (N,3) or (B,N,3)
    MSE = mean(‖p1 - p2‖²)

    reduction:
      "mean" → scalar loss over all points (and batches) (可反向)
      "batch" → per-batch scalar loss, shape (B,)
      "none" → per-point loss, shape (N,) or (B,N)
    """
    # Convert to tensor
    p1 = as_tensor(points1, dtype=dtype, device=device)
    p2 = as_tensor(points2, dtype=dtype, device=device)

    # Ensure shapes match
    if p1.shape != p2.shape:
        raise ValueError(f"Point sets must have the same shape, got {p1.shape} vs {p2.shape}")

    # Compute squared distances
    # Handle both (N,3) and (B,N,3)
    if p1.ndim == 2:
        # (N,3) → diff2: (N,)
        diff2 = (p1 - p2).pow(2).sum(dim=1)
    elif p1.ndim == 3:
        # (B,N,3) → diff2: (B,N)
        diff2 = (p1 - p2).pow(2).sum(dim=2)
    else:
        raise ValueError(f"Unsupported tensor dimension {p1.ndim}, expected 2 or 3")

    # Apply reduction
    if reduction == "mean":
        return diff2.mean()
    elif reduction == "batch":
        if p1.ndim != 3:
            raise ValueError("'batch' reduction requires batched input of shape (B,N,3)")
        return diff2.mean(dim=1)  # (B,)
    elif reduction == "none":
        return diff2
    else:
        raise ValueError(f"Unsupported reduction type '{reduction}', choose from 'mean','batch','none'")


# ---------- 工具 4-FAST: 高效版Surface MSE ----------
def compute_surface_mse_fast(
        points1: torch.Tensor,
        points2: torch.Tensor,
        reduction: str = "mean"
) -> torch.Tensor:
    """
    高效版本的Surface MSE，避免重复转换
    假设输入已经是正确device上的torch.Tensor

    Args:
        points1, points2: torch.Tensor, shape (N,3) or (B,N,3)
        reduction: "mean", "batch", "none"
    """
    # 直接计算，避免重复检查和转换
    diff = points1 - points2
    # 使用更高效的平方计算
    squared_diff = torch.sum(diff * diff, dim=-1)  # 比 .pow(2).sum() 快

    if reduction == "mean":
        return squared_diff.mean()
    elif reduction == "batch" and points1.ndim == 3:
        return squared_diff.mean(dim=1)  # (B,)
    elif reduction == "none":
        return squared_diff
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


# ---------- 工具 5: 训练专用Surface MSE Loss ----------
class SurfaceMSELoss(torch.nn.Module):
    """
    专门为训练优化的Surface MSE Loss
    预先加载模型点云，避免重复加载
    """

    def __init__(self, model_points: torch.Tensor, reduction: str = "mean"):
        super().__init__()
        # 将模型点云注册为buffer，自动处理device转换
        self.register_buffer('model_points', model_points)
        self.reduction = reduction

    def forward(self, pred_transforms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_transforms: (B, 4, 4) 预测的变换矩阵
        Returns:
            loss: scalar tensor
        """
        batch_size = pred_transforms.shape[0]

        # 扩展模型点云到batch维度
        batched_points = self.model_points.unsqueeze(0).expand(batch_size, -1, -1)

        # 应用变换
        transformed_points = apply_transform_fast(batched_points, pred_transforms)

        # 计算MSE (与原始点云比较)
        target_points = self.model_points.unsqueeze(0).expand_as(transformed_points)
        return compute_surface_mse_fast(transformed_points, target_points, self.reduction)