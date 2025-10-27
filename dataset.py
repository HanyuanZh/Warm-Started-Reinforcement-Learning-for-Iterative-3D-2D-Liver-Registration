import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Union, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Union, Optional
from matrices_tranversion_tensor import vtk_2_PyTorch3D, surgvtk_2_vtk
from matrices_tranversion import PyTorch3d_2_surgvtk  # 如训练时需要回转
from matrices_tranversion_tensor1 import surgvtk_to_pytorch3d,pytorch3d_to_surgvtk,compose_in_pytorch3d
class PosePairDataset(Dataset):
    """
    从 mat_id.npy 指定的“可用矩阵ID集合”中，每次随机抽取两个不同的 ID，
    在 training_pose.npy 中取对应 4x4 姿态，并转换为 surgvtk/vtk/p3d 三种格式。
    返回:
      {
        "index_pair": LongTensor(2,),  # 选到的两个矩阵的 ID（即 training_pose 的索引）
        "surgvtk":   Tensor(2,4,4),
        "vtk":       Tensor(2,4,4),
        "p3d":       Tensor(2,4,4),
      }
    """
    def __init__(
        self,
        pose_path: str = "./data/Bartoli/training_pose.npy",
        id_path: str = "./data/mat_id.npy",
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.float32,
        ensure_unique_ids: bool = True,   # True: 用唯一ID集合做随机；False: 按原数组（可能重复）
    ):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype

        # 加载姿态库
        poses_np = np.load(pose_path)  # (M,4,4)
        if poses_np.ndim != 3 or poses_np.shape[1:] != (4, 4):
            raise ValueError(f"training_pose.npy 形状应为 (N,4,4)，当前为 {poses_np.shape}")
        self.poses_np = poses_np
        self.num_poses = len(poses_np)

        # 读取允许的ID集合（这些数字就是 training_pose 的索引）
        ids = np.asarray(np.load(id_path), dtype=np.int64).ravel()
        if ids.size < 2:
            raise ValueError("mat_id.npy 至少需要包含 2 个索引。")

        # 边界检查
        if np.any(ids < 0) or np.any(ids >= self.num_poses):
            raise IndexError(f"mat_id.npy 中存在越界索引，允许范围 [0, {self.num_poses-1}]")

        # 是否去重
        self.id_pool = np.unique(ids) if ensure_unique_ids else ids

        # 转换函数

        self.surgvtk_2_vtk = surgvtk_2_vtk
        self.vtk_2_PyTorch3D = vtk_2_PyTorch3D
        self.PyTorch3d_2_surgvtk = PyTorch3d_2_surgvtk

        if self.id_pool.size < 2:
            raise ValueError("有效的候选 ID 少于 2，无法无放回随机抽取两个不同的 ID。")

    def __len__(self) -> int:
        # 随机采样数据集长度对训练影响不大；返回候选集合大小更直观
        return int(self.id_pool.size)

    def _convert_all(self, surgvtk_mat_np: np.ndarray):
        surgvtk_t = torch.as_tensor(surgvtk_mat_np, dtype=self.dtype, device=self.device)
        vtk_t = self.surgvtk_2_vtk(surgvtk_t)
        p3d_t = self.vtk_2_PyTorch3D(vtk_t)
        return surgvtk_t, vtk_t, p3d_t

    def __getitem__(self, idx: int):
        # 在 mat_id.npy 的候选集合里随机选两个不同的 ID（无放回）
        if self.id_pool.size >= 2:
            i0, i1 = np.random.choice(self.id_pool, size=2, replace=False)
        else:
            # 理论上前面已做检查不会触发
            raise RuntimeError("候选 ID 不足以采样两不同元素。")

        # 取对应姿态（surgvtk 4x4 in numpy）
        mat0_np = self.poses_np[int(i0)]
        mat1_np = self.poses_np[int(i1)]

        # 转三种格式
        surg0, vtk0, p3d0 = self._convert_all(mat0_np)
        surg1, vtk1, p3d1 = self._convert_all(mat1_np)

        sample = {
            "index_pair": torch.tensor([int(i0), int(i1)], device=self.device, dtype=torch.int64),
            "surgvtk": torch.stack([surg0, surg1], dim=0),  # (2,4,4)
            "vtk":     torch.stack([vtk0, vtk1],   dim=0),  # (2,4,4)
            "p3d":     torch.stack([p3d0, p3d1],   dim=0),  # (2,4,4)
        }
        return sample

# class PosePairDataset(Dataset):
#     """
#     从 mat_id.npy 指定的“可用矩阵ID集合”中，每次随机抽取两个不同的 ID，
#     在 training_pose.npy 中取对应 4x4 姿态，并转换为 surgvtk/vtk/p3d 三种格式。
#     返回:
#       {
#         "index_pair": LongTensor(2,),  # 选到的两个矩阵的 ID（即 training_pose 的索引）
#         "surgvtk":   Tensor(2,4,4),
#         "vtk":       Tensor(2,4,4),
#         "p3d":       Tensor(2,4,4),
#       }
#     """
#     def __init__(
#         self,
#         pose_path: str = "./data/Bartoli/training_pose.npy",
#         id_path: str = "./data/mat_id.npy",
#         device: Union[str, torch.device] = "cuda",
#         dtype: torch.dtype = torch.float32,
#         ensure_unique_ids: bool = True,   # True: 用唯一ID集合做随机；False: 按原数组（可能重复）
#     ):
#         super().__init__()
#         self.device = torch.device(device)
#         self.dtype = dtype

#         # 加载姿态库
#         poses_np = np.load(pose_path)  # (M,4,4)
#         if poses_np.ndim != 3 or poses_np.shape[1:] != (4, 4):
#             raise ValueError(f"training_pose.npy 形状应为 (N,4,4)，当前为 {poses_np.shape}")
#         self.poses_np = poses_np
#         self.num_poses = len(poses_np)

#         # 读取允许的ID集合（这些数字就是 training_pose 的索引）
#         ids = np.asarray(np.load(id_path), dtype=np.int64).ravel()
#         if ids.size < 2:
#             raise ValueError("mat_id.npy 至少需要包含 2 个索引。")

#         # 边界检查
#         if np.any(ids < 0) or np.any(ids >= self.num_poses):
#             raise IndexError(f"mat_id.npy 中存在越界索引，允许范围 [0, {self.num_poses-1}]")

#         # 是否去重
#         self.id_pool = np.unique(ids) if ensure_unique_ids else ids

#         # 转换函数
#         from matrices_tranversion_tensor import vtk_2_PyTorch3D, surgvtk_2_vtk
#         from matrices_tranversion import PyTorch3d_2_surgvtk  # 如训练时需要回转
#         self.surgvtk_2_vtk = surgvtk_2_vtk
#         self.vtk_2_PyTorch3D = vtk_2_PyTorch3D
#         self.PyTorch3d_2_surgvtk = PyTorch3d_2_surgvtk

#         if self.id_pool.size < 2:
#             raise ValueError("有效的候选 ID 少于 2，无法无放回随机抽取两个不同的 ID。")

#     def __len__(self) -> int:
#         # 随机采样数据集长度对训练影响不大；返回候选集合大小更直观
#         return int(self.id_pool.size)

#     def _convert_all(self, surgvtk_mat_np: np.ndarray):
#         surgvtk_t = torch.as_tensor(surgvtk_mat_np, dtype=self.dtype, device=self.device)
#         vtk_t = self.surgvtk_2_vtk(surgvtk_t)
#         p3d_t = self.vtk_2_PyTorch3D(vtk_t)
#         return surgvtk_t, vtk_t, p3d_t

#     def __getitem__(self, idx: int):
#         # 在 mat_id.npy 的候选集合里随机选两个不同的 ID（无放回）
#         if self.id_pool.size >= 2:
#             i0, i1 = np.random.choice(self.id_pool, size=2, replace=False)
#         else:
#             # 理论上前面已做检查不会触发
#             raise RuntimeError("候选 ID 不足以采样两不同元素。")

#         # 取对应姿态（surgvtk 4x4 in numpy）
#         mat0_np = self.poses_np[int(i0)]
#         mat1_np = self.poses_np[int(i1)]

#         # 转三种格式
#         surg0, vtk0, p3d0 = self._convert_all(mat0_np)
#         surg1, vtk1, p3d1 = self._convert_all(mat1_np)

#         sample = {
#             "index_pair": torch.tensor([int(i0), int(i1)], device=self.device, dtype=torch.int64),
#             "surgvtk": torch.stack([surg0, surg1], dim=0),  # (2,4,4)
#             "vtk":     torch.stack([vtk0, vtk1],   dim=0),  # (2,4,4)
#             "p3d":     torch.stack([p3d0, p3d1],   dim=0),  # (2,4,4)
#         }
#         return sample

class FixedPosePairDataset:
    """
    包装原始数据集，提供固定的pose pair
    """
    def __init__(self, original_dataset: PosePairDataset, target_ids: tuple):
        self.original_dataset = original_dataset
        self.target_ids = target_ids  # (start_id, target_id)
        
        # 验证目标IDs是否在候选池中
        if target_ids[0] not in original_dataset.id_pool or target_ids[1] not in original_dataset.id_pool:
            raise ValueError(f"目标IDs {target_ids} 不在数据集的候选池中")
        
        # 预生成固定样本
        self.fixed_sample = self._generate_fixed_sample()
        
    def _generate_fixed_sample(self):
        """生成固定的样本"""
        i0, i1 = self.target_ids
        
        # 直接从poses中获取对应的矩阵
        mat0_np = self.original_dataset.poses_np[int(i0)]
        mat1_np = self.original_dataset.poses_np[int(i1)]
        
        # 转换格式
        surg0, vtk0, p3d0 = self.original_dataset._convert_all(mat0_np)
        surg1, vtk1, p3d1 = self.original_dataset._convert_all(mat1_np)
        
        sample = {
            "index_pair": torch.tensor([int(i0), int(i1)], 
                                     device=self.original_dataset.device, 
                                     dtype=torch.int64),
            "surgvtk": torch.stack([surg0, surg1], dim=0),
            "vtk": torch.stack([vtk0, vtk1], dim=0),
            "p3d": torch.stack([p3d0, p3d1], dim=0),
        }
        return sample
    
    def __getitem__(self, idx: int):
        # 无论传入什么索引，都返回相同的固定样本
        return self.fixed_sample
    
    def __len__(self):
        return 1  # 只有一个固定样本