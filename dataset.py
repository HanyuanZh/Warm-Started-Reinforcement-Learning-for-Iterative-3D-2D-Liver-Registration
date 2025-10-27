import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Union, Optional
from matrices_tranversion_tensor import vtk_2_PyTorch3D, surgvtk_2_vtk
from matrices_tranversion import PyTorch3d_2_surgvtk
from matrices_tranversion_tensor1 import surgvtk_to_pytorch3d, pytorch3d_to_surgvtk, compose_in_pytorch3d


class PosePairDataset(Dataset):
    """
    Dataset that randomly samples two different IDs from the "available matrix ID set" 
    specified in mat_id.npy, retrieves corresponding 4x4 poses from training_pose.npy,
    and converts them to surgvtk/vtk/p3d formats.
    
    Returns:
      {
        "index_pair": LongTensor(2,),  # IDs of selected matrices (indices in training_pose)
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
        ensure_unique_ids: bool = True,   # True: use unique ID set; False: use original array (may contain duplicates)
    ):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype

        # Load pose library
        poses_np = np.load(pose_path)  # (M,4,4)
        if poses_np.ndim != 3 or poses_np.shape[1:] != (4, 4):
            raise ValueError(f"training_pose.npy shape should be (N,4,4), but got {poses_np.shape}")
        self.poses_np = poses_np
        self.num_poses = len(poses_np)

        # Load allowed ID set (these numbers are indices into training_pose)
        ids = np.asarray(np.load(id_path), dtype=np.int64).ravel()
        if ids.size < 2:
            raise ValueError("mat_id.npy must contain at least 2 indices.")

        # Boundary check
        if np.any(ids < 0) or np.any(ids >= self.num_poses):
            raise IndexError(f"mat_id.npy contains out-of-bounds indices. Valid range: [0, {self.num_poses-1}]")

        # Remove duplicates if requested
        self.id_pool = np.unique(ids) if ensure_unique_ids else ids

        # Conversion functions
        self.surgvtk_2_vtk = surgvtk_2_vtk
        self.vtk_2_PyTorch3D = vtk_2_PyTorch3D
        self.PyTorch3d_2_surgvtk = PyTorch3d_2_surgvtk

        if self.id_pool.size < 2:
            raise ValueError("Valid candidate IDs are less than 2, cannot sample two different IDs without replacement.")

    def __len__(self) -> int:
        # Random sampling dataset length doesn't significantly affect training; 
        # returning candidate set size is more intuitive
        return int(self.id_pool.size)

    def _convert_all(self, surgvtk_mat_np: np.ndarray):
        """Convert surgvtk matrix to all three formats"""
        surgvtk_t = torch.as_tensor(surgvtk_mat_np, dtype=self.dtype, device=self.device)
        vtk_t = self.surgvtk_2_vtk(surgvtk_t)
        p3d_t = self.vtk_2_PyTorch3D(vtk_t)
        return surgvtk_t, vtk_t, p3d_t

    def __getitem__(self, idx: int):
        # Randomly select two different IDs from mat_id.npy candidate set (without replacement)
        if self.id_pool.size >= 2:
            i0, i1 = np.random.choice(self.id_pool, size=2, replace=False)
        else:
            # Should not be triggered due to earlier checks
            raise RuntimeError("Insufficient candidate IDs to sample two different elements.")

        # Get corresponding poses (surgvtk 4x4 in numpy)
        mat0_np = self.poses_np[int(i0)]
        mat1_np = self.poses_np[int(i1)]

        # Convert to all three formats
        surg0, vtk0, p3d0 = self._convert_all(mat0_np)
        surg1, vtk1, p3d1 = self._convert_all(mat1_np)

        sample = {
            "index_pair": torch.tensor([int(i0), int(i1)], device=self.device, dtype=torch.int64),
            "surgvtk": torch.stack([surg0, surg1], dim=0),  # (2,4,4)
            "vtk":     torch.stack([vtk0, vtk1],   dim=0),  # (2,4,4)
            "p3d":     torch.stack([p3d0, p3d1],   dim=0),  # (2,4,4)
        }
        return sample


class FixedPosePairDataset:
    """
    Wrapper around the original dataset that provides fixed pose pairs.
    """
    def __init__(self, original_dataset: PosePairDataset, target_ids: tuple):
        self.original_dataset = original_dataset
        self.target_ids = target_ids  # (start_id, target_id)
        
        # Verify target IDs are in the candidate pool
        if target_ids[0] not in original_dataset.id_pool or target_ids[1] not in original_dataset.id_pool:
            raise ValueError(f"Target IDs {target_ids} are not in the dataset's candidate pool")
        
        # Pre-generate fixed sample
        self.fixed_sample = self._generate_fixed_sample()
        
    def _generate_fixed_sample(self):
        """Generate the fixed sample"""
        i0, i1 = self.target_ids
        
        # Get corresponding matrices directly from poses
        mat0_np = self.original_dataset.poses_np[int(i0)]
        mat1_np = self.original_dataset.poses_np[int(i1)]
        
        # Convert formats
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
        # Always return the same fixed sample regardless of input index
        return self.fixed_sample
    
    def __len__(self):
        return 1  # Only one fixed sample
