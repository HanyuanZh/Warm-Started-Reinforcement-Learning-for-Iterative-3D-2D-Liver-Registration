import torch

# 定义常用的对角矩阵
E = torch.diag(torch.tensor([-1.,  1., -1.], dtype=torch.float32))
D = torch.diag(torch.tensor([ 1., -1., -1.], dtype=torch.float32))
F = torch.diag(torch.tensor([-1., -1.,  1.], dtype=torch.float32))  # 只是推导里会用到

def vtk_2_surgvtk(mat):
    """
    Convert VTK matrix or batch of matrices to SurgVTK format.
    Args:
        mat: torch.Tensor of shape (..., 4, 4)
    Returns:
        torch.Tensor of shape (..., 4, 4)
    """
    temp = mat.clone()
    temp = torch.linalg.inv(temp)
    
    # 构造对角矩阵，确保设备和数据类型一致
    diag_neg = torch.diag(torch.tensor([-1., -1., -1.], device=mat.device, dtype=mat.dtype))
    
    # 批量矩阵乘法
    temp[..., :3, :3] = temp[..., :3, :3] @ diag_neg
    return temp

def vtk_2_PyTorch3D(mat):
    """
    Convert VTK matrix or batch of matrices to PyTorch3D format.
    Args:
        mat: torch.Tensor of shape (..., 4, 4)
    Returns:
        torch.Tensor of shape (..., 4, 4)
    """
    temp = mat.clone()
    
    # 构造缩放向量
    scale_vec = torch.tensor([-1., 1., -1.], device=mat.device, dtype=mat.dtype)
    
    # 缩放平移部分
    temp[..., :3, 3] = temp[..., :3, 3] * scale_vec
    
    # 转置旋转部分
    temp[..., :3, :3] = temp[..., :3, :3].transpose(-2, -1)
    
    # 构造对角缩放矩阵并应用
    diag_scale = torch.diag(scale_vec)
    temp[..., :3, :3] = temp[..., :3, :3] @ diag_scale
    
    return temp

def surgvtk_2_vtk(temp):
    """
    Convert SurgVTK matrix or batch of matrices to VTK format.
    Args:
        temp: torch.Tensor of shape (..., 4, 4)
    Returns:
        torch.Tensor of shape (..., 4, 4)
    """
    A = temp.clone()
    
    # 构造对角矩阵D
    D = torch.diag(torch.tensor([1., -1., -1.], device=temp.device, dtype=temp.dtype))
    
    # 应用D变换
    A[..., :3, :3] = A[..., :3, :3] @ D
    
    # 求逆
    vtk_extrinsic = torch.linalg.inv(A)
    return vtk_extrinsic

# f = vtk_2_PyTorch3D ∘ surgvtk_2_vtk
def surgvtk_to_pytorch3d(X_surg):
    """
    Convert SurgVTK matrix to PyTorch3D format.
    Args:
        X_surg: torch.Tensor of shape (..., 4, 4)
    Returns:
        torch.Tensor of shape (..., 4, 4)
    """
    return vtk_2_PyTorch3D(surgvtk_2_vtk(X_surg))

# 反变换：f^{-1}
def pytorch3d_to_surgvtk(M_py):
    """
    Convert PyTorch3D matrix back to SurgVTK format.
    Args:
        M_py: torch.Tensor of shape (..., 4, 4)
    Returns:
        torch.Tensor of shape (..., 4, 4)
    """
    Y = M_py.clone()
    
    # 构造E矩阵
    E_local = torch.diag(torch.tensor([-1., 1., -1.], device=M_py.device, dtype=M_py.dtype))
    
    # 提取旋转和平移
    R_out = Y[..., :3, :3].clone()
    t_out = Y[..., :3, 3].clone()
    
    # 逆变换旋转部分：R_Y = E @ R_out^T
    R_Y = E_local @ R_out.transpose(-2, -1)
    
    # 逆变换平移部分
    scale_vec = torch.tensor([-1., 1., -1.], device=M_py.device, dtype=M_py.dtype)
    t_Y = t_out * scale_vec
    
    # 更新Y
    Y[..., :3, :3] = R_Y
    Y[..., :3, 3] = t_Y
    
    # 逆变换 surgvtk_2_vtk：Y = inv(A')，且 A'_R = A_R @ D, A'_t = A_t
    A_prime = torch.linalg.inv(Y)
    X_surg = A_prime.clone()
    
    # 构造D矩阵
    D_local = torch.diag(torch.tensor([1., -1., -1.], device=M_py.device, dtype=M_py.dtype))
    
    # A_R = A'_R @ D
    X_surg[..., :3, :3] = A_prime[..., :3, :3] @ D_local
    
    return X_surg

def compose_in_pytorch3d(A_py, B_surg):
    """
    Compose transformation in PyTorch3D space.
    
    Given A_py = f(A) and B (in surgvtk space), compute f(A @ B) without needing original A.
    
    Args:
        A_py: torch.Tensor of shape (..., 4, 4) - Matrix in PyTorch3D format
        B_surg: torch.Tensor of shape (..., 4, 4) or (4, 4) - Transformation in SurgVTK format
    
    Returns:
        torch.Tensor of shape (..., 4, 4) - Result in PyTorch3D format
    """
    # 1) 逆回 surgvtk
    A_surg = pytorch3d_to_surgvtk(A_py)
    
    # 2) 在 surgvtk 里右乘
    # 如果B_surg是单个矩阵而A_surg是批量的，会自动广播
    A_new_surg = A_surg @ B_surg
    
    # 3) 转回 PyTorch3D
    return surgvtk_to_pytorch3d(A_new_surg)

