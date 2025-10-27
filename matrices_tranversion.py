import numpy as np
def vtk_2_surgvtk(mat):
    temp = mat.copy()
    temp = np.linalg.inv(temp)
    temp[:3,:3] = temp[:3, :3] @ np.diag([-1, -1, -1])
    return temp

def vtk_2_PyTorch3D(mat):
    temp =mat.copy()
    temp[:3, 3] = temp[:3, 3] * np.array([-1, 1, -1], dtype=np.float32)
    temp[:3,:3] = temp[:3, :3].T
    temp[:3, :3] = temp[:3, :3] @ np.diag(np.array([-1, 1, -1], dtype=np.float32))
    return temp

def surgvtk_2_vtk(temp):
    D = np.diag([1, -1, -1])
    A = temp.copy()
    A[:3, :3] = A[:3, :3] @ D  # 等效于 np.linalg.inv(D)
    vtk_extrinsic = np.linalg.inv(A)
    return vtk_extrinsic

def PyTorch3D_2_vtk(mat):
    temp = mat.copy()
    temp[:3, :3] = temp[:3, :3] @ np.diag([-1, 1, -1])
    temp[:3, :3] = temp[:3, :3].T
    temp[:3, 3] = temp[:3, 3] * np.array([-1, 1, -1], dtype=np.float32)
    return temp

def PyTorch3d_2_surgvtk(mat):
    temp = mat.copy()
    a= PyTorch3D_2_vtk(temp)
    return vtk_2_surgvtk(a)