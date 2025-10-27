import pickle
import torch


def compute_mse_torch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute mean squared error between two point clouds.
    A, B: tensors of shape (..., N, 3)
    returns: tensor of shape (...) with MSE per sample
    """
    diff = A - B  # shape (..., N, 3)
    mse = torch.mean(torch.sum(diff ** 2, dim=-1), dim=-1)
    return mse


def reconstruct_batch(weights, pca_dict, device=None):
    """
    Batch PCA-based point cloud reconstruction.
    weights: Tensor 或可转换为 Tensor，shape (B, K) 或 (K,)
    pca_dict: 包含 'ref_shape', 'all_pcs', 'all_stds'（numpy array）
    device: torch.device，可选
    返回: recon_pts, Tensor of shape (B, N, 3)
    """
    # 1) 确定 device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) 从 numpy 加载并立即移动到 device
    mean_vec = torch.from_numpy(pca_dict["ref_shape"].reshape(1, -1)).to(device)
    components = torch.from_numpy(pca_dict["all_pcs"]).to(device)
    all_stds = torch.from_numpy(pca_dict["all_stds"]).to(device)

    # 3) 准备 weights - 确保在正确的设备上
    if isinstance(weights, torch.Tensor):
        w = weights.to(device)  # 强制移动到指定设备
    else:
        w = torch.tensor(weights, dtype=all_stds.dtype, device=device)

    if w.dim() == 1:
        w = w.unsqueeze(0)  # (1, K)
    B, K = w.shape
    K_total = components.shape[0]
    assert K <= K_total, f"最多只能有 {K_total} 个权重, 但收到 {K} 个。"

    # 4) 选取前 K 个主成分和标准差
    pcs_k = components[:K, :]  # (K, N*3)
    sigmas_k = all_stds[:K]  # (K,)

    # 5) 关键：此时 w 和 sigmas_k 均在同一 device
    w_sigmas = w * sigmas_k.unsqueeze(0)  # (B, K)
    delta_flat = w_sigmas @ pcs_k  # (B, N*3)

    # 6) 加上均值并 reshape
    recon_flat = delta_flat + mean_vec  # 广播得到 (B, N*3)
    N = recon_flat.shape[-1] // 3
    recon_pts = recon_flat.view(B, N, 3)  # (B, N, 3)

    return recon_pts


def mse_between_weights_torch(w1, w2, pca_dict, device=None):
    """
    计算两组权重重建的点云之间的 MSE，返回 shape (B,) 的 Tensor。
    w1, w2: Tensor 或 list, shape (B, K) 或 (K,)
    """
    # 确保传入同样的 device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pts1 = reconstruct_batch(w1, pca_dict, device)
    pts2 = reconstruct_batch(w2, pca_dict, device)
    # 计算每个样本的 MSE
    diff = pts1 - pts2               # (B, N, 3)
    mse  = torch.mean(torch.sum(diff**2, dim=-1), dim=-1)  # (B,)
    return mse

if __name__ == '__main__':
    # 1. Load PCA dictionary
    with open(r"C:\Users\hanyu\Desktop\pca_cache_merged4.pkl", 'rb') as f:
        pca_dict = pickle.load(f)

    # 2. Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 3. Example batched weights
    weights_a = [[1, 1, 2, 2], [0.5, -1, 0, 0]]  # shape (2, 4)
    weights_b = [[0, 0.5, -1, 2], [1, 0, 1, -1]]

    w1 = torch.tensor(weights_a, dtype=torch.float32, device=device)
    w2 = torch.tensor(weights_b, dtype=torch.float32, device=device)

    # 4. Calculate MSE for each batch
    mse_ab = mse_between_weights_torch(w1, w2, pca_dict, device)
    print(f"Batch MSE between weights_a and weights_b: {mse_ab.cpu().numpy()}")
