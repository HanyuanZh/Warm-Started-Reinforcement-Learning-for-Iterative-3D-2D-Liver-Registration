import torch
import torch.nn as nn
from kornia.contrib import distance_transform
import torch
import torch.nn as nn
import torch

def dice_loss(pred, target, smooth=1):
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice
class MutualInformationLoss(nn.Module):
    def __init__(self, nbins: int = 64, sigma: float = None, eps: float = 1e-10):
        """
        可微分互信息损失，用于训练神经网络。

        Args:
            nbins (int): 量化的 bin 数量，默认 64。
            sigma (float): 高斯核的标准差。如果为 None，则自动设为 1/(nbins-1)。
            eps (float): 防止除零的数值稳定项。
        """
        super().__init__()
        self.nbins = nbins
        self.eps = eps
        # 如果 sigma 未指定，则设为一个 bin 间距的一半
        self.sigma = sigma if sigma is not None else (1.0 / (nbins - 1))

        # 预先生成 bin 的中心 (float Tensor，形状 (1, 1, nbins))
        centers = torch.linspace(0., 1., nbins)
        # 在 forward 时自动把它放到相同 device
        self.register_buffer('centers', centers.view(1, 1, nbins))  # 形状 (1,1,nbins)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred (Tensor): 网络输出的深度图，形状可以是 (B, 1, H, W) 或 (B, H, W)。
                           数值范围不限，内部会归一化到 [0,1]。
            target (Tensor): 真实深度图，形状可以是 (B, 1, H, W) 或 (B, H, W)。
                             数值范围不限，内部会归一化到 [0,1]。

        Returns:
            loss (Tensor): 标量，等于 -E_b[ I(pred_b; target_b) ]。
        """

        # -----------------------
        # 1. 如果输入是 (B, H, W)，则插入一个 channel 维度，变成 (B, 1, H, W)
        # -----------------------
        #  pred.dim() 可能是 3 或 4：
        #   - 如果是 4，假设为 (B, 1, H, W)
        #   - 如果是 3，假设为 (B, H, W)，需要在第 1 维插入一个长度为 1 的通道维
        if pred.dim() == 3:
            # (B, H, W) -> (B, 1, H, W)
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # 此时 pred, target 都应该是 (B, 1, H, W)
        B, C, H, W = pred.shape
        assert C == 1 and target.shape == pred.shape, \
            "传入的 pred 和 target 必须是形状 (B,1,H,W) 或 (B, H, W)"

        # -----------------------
        # 2. 归一化到 [0,1]（对每个样本在 batch 维度单独做 min-max 归一）
        # -----------------------
        def normalize_map(x: torch.Tensor) -> torch.Tensor:
            # x: (B,1,H,W)
            x_flat = x.view(B, -1)  # (B, H*W)
            x_min = x_flat.min(dim=1, keepdim=True)[0]  # (B,1)
            x_max = x_flat.max(dim=1, keepdim=True)[0]  # (B,1)
            # 扩展回 (B,1,H,W)
            x_min = x_min.view(B, 1, 1, 1)
            x_max = x_max.view(B, 1, 1, 1)
            return (x - x_min) / (x_max - x_min + self.eps)

        pred_norm = normalize_map(pred)
        target_norm = normalize_map(target)

        # -----------------------
        # 3. 计算 soft‐histogram 权重 w_pred、w_target
        # -----------------------
        Npix = H * W
        # 将 (B,1,H,W) 展平为 (B, Npix, 1)
        pred_flat = pred_norm.view(B, -1).unsqueeze(2)   # (B, Npix, 1)
        target_flat = target_norm.view(B, -1).unsqueeze(2)  # (B, Npix, 1)

        # centers 是 (1,1,nbins)，广播后可与 pred_flat/target_flat 相减
        centers = self.centers.to(pred.device)  # (1,1,nbins)

        # w_pred: (B, Npix, nbins)，按高斯核做“软赋值”
        diff_pred = pred_flat - centers            # (B, Npix, nbins)
        w_pred = torch.exp(- (diff_pred ** 2) / (2 * (self.sigma ** 2)))  # (B, Npix, nbins)

        # w_tgt: (B, Npix, nbins)
        diff_tgt = target_flat - centers           # (B, Npix, nbins)
        w_tgt = torch.exp(- (diff_tgt ** 2) / (2 * (self.sigma ** 2)))   # (B, Npix, nbins)

        # -----------------------
        # 4. 先算各自的边缘“直方图”（未归一），形状 (B, nbins)
        # -----------------------
        hist_pred = w_pred.sum(dim=1)  # (B, nbins)
        hist_tgt = w_tgt.sum(dim=1)    # (B, nbins)

        # -----------------------
        # 5. 归一化为概率分布 p_x, p_y
        # -----------------------
        p_pred = hist_pred / (hist_pred.sum(dim=1, keepdim=True) + self.eps)  # (B, nbins)
        p_tgt  = hist_tgt  / (hist_tgt.sum(dim=1, keepdim=True) + self.eps)   # (B, nbins)

        # -----------------------
        # 6. 计算联合概率 p_xy[b, i, j] = (1/Npix) * sum_u w_pred[b,u,i] * w_tgt[b,u,j]
        # -----------------------
        # w_pred.permute(0, 2, 1): (B, nbins, Npix)
        # w_tgt: (B, Npix, nbins)
        w_pred_T = w_pred.permute(0, 2, 1)
        p_xy = torch.bmm(w_pred_T, w_tgt) / float(Npix)  # (B, nbins, nbins)

        # -----------------------
        # 7. 重新计算 p_x 和 p_y（为了与联合分布对应，但数值上与上面 p_pred, p_tgt 等价）
        # -----------------------
        p_x = p_xy.sum(dim=2)  # (B, nbins)
        p_y = p_xy.sum(dim=1)  # (B, nbins)

        # -----------------------
        # 8. 互信息 I = sum_{i,j} p_xy[i,j] * log( p_xy[i,j] / (p_x[i] * p_y[j]) )
        # -----------------------
        px_py = p_x.unsqueeze(2) * p_y.unsqueeze(1)      # (B, nbins, nbins)
        ratio = (p_xy + self.eps) / (px_py + self.eps)   # 防止分母为 0
        mi_map = p_xy * torch.log(ratio)                # (B, nbins, nbins)
        mi = mi_map.sum(dim=(1, 2))                      # (B,)

        # -----------------------
        # 9. 损失 = - E_b[ I ]，对 batch 求平均
        # -----------------------
        loss = - mi.mean()
        return loss
from kornia.contrib import distance_transform
class SoftHausdorffLoss(nn.Module):
    def __init__(self, h: float = 3.0, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum' or 'none'")
        self.h, self.eps, self.reduction = h, eps, reduction

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(self.eps, 1. - self.eps)
        B, C, H, W = x.shape
        return x.view(B * C, 1, H, W)

    def _dt(self, fg: torch.Tensor) -> torch.Tensor:
        # 前景概率 (fg∈(0,1]) → 距离

        return distance_transform(fg, h=self.h)  # 已是可微距离

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError("pred 和 target 必须同 shape")

        pred_flat   = self._prepare(pred)
        target_flat = self._prepare(target.detach())

        dt_pred   = self._dt(pred_flat)
        dt_target = self._dt(target_flat)

        t1 = (pred_flat   * dt_target).flatten(1).mean(1)   # (B*C,)
        t2 = (target_flat * dt_pred  ).flatten(1).mean(1)   # (B*C,)
        loss = t1 + t2                                      # (B*C,)

        if self.reduction == "none":
            B, C = pred.shape[:2]
            return loss.view(B, C)
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss.mean()


class SoftHausdorffLoss(nn.Module):
    def __init__(self, h: float = 3.0, eps: float = 1e-6, reduction: str = "sum",
                 empty_penalty: float = 1.0, symmetric: bool = True):
        """
        改进的 Soft Hausdorff Loss

        Args:
            h: 距离变换的参数
            eps: 防止数值不稳定的小值
            reduction: 'mean', 'sum' 或 'none'
            empty_penalty: 当pred为空（全黑）时的惩罚系数
            symmetric: 是否使用对称的Hausdorff距离
        """
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum' or 'none'")
        self.h = h
        self.eps = eps
        self.reduction = reduction
        self.empty_penalty = empty_penalty
        self.symmetric = symmetric

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        # 把 0/1 mask 转为概率并 clamp
        x = x.float().clamp(self.eps, 1. - self.eps)
        B, C, H, W = x.shape
        return x.view(B * C, 1, H, W)

    def _dt(self, fg: torch.Tensor) -> torch.Tensor:
        # 前景概率 → 可微距离
        return distance_transform(fg, h=self.h)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C, H, W = pred.shape
        pred_flat = self._prepare(pred)  # (B*C, 1, H, W)
        target_flat = self._prepare(target).detach()  # (B*C, 1, H, W)

        # 二值化 mask 并统计每个通道的前景像素数
        flat_pred = pred_flat.view(B * C, -1)  # (B*C, H*W)
        flat_target = target_flat.view(B * C, -1)  # (B*C, H*W)

        mask_pred = (flat_pred > 0.5).float()  # (B*C, H*W)
        mask_target = (flat_target > 0.5).float()  # (B*C, H*W)

        N_pred = mask_pred.sum(dim=1)  # (B*C,)
        N_target = mask_target.sum(dim=1).clamp_min(1.0)  # (B*C,)

        # 计算距离变换
        dt_target = self._dt(target_flat)  # (B*C, 1, H, W)

        # 计算 pred → target 的距离
        loss_p2t = (pred_flat * dt_target).view(B * C, -1).mean(1)  # (B*C,)

        # 处理 pred 为空的情况
        is_pred_empty = (N_pred < 1.0).float()  # (B*C,)

        # 如果 pred 为空，使用 target 的平均距离作为惩罚
        # 这确保了当 pred 全黑时，损失会很大
        empty_loss = dt_target.view(B * C, -1).mean(1) * self.empty_penalty

        # 结合正常损失和空集惩罚
        loss_p2t = (1 - is_pred_empty) * loss_p2t + is_pred_empty * empty_loss

        # 如果使用对称距离，还要计算 target → pred 的距离
        if self.symmetric:
            dt_pred = self._dt(pred_flat)  # (B*C, 1, H, W)
            loss_t2p = (target_flat * dt_pred).view(B * C, -1).mean(1)  # (B*C,)

            # 对称损失取最大值（更严格）或平均值（更平滑）
            loss_pc = (loss_p2t + loss_t2p) / 2.0  # 平均
            # loss_pc = torch.max(loss_p2t, loss_t2p)       # 最大值
        else:
            loss_pc = loss_p2t

        # 计算权重 w_i = N_i / sum_j N_j
        total_N = N_target.sum()  # scalar
        weights = N_target / total_N  # (B*C,)

        # 加权损失
        weighted = loss_pc * weights  # (B*C,)

        # reduction
        if self.reduction == "none":
            return weighted.view(B, C)
        else:
            return weighted.sum()


class SoftHausdorffIoULoss(nn.Module):
    def __init__(self, h: float = 3.0, eps: float = 1e-6, reduction: str = "sum",
                 alpha: float = 0.5, beta: float = 0.5):
        """
        结合 Hausdorff 距离和 IoU 的损失函数

        Args:
            h: 距离变换的参数
            eps: 防止数值不稳定的小值
            reduction: 'mean', 'sum' 或 'none'
            alpha: Hausdorff 损失的权重
            beta: IoU 损失的权重
        """
        super().__init__()
        self.hausdorff = SoftHausdorffLoss(h=h, eps=eps, reduction=reduction, symmetric=True)
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def iou_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算 1 - IoU 作为损失"""
        pred_flat = pred.flatten(2)  # (B, C, H*W)
        target_flat = target.flatten(2)  # (B, C, H*W)

        intersection = (pred_flat * target_flat).sum(dim=2)  # (B, C)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2) - intersection

        iou = (intersection + self.eps) / (union + self.eps)  # (B, C)
        return 1.0 - iou  # IoU loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Hausdorff 损失
        h_loss = self.hausdorff(pred, target)

        # IoU 损失
        iou_loss = self.iou_loss(pred, target).mean()

        # 组合损失
        return self.alpha * h_loss + self.beta * iou_loss