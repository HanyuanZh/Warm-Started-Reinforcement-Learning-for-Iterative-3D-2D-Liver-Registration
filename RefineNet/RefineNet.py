import math
from typing import Optional, Dict, Tuple
import torch.nn as nn
import os
import glob
import re
import cv2
import random
import numpy as np
from PIL import Image
import torch
torch.random.manual_seed(0)
from torch.utils.data import Dataset
import torchvision.transforms as T
# from RefineNet.data_augmentation import augment_contour_map
from RefineNet.data_augmentation_fast import augment_contour_map_fast
from RefineNet.depth_augmentation import EnhancedDepthAugmenter

def augment_mask(mask_img, probability=0.5, kernel_size_range=(2, 4)):
    """
    对二值mask进行随机erosion或dilation增强
    
    Args:
        mask_img: PIL Image对象，二值mask
        probability: 执行增强的概率
        kernel_size_range: 形态学操作核大小范围 (min, max)
    
    Returns:
        增强后的PIL Image对象
    """
    if random.random() > probability:
        return mask_img
    
    # 转换为numpy数组
    mask_array = np.array(mask_img)
    
    # 随机选择操作类型和核大小
    operation = random.choice(['erosion', 'dilation'])
    kernel_size = random.randint(*kernel_size_range)
    
    # 确保核大小为奇数
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # 创建形态学核
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # 执行形态学操作
    if operation == 'erosion':
        result = cv2.erode(mask_array, kernel, iterations=1)
    else:  # dilation
        result = cv2.dilate(mask_array, kernel, iterations=1)
    
    return Image.fromarray(result, mode='L')

class PairedImageMatrixDataset(Dataset):
    def __init__(self,
                 mask_dir: str,
                 contour_dir: str,
                 depth_dir: str = None,
                 matrix_path: str = None,
                 num_pairs: int = 1,
                 transform: callable = None,
                 # 新增深度增强参数
                 apply_depth_augmentation: bool = True,
                 depth_aug_config: dict = None,
                 mask_augment_prob: float = 0.3):
        """
        PyTorch Dataset that returns pairs of images with corresponding 4×4 transformation matrices.

        Args:
            mask_dir (str): Directory containing binary mask images (.png).
            contour_dir (str): Directory containing RGB contour images (.png) with pure red/green/blue/white.
            depth_dir (str, optional): Directory containing depth images (.png). If None, depth will not be used.
            matrix_path (str): Path to a .npy file of shape (N, 4, 4) containing transformation matrices.
            num_pairs (int): Number of paired samples to generate per base image (e.g., 2 or 3).
            transform (callable, optional): A function/transform to apply to the combined tensor output.
            apply_depth_augmentation (bool): Whether to apply depth augmentation.
            depth_aug_config (dict): Configuration for depth augmentation parameters.
        """
        # Gather and validate file lists
        self.mask_list = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        self.contour_list = sorted(glob.glob(os.path.join(contour_dir, "*.png")))
        
        # 检查是否使用depth
        self.use_depth = depth_dir is not None
        if self.use_depth:
            self.depth_list = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
            assert len(self.mask_list) == len(self.contour_list) == len(self.depth_list), \
                f"File count mismatch: masks({len(self.mask_list)}), contours({len(self.contour_list)}), depths({len(self.depth_list)})"
        else:
            self.depth_list = None
            assert len(self.mask_list) == len(self.contour_list), \
                f"File count mismatch: masks({len(self.mask_list)}), contours({len(self.contour_list)})"

        # Load transformation matrices
        self.matrices = np.load(matrix_path).astype(np.float32)  # Expect shape (N, 4, 4)
        self.id_pattern = re.compile(r"(\d+)")
        self.num_samples = len(self.mask_list)
        self.num_pairs = num_pairs
        self.transform = transform
        self.mask_augment_prob = mask_augment_prob
        # 设置深度增强
        self.apply_depth_augmentation = apply_depth_augmentation and self.use_depth
        if self.apply_depth_augmentation:
            # 默认深度增强配置
            default_depth_aug_config = {
                'occluder_probability': 0.4,
                'random_erasing_probability': 0.3,
                'depth_normalization_probability': 0.5,
                'scale_perturbation_probability': 0.4,
                'length_range': (20, 80),  # 适应128x128的图像尺寸
                'width_range': (4, 12),
                'angle_range': (-45, 45),
                'norm_min_range': (0.1, 0.3),
                'norm_max_range': (0.7, 0.9),
                'scale_factor_range': (0.8, 1.2),
                'depth_shift_range': (-15, 15),
                'noise_std_range': (0.01, 0.03)
            }
            
            # 合并用户配置
            if depth_aug_config is not None:
                default_depth_aug_config.update(depth_aug_config)
            
            # 创建深度增强器
            self.depth_augmenter = EnhancedDepthAugmenter(**default_depth_aug_config)

        # Define transforms for each modality (nearest-neighbor to preserve binary/discrete values)
        self.transform_color = T.Compose([
            T.Resize((128, 128), interpolation=Image.NEAREST),
            T.ToTensor(),
        ])
        self.transform_depth = T.Compose([
            T.Resize((128, 128), interpolation=Image.BILINEAR),
            T.ToTensor(),
        ])
        self.transform_mask = T.Compose([
            T.Resize((128, 128), interpolation=Image.NEAREST),
            T.ToTensor(),
        ])

    def __len__(self):
        # Total number of pairs
        return self.num_samples * self.num_pairs

    def _get_id(self, filepath: str) -> int:
        match = self.id_pattern.search(os.path.basename(filepath))
        if not match:
            raise ValueError(f"Cannot parse index from filename: {filepath}")
        return int(match.group(1))

    def _extract_color_masks(self, contour_img: Image.Image):
        # Convert RGB contour image into four binary masks (red, green, blue, white)
        arr = np.array(contour_img)  # (H, W, 3)
        channels = []
        colors = [
            (255, 0,   0),   # red
            (0,   255, 0),   # green
            (0,   0,   255), # blue
            (255, 255, 255), # white
        ]
        for (r, g, b) in colors:
            mask_bool = (arr[:, :, 0] == r) & (arr[:, :, 1] == g) & (arr[:, :, 2] == b)
            mask_u8 = (mask_bool.astype(np.uint8)) * 255
            channels.append(Image.fromarray(mask_u8, mode='L'))
        return channels  # [red, green, blue, white]

    def _load_one(self, mask_fp: str, contour_fp: str, depth_fp: str = None):
        # Extract index and load each modality
        idx = self._get_id(mask_fp)
        mask_img = Image.open(mask_fp).convert('L')
        mask_img = augment_mask(mask_img, probability=self.mask_augment_prob)
        # 保留原始的contour增强处理
        # contour_img = augment_contour_map_fast(cv2.imread(contour_fp, cv2.IMREAD_COLOR_RGB))
        # contour_img = augment_contour_map_fast(contour_img)
        contour_img = augment_contour_map_fast(cv2.cvtColor(cv2.imread(contour_fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
        
        # Extract and transform color masks
        color_pils = self._extract_color_masks(contour_img)
        color_tensors = [self.transform_color(pil) for pil in color_pils]
        color_stack = torch.cat(color_tensors, dim=0)  # (4, 128, 128)

        # Transform mask
        mask_t = self.transform_mask(mask_img)     # (1, 128, 128)

        # 根据是否使用depth来组合通道
        if self.use_depth and depth_fp is not None:
            depth_img = Image.open(depth_fp).convert('L')
            
            # 应用深度增强
            if self.apply_depth_augmentation:
                # 转换为numpy数组进行增强
                depth_array = np.array(depth_img)
                # 应用深度增强
                depth_array = self.depth_augmenter(depth_array)
                # 转换回PIL图像
                depth_img = Image.fromarray(depth_array, mode='L')
            
            depth_t = self.transform_depth(depth_img)  # (1, 128, 128)
            # Combine all channels: 4 color + 1 depth + 1 mask = 6
            combined = torch.cat([color_stack, depth_t, mask_t], dim=0)  # (6, 128, 128)
        else:
            # Combine channels without depth: 4 color + 1 mask = 5
            combined = torch.cat([color_stack, mask_t], dim=0)  # (5, 128, 128)

        # Apply optional external transform
        if self.transform is not None:
            combined = self.transform(combined)

        matrix = torch.from_numpy(self.matrices[idx])  # (4, 4)

        return {"combined": combined, "matrix": matrix}

    def __getitem__(self, index: int):
        # Determine base index and pairing offset
        base_idx = index // self.num_pairs
        offset = index % self.num_pairs
        # Partner index is the next images (with wrap-around)
        partner_idx = (base_idx + offset + 1) % self.num_samples

        # Load both samples
        if self.use_depth:
            sample1 = self._load_one(
                self.mask_list[base_idx],
                self.contour_list[base_idx],
                self.depth_list[base_idx]
            )
            sample2 = self._load_one(
                self.mask_list[partner_idx],
                self.contour_list[partner_idx],
                self.depth_list[partner_idx]
            )
        else:
            sample1 = self._load_one(
                self.mask_list[base_idx],
                self.contour_list[base_idx],
                None
            )
            sample2 = self._load_one(
                self.mask_list[partner_idx],
                self.contour_list[partner_idx],
                None
            )

        return sample1, sample2
    
# =============================================================================
# RefineNet model (NVIDIA 2023)
# =============================================================================

# ---- convolution helpers ----

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

# ---- basic blocks ----

class ConvBNReLU(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, groups=1, bias=True,
                 dilation=1, norm_layer: Optional[nn.Module] = nn.BatchNorm2d):
        super().__init__()
        padding = (kernel_size - 1) // 2
        layers = [
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=groups, bias=bias, dilation=dilation)
        ]
        if norm_layer is not None:
            layers.append(norm_layer(C_out))
        layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResnetBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=nn.BatchNorm2d, bias=False):
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1, base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported")
        self.conv1 = conv3x3(inplanes, planes, stride, bias=bias)
        self.bn1 = norm_layer(planes) if norm_layer else None
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, bias=bias)
        self.bn2 = norm_layer(planes) if norm_layer else None
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        if self.bn1:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.bn2:
            out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# ---- positional embedding ----

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # 1×max_len×d_model

    def forward(self, x):
        # x: B×N×D
        return x + self.pe[:, :x.size(1)]

# ---- RefineNet backbone ----

class RefineNet(nn.Module):
    def __init__(self, cfg: Optional[dict] = None, c_in: int = 6):
        super().__init__()
        self.cfg = cfg or {}
        use_bn = self.cfg.get("use_BN", False)
        norm_layer = nn.BatchNorm2d if use_bn else None

        # Encoder‑A (process A or B separately)
        self.encodeA = nn.Sequential(
            ConvBNReLU(c_in, 64, kernel_size=7, stride=2, norm_layer=norm_layer),
            ConvBNReLU(64, 128, stride=2, norm_layer=norm_layer),
            ResnetBasicBlock(128, 128, bias=True, norm_layer=norm_layer),
            ResnetBasicBlock(128, 128, bias=True, norm_layer=norm_layer),
        )

        # Joint encoder on concatenated features (a|b)
        self.encodeAB = nn.Sequential(
            ResnetBasicBlock(256, 256, bias=True, norm_layer=norm_layer),
            ResnetBasicBlock(256, 256, bias=True, norm_layer=norm_layer),
            ConvBNReLU(256, 512, stride=2, norm_layer=norm_layer),
            ResnetBasicBlock(512, 512, bias=True, norm_layer=norm_layer),
            ResnetBasicBlock(512, 512, bias=True, norm_layer=norm_layer),
        )

        embed_dim = 512
        num_heads = 4
        self.pos_embed = PositionalEmbedding(embed_dim, max_len=400)
        self.trans_head = nn.Sequential(
            nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=512, batch_first=True),
            nn.Linear(embed_dim, 3),
        )
        rot_dim = 6 if self.cfg.get("rot_rep") == "6d" else 3
        self.rot_head = nn.Sequential(
            nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=512, batch_first=True),
            nn.Linear(embed_dim, rot_dim),
        )

    def forward(self, A: torch.Tensor, B: torch.Tensor):
        bs = A.size(0)
        x = torch.cat([A, B], dim=0)        # 2B×C×H×W
        x = self.encodeA(x)                 # 2B×128×H/4×W/4
        a, b = x[:bs], x[bs:]               # split back
        ab = torch.cat([a, b], dim=1)       # B×256×…
        ab = self.encodeAB(ab)              # B×512×H/8×W/8
        ab = ab.flatten(2).permute(0, 2, 1) # B×N×512
        ab = self.pos_embed(ab)
        return {
            "trans": self.trans_head(ab).mean(dim=1),
            "rot": self.rot_head(ab).mean(dim=1),
        }