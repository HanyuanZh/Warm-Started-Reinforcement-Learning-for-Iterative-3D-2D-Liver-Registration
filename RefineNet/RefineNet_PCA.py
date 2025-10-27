import math
from typing import Optional, Dict, Tuple
import torch.nn as nn
import torch
torch.random.manual_seed(0)
from RefineNet.data_augmentation_fast import augment_contour_map_fast

import os
import glob
import re
import random

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import cv2

class PairedImageMatrixDataset(Dataset):
    def __init__(
        self,
        zero_mask_dir: str,
        zero_contour_dir: str,
        zero_depth_dir: str,
        def_mask_dir: str,
        def_contour_dir: str,
        def_depth_dir: str,
        matrix_path: str,
        def_deformed_dir: str = None,
        num_pairs: int = 1,
        transform: callable = None
    ):
        # 1. 读取零变形目录
        zero_masks   = sorted(glob.glob(os.path.join(zero_mask_dir,   "*.png")))
        zero_ctrs    = sorted(glob.glob(os.path.join(zero_contour_dir, "*.png")))
        zero_depths  = sorted(glob.glob(os.path.join(zero_depth_dir,   "*.png")))
        assert len(zero_masks) == len(zero_ctrs) == len(zero_depths), \
            f"Zero counts mismatch: {len(zero_masks)}, {len(zero_ctrs)}, {len(zero_depths)}"

        # 2. 读取有变形目录
        def_masks   = sorted(glob.glob(os.path.join(def_mask_dir,    "*.png")))
        def_ctrs    = sorted(glob.glob(os.path.join(def_contour_dir, "*.png")))
        def_depths  = sorted(glob.glob(os.path.join(def_depth_dir,    "*.png")))
        assert len(def_masks) == len(def_ctrs) == len(def_depths), \
            f"Deformed counts mismatch: {len(def_masks)}, {len(def_ctrs)}, {len(def_depths)}"

        # 3. 拼接总体列表
        self.mask_list    = zero_masks  + def_masks
        self.contour_list = zero_ctrs   + def_ctrs
        self.depth_list   = zero_depths + def_depths

        # 4. 构造 deformation 列表：零部分 None，非零解析文件名
        if def_deformed_dir:
            def_deforms = sorted(glob.glob(os.path.join(def_deformed_dir, "*.png")))
            assert len(def_deforms) == len(def_masks), \
                f"Deformation files mismatch: {len(def_deforms)} vs {len(def_masks)}"
            self.deformed_list = [None] * len(zero_masks) + def_deforms
        else:
            self.deformed_list = [None] * len(self.mask_list)

        # 5. 加载矩阵及正则
        self.matrices    = np.load(matrix_path).astype(np.float32)
        self.id_pattern  = re.compile(r"(\d+)")
        self.num_pairs   = num_pairs
        self.transform   = transform

        # 6. 划分零 vs 非零索引
        zero_idxs, nonzero_idxs = [], []
        for i, fp in enumerate(self.deformed_list):
            vec = self._parse_deformation(fp)
            if vec.abs().sum() == 0:
                zero_idxs.append(i)
            else:
                nonzero_idxs.append(i)
        self.zero_idxs    = zero_idxs
        self.nonzero_idxs = nonzero_idxs

        # 7. 图像预处理 Transforms
        self.transform_color = T.Compose([
            T.Resize((128, 128), interpolation=Image.NEAREST),
            T.ToTensor(),
        ])
        self.transform_depth = T.Compose([
            T.Resize((128, 128), interpolation=Image.NEAREST),
            T.ToTensor(),
        ])
        self.transform_mask = T.Compose([
            T.Resize((128, 128), interpolation=Image.NEAREST),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.zero_idxs) * self.num_pairs

    def _get_id(self, filepath: str) -> int:
        m = self.id_pattern.search(os.path.basename(filepath))
        if not m:
            raise ValueError(f"Cannot parse index from filename: {filepath}")
        return int(m.group(1))

    def _parse_deformation(self, deformed_fp: str):
        if not deformed_fp:
            return torch.zeros(10)
        base = os.path.splitext(os.path.basename(deformed_fp))[0]
        parts = base.split('_')
        if len(parts) != 11:
            return torch.zeros(10)
        try:
            vals = list(map(float, parts[1:]))
            return torch.tensor(vals, dtype=torch.float32)
        except:
            return torch.zeros(10)

    def _extract_color_masks(self, contour_img):
        arr = np.array(contour_img)
        channels = []
        colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,255)]
        for color in colors:
            mask = ((arr[:,:,0]==color[0]) & (arr[:,:,1]==color[1]) & (arr[:,:,2]==color[2])).astype(np.uint8) * 255
            channels.append(Image.fromarray(mask, mode='L'))
        return channels

    def _load_sample(self, idx):
        mask    = Image.open(self.mask_list[idx]).convert('L')
        contour = augment_contour_map_fast(cv2.cvtColor(cv2.imread(self.contour_list[idx], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
        depth   = Image.open(self.depth_list[idx]).convert('L')

        # 拆色彩通道并合并所有通道
        color_pils    = self._extract_color_masks(contour)
        color_tensors = [self.transform_color(p) for p in color_pils]
        combined = torch.cat(color_tensors + [
            self.transform_depth(depth),
            self.transform_mask(mask)
        ], dim=0)
        if self.transform:
            combined = self.transform(combined)

        mat    = torch.from_numpy(self.matrices[self._get_id(self.mask_list[idx])])
        deform = self._parse_deformation(self.deformed_list[idx])
        return {"combined": combined, "matrix": mat, "deformation": deform}

    def __getitem__(self, _):
        first_idx = random.choice(self.zero_idxs)
        if random.random() < 0.5 and self.zero_idxs:
            second_idx = random.choice(self.zero_idxs)
        elif self.nonzero_idxs:
            second_idx = random.choice(self.nonzero_idxs)
        else:
            second_idx = random.choice(self.zero_idxs)

        sample1 = self._load_sample(first_idx)
        sample2 = self._load_sample(second_idx)
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
        self.PCA_head = nn.Sequential(
            nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=512, batch_first=True),
            nn.Linear(embed_dim, 10),
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
            "PCA": self.PCA_head(ab).mean(dim=1),
        }