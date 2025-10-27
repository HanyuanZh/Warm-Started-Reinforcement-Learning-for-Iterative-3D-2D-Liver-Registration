from __future__ import annotations

"""
PPO training for MSE-based SE3PoseEnvDiscrete environment - CURRICULUM START VERSION
- Uses RefineNet backbone with pretrained weights
- Fixed-pair curriculum to get learning off the ground
- Rollout uses model.train() (BN stats consistent; Dropout=0)
- Keep empty-image transitions (handle_empty='undo')
- Reward: curr_first_positive (denser & stable early signal) 
- Entropy anneals high -> 0
- Decision heads for step mode and termination prediction
"""
from augmentation import TargetImageAugmenter
import os
import random
import math
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical

# Image saving related imports
import cv2

# ========= your project modules =========
from dataset import FixedPosePairDataset
from dataset import PosePairDataset
from env_auto import (
    SE3PoseEnvDiscrete, DiscreteStepSizes,
    VTKRenderer, load_vtk_points,
    get_default_surfaces, get_default_camera_params,
)


# ========= Deterministic setup =========
def set_deterministic_training(seed=612):
    """Set up deterministic training for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ========= utils =========
def tensor01_from_obs(obs, device):
    return tuple(o.unsqueeze(0).to(device) for o in obs)


def reset_compat(env, **kwargs):
    out = env.reset(**kwargs) if kwargs else env.reset()
    return out if isinstance(out, tuple) else (out, {})


def save_episode_images(start_obs, end_obs, episode_count, save_dir="episode_images",
                        success=False, mse=0.0, steps=0):
    """
    Save rendered images at episode start and end (use channel 5 for liver mask display)
    """
    os.makedirs(save_dir, exist_ok=True)

    def extract_liver_mask(obs_tuple):
        current_img, target_img = obs_tuple
        if isinstance(current_img, torch.Tensor):
            current_mask = current_img[4].cpu().numpy()
            target_mask = target_img[4].cpu().numpy()
        else:
            current_mask = current_img[4]
            target_mask = target_img[4]
        current_mask = ((current_mask - current_mask.min()) /
                        (current_mask.max() - current_mask.min() + 1e-8) * 255).astype(np.uint8)
        target_mask = ((target_mask - target_mask.min()) /
                       (target_mask.max() - target_mask.min() + 1e-8) * 255).astype(np.uint8)
        return current_mask, target_mask

    try:
        start_current, start_target = extract_liver_mask(start_obs)
        end_current, end_target = extract_liver_mask(end_obs)
        h, w = start_current.shape
        grid_img = np.zeros((h * 2, w * 2), dtype=np.uint8)
        grid_img[0:h, 0:w] = start_current
        grid_img[0:h, w:2 * w] = start_target
        grid_img[h:2 * h, 0:w] = end_current
        grid_img[h:2 * h, w:2 * w] = end_target
        grid_img_color = cv2.cvtColor(grid_img, cv2.COLOR_GRAY2BGR)
        font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        color = (0, 255, 0) if success else (0, 0, 255)
        cv2.putText(grid_img_color, "Start Current", (5, 15), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(grid_img_color, "Start Target", (w + 5, 15), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(grid_img_color, "End Current", (5, h + 15), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(grid_img_color, "End Target", (w + 5, h + 15), font, font_scale, (255, 255, 255), thickness)
        status = "SUCCESS" if success else "FAILED"
        info_text = f"Ep{episode_count}: {status} | MSE: {mse:.2f} | Steps: {steps}"
        cv2.putText(grid_img_color, info_text, (5, h * 2 - 5), font, font_scale, color, thickness)
        filename = f"episode_{episode_count:04d}_{status}_mse{mse:.2f}_steps{steps}.png"
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, grid_img_color)
        if (episode_count % 500 == 0) or success:
            print(f"Saved episode image: {filepath}")
    except Exception as e:
        print(f"Error saving episode {episode_count} image: {e}")


# ========= PPO Buffer =========
@dataclass
class PPOTransition:
    """Single transition for PPO"""
    obs1: np.ndarray
    obs2: np.ndarray
    action: int
    reward: float
    value: float
    log_prob: float
    done: bool
    is_success: bool = False
    gt_step_mode: float = 0.0
    gt_terminate: float = 0.0


class PPOBuffer:
    """Rollout buffer for PPO"""

    def __init__(self):
        self.transitions: List[PPOTransition] = []

    def push(self, transition: PPOTransition):
        self.transitions.append(transition)

    def clear(self):
        self.transitions = []

    def get_batches(self, batch_size: int, device: torch.device):
        """Get randomized batches for PPO training"""
        n = len(self.transitions)
        indices = np.arange(n)
        np.random.shuffle(indices)

        for start_idx in range(0, n, batch_size):
            end_idx = min(start_idx + batch_size, n)
            batch_indices = indices[start_idx:end_idx]

            obs1_batch = torch.stack([
                torch.tensor(self.transitions[i].obs1, dtype=torch.float32)
                for i in batch_indices
            ]).to(device)
            obs2_batch = torch.stack([
                torch.tensor(self.transitions[i].obs2, dtype=torch.float32)
                for i in batch_indices
            ]).to(device)
            actions = torch.tensor(
                [self.transitions[i].action for i in batch_indices],
                dtype=torch.long, device=device
            )
            old_log_probs = torch.tensor(
                [self.transitions[i].log_prob for i in batch_indices],
                dtype=torch.float32, device=device
            )
            old_values = torch.tensor(
                [self.transitions[i].value for i in batch_indices],
                dtype=torch.float32, device=device
            )
            rewards = torch.tensor(
                [self.transitions[i].reward for i in batch_indices],
                dtype=torch.float32, device=device
            )
            dones = torch.tensor(
                [self.transitions[i].done for i in batch_indices],
                dtype=torch.float32, device=device
            )
            succ_flags = torch.tensor(
                [1.0 if self.transitions[i].is_success else 0.0 for i in batch_indices],
                dtype=torch.float32, device=device
            )
            # decision labels
            gt_step_modes = torch.tensor(
                [self.transitions[i].gt_step_mode for i in batch_indices],
                dtype=torch.float32, device=device
            )
            gt_terminates = torch.tensor(
                [self.transitions[i].gt_terminate for i in batch_indices],
                dtype=torch.float32, device=device
            )

            yield {
                'obs': (obs1_batch, obs2_batch),
                'actions': actions,
                'old_log_probs': old_log_probs,
                'old_values': old_values,
                'rewards': rewards,
                'dones': dones,
                'succ_flags': succ_flags,
                'indices': batch_indices,
                'gt_step_modes': gt_step_modes,
                'gt_terminates': gt_terminates,
            }

    def compute_returns_and_advantages(self, gamma: float, lam: float, device: torch.device,
                                       last_value: float = 0.0, last_done: float = 1.0):
        """Compute GAE returns and advantages with proper bootstrap"""
        n = len(self.transitions)
        returns = torch.zeros(n, dtype=torch.float32)
        advantages = torch.zeros(n, dtype=torch.float32)

        rewards = torch.tensor([t.reward for t in self.transitions], dtype=torch.float32)
        values = torch.tensor([t.value for t in self.transitions], dtype=torch.float32)
        dones = torch.tensor([t.done for t in self.transitions], dtype=torch.float32)

        gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value_t = 0.0 if last_done > 0.5 else last_value
            else:
                next_value_t = values[t + 1]
            delta = rewards[t] + gamma * next_value_t * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns.to(device), advantages.to(device)

    def __len__(self):
        return len(self.transitions)


# ========= RefineNet Components =========
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


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


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class RefineNet(nn.Module):
    def __init__(self, cfg: Optional[dict] = None, c_in: int = 6):
        super().__init__()
        self.cfg = cfg or {}
        use_bn = self.cfg.get("use_BN", False)
        norm_layer = nn.BatchNorm2d if use_bn else None

        self.encodeA = nn.Sequential(
            ConvBNReLU(c_in, 64, kernel_size=7, stride=2, norm_layer=norm_layer),
            ConvBNReLU(64, 128, stride=2, norm_layer=norm_layer),
            ResnetBasicBlock(128, 128, bias=True, norm_layer=norm_layer),
            ResnetBasicBlock(128, 128, bias=True, norm_layer=norm_layer),
        )

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
        x = torch.cat([A, B], dim=0)
        x = self.encodeA(x)
        a, b = x[:bs], x[bs:]
        ab = torch.cat([a, b], dim=1)
        ab = self.encodeAB(ab)
        ab = ab.flatten(2).permute(0, 2, 1)
        ab = self.pos_embed(ab)
        return {
            "trans": self.trans_head(ab).mean(dim=1),
            "rot": self.rot_head(ab).mean(dim=1),
        }


# ========= RefineNet Actor-Critic =========
class RefineNetActorCritic(nn.Module):
    """
    RefineNet-based Actor-Critic for pose estimation RL
    支持加载预训练的RefineNet权重，并增加step mode和termination decision heads
    """

    def __init__(self, num_actions: int, input_size: int = 128,
                 dropout_rate: float = 0.0,
                 use_bn: bool = False,
                 rot_rep: str = "6d",
                 pretrained_path: str = None,
                 freeze_encoder: bool = False):
        """
        Args:
            num_actions: 动作空间大小
            input_size: 输入图像尺寸
            dropout_rate: Dropout比率
            use_bn: 是否使用BatchNorm
            rot_rep: 旋转表示 ("6d" or "3d")
            pretrained_path: 预训练RefineNet权重路径
            freeze_encoder: 是否冻结encoder参数
        """
        super().__init__()

        # RefineNet backbone configuration
        refinenet_cfg = {
            "use_BN": use_bn,
            "rot_rep": rot_rep
        }

        # RefineNet encoder
        self.encoder = RefineNet(cfg=refinenet_cfg, c_in=6)

        if pretrained_path is not None:
            self._load_pretrained_weights(pretrained_path, freeze_encoder)

        # Feature projector (512 -> 512)
        self.feature_projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        hidden = 512

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(512, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, num_actions)
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(512, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, 1)
        )

        self.step_mode_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1),
        )

        self.terminate_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1),
        )

        # Initialize new layers
        for module in [self.feature_projector, self.actor, self.critic,
                       self.step_mode_head, self.terminate_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        # Special initialization for output layers
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.orthogonal_(self.step_mode_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.terminate_head[-1].weight, gain=0.01)

        # Sanity check
        with torch.no_grad():
            t1 = torch.zeros(1, 6, input_size, input_size)
            t2 = torch.zeros(1, 6, input_size, input_size)
            _ = self.forward((t1, t2))

        print(f"✓ RefineNetActorCritic initialized (with decision heads)")
        print(f"  - Pretrained: {pretrained_path is not None}")
        print(f"  - Encoder frozen: {freeze_encoder}")
        print(f"  - Rotation rep: {rot_rep}")
        print(f"  - Decision heads: step_mode + terminate")

    def _load_pretrained_weights(self, pretrained_path: str, freeze: bool = False):
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Pretrained weights not found: {pretrained_path}")

        print(f"Loading pretrained RefineNet from: {pretrained_path}")

        checkpoint = torch.load(pretrained_path, map_location='cpu')

        if isinstance(checkpoint, dict):
            if 'model_state' in checkpoint:
                state_dict = checkpoint['model_state']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        encoder_state_dict = {}
        for key, value in state_dict.items():
            if any(key.startswith(prefix) for prefix in [
                'encodeA', 'encodeAB', 'pos_embed', 'trans_head', 'rot_head'
            ]):
                encoder_state_dict[key] = value

        missing_keys, unexpected_keys = self.encoder.load_state_dict(
            encoder_state_dict, strict=False
        )

        if missing_keys:
            print(f"  ⚠ Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"  ⚠ Unexpected keys: {len(unexpected_keys)}")

        loaded_count = len(encoder_state_dict)
        total_encoder_params = sum(1 for _ in self.encoder.state_dict().keys())

        print(f"  ✓ Loaded {loaded_count}/{total_encoder_params} encoder parameters")

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print(f"  ✓ Encoder parameters frozen")

    def forward(self, inputs):
        if not isinstance(inputs, tuple) or len(inputs) != 2:
            raise ValueError("Expected tuple (x1, x2)")

        x1, x2 = inputs
        bs = x1.size(0)

        x = torch.cat([x1, x2], dim=0)
        x = self.encoder.encodeA(x)

        a, b = x[:bs], x[bs:]
        ab = torch.cat([a, b], dim=1)
        ab = self.encoder.encodeAB(ab)

        features = ab.mean(dim=[2, 3])
        z = self.feature_projector(features)

        logits = self.actor(z)
        value = self.critic(z)
        step_mode_logit = self.step_mode_head(z)
        terminate_logit = self.terminate_head(z)

        return logits, value, step_mode_logit, terminate_logit

    def get_action_and_value(self, obs, action=None):
        logits, value, step_mode_logit, terminate_logit = self.forward(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, value.squeeze(-1), entropy, step_mode_logit, terminate_logit

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("✓ Encoder parameters unfrozen")


# ========= PPO Agent =========
class PPOAgent:
    def __init__(
            self,
            model: nn.Module,
            lr: float = 1e-4,
            gamma: float = 0.99,
            lam: float = 0.95,
            clip_eps: float = 0.3,
            value_coef: float = 0.3,
            entropy_coef: float = 0.05,
            bc_coef: float = 0.1,
            decision_coef: float = 0.5,
            terminate_pos_weight: float = 5.0,
            device: torch.device = torch.device("cpu")
    ):
        self.model = model
        self.optimizer = Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.bc_coef = bc_coef
        self.decision_coef = decision_coef
        self.device = device
        self.terminate_pos_weight = terminate_pos_weight

    def update(self, buffer: PPOBuffer, epochs: int = 6, batch_size: int = 32,
               last_value: float = 0.0, last_done: float = 1.0):
        returns, advantages = buffer.compute_returns_and_advantages(
            self.gamma, self.lam, self.device, last_value, last_done
        )
        total_loss = total_policy_loss = total_value_loss = total_entropy = total_bc = 0.0
        total_step_mode_loss = total_terminate_loss = 0.0
        update_count = 0

        self.model.train()

        for _ in range(epochs):
            for batch in buffer.get_batches(batch_size, self.device):
                obs = batch['obs']
                actions = batch['actions']
                old_log_probs = batch['old_log_probs']
                old_values = batch['old_values']
                batch_indices = batch['indices']
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                gt_step_modes = batch['gt_step_modes']
                gt_terminates = batch['gt_terminates']

                sample_w = torch.ones_like(batch_advantages, device=self.device)

                _, log_probs, values, entropy, step_mode_logit, terminate_logit = \
                    self.model.get_action_and_value(obs, actions)

                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                policy_loss_unreduced = -torch.min(surr1, surr2)
                policy_loss = (policy_loss_unreduced * sample_w).sum() / sample_w.sum()

                value_pred_clipped = old_values + torch.clamp(values - old_values, -self.clip_eps, self.clip_eps)
                value_losses = (values - batch_returns) ** 2
                value_losses_clipped = (value_pred_clipped - batch_returns) ** 2
                value_loss_unreduced = torch.max(value_losses, value_losses_clipped) * 0.5
                value_loss = (value_loss_unreduced * sample_w).sum() / sample_w.sum()

                entropy_mean = (entropy * sample_w).sum() / sample_w.sum()

                bc_loss = torch.tensor(0.0, device=self.device)

                # Decision losses
                step_mode_loss = F.binary_cross_entropy_with_logits(
                    step_mode_logit.squeeze(-1), gt_step_modes, reduction='mean'
                )

                terminate_loss = F.binary_cross_entropy_with_logits(
                    terminate_logit.squeeze(-1),
                    gt_terminates,
                    reduction='mean',
                    pos_weight=torch.tensor([2.0], device=self.device)
                    # pos_weight=torch.tensor([self.terminate_pos_weight], device=self.device)
                )
                loss = (policy_loss +
                        self.value_coef * value_loss -
                        self.entropy_coef * entropy_mean +
                        self.bc_coef * bc_loss +
                        self.decision_coef * (step_mode_loss + terminate_loss))

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_mean.item()
                total_bc += bc_loss.item()
                total_step_mode_loss += step_mode_loss.item()
                total_terminate_loss += terminate_loss.item()
                update_count += 1

        if update_count > 0:
            return {
                'loss': total_loss / update_count,
                'policy_loss': total_policy_loss / update_count,
                'value_loss': total_value_loss / update_count,
                'entropy': total_entropy / update_count,
                'bc_loss': total_bc / update_count,
                'step_mode_loss': total_step_mode_loss / update_count,
                'terminate_loss': total_terminate_loss / update_count,
            }
        return {}


# ========= Sanity check function =========
def sanity_check_determinism(model, obs, device, num_trials=10):
    """Check if model outputs are deterministic in eval mode"""
    model.eval()
    s = tensor01_from_obs(obs, device)
    probs = []

    for _ in range(num_trials):
        with torch.no_grad():
            logits, _, _, _ = model(s)
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())

    probs = np.stack(probs, 0)
    std_dev = probs.std(0).mean()
    greedy_action = probs.mean(0).argmax()

    print(f"Determinism check - std over {num_trials} forward passes: {std_dev:.8f}")
    print(f"Greedy action: {greedy_action}")

    if std_dev < 1e-6:
        print("✓ Model outputs are deterministic in eval mode")
    else:
        print("✗ Model outputs are not deterministic - check for dropout/batch norm issues")

    return std_dev < 1e-6


def train():
    # deterministic
    set_deterministic_training(seed=612)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    TARGET_IDS = (7, 34)  # Fixed ID combination (curriculum start)

    surfaces = get_default_surfaces()
    camera_params = get_default_camera_params()

    renderer = VTKRenderer(
        surfaces=surfaces,
        camera_params=camera_params,
        extrinsic_matrix=np.eye(4, dtype=np.float32),
        out_size=128,
    )

    # Load dataset
    original_pair_ds = PosePairDataset(
        pose_path="./data/training_pose.npy",
        id_path="./data/mat_id.npy",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32,
        ensure_unique_ids=True,
    )
    pair_ds = FixedPosePairDataset(original_pair_ds, TARGET_IDS)

    model_pts = load_vtk_points("./data/liver.vtk", device=device)

    out_size = 128
    env = SE3PoseEnvDiscrete(
        renderer=renderer,
        target_image_6ch=None,
        start_extrinsic=None,
        model_pts=model_pts,
        dataset=original_pair_ds,
        dataset_mode="random",
        # fixed_mat_ids=TARGET_IDS,
        out_size=out_size,
        target_mask_path="./largest.png",
        step_sizes=DiscreteStepSizes(trans_mm=10.0, rot_deg=5.0),
        max_steps=256,

        compose_mode="right",
        reward_mode='simple_step_with_final_bonus',
        reward_scale=1.0,
        success_threshold=10.0,
        success_bonus=50.0,
        empty_penalty=-2.0,
        handle_empty="undo",

        adaptive_step=True,
        coarse_trans_mm=10.0,
        coarse_rot_deg=5.0,
        fine_trans_mm=2.0,
        fine_rot_deg=1.0,
        fine_threshold=100.0,
    )
    env.target_augmenter = TargetImageAugmenter(
        device=env.device,
        apply_contour_aug=False,
        contour_aug_prob=0.2,
        skeleton_dilate_range=(1, 5),
        apply_depth_aug=False,
        depth_aug_prob=0.6,
        occluder_probability=0.6,
        mask_dilate_prob=0.0,
        depth_normalization_probability=0.6,
        scale_perturbation_probability=0.6,
    )
    env.augment_target_on_reset = True  # Enable augmentation
    # Sanity: first obs
    obs, info = reset_compat(env)
    if not isinstance(obs, tuple) or len(obs) != 2:
        raise ValueError("Expected tuple observation (current_img, target_img)")
    obs_shape = obs[0].shape
    n_actions = env.action_space.n
    print(f"Observation shape: {obs_shape}, Action count: {n_actions}")

    # Build RefineNet model with pretrained weights
    model = RefineNetActorCritic(
        num_actions=n_actions,
        input_size=out_size,
        dropout_rate=0.0,
        use_bn=False,
        rot_rep="6d",
        # pretrained_path="../Pytorch3D_renderer/0807_patient4_fold3_best_val.pth",
        freeze_encoder=True
    ).to(device)

    ppo_agent = PPOAgent(
        model=model,
        lr=1e-4,
        gamma=0.98,
        lam=0.95,
        clip_eps=0.3,
        value_coef=0.3,
        entropy_coef=0.05,
        bc_coef=0.0,
        decision_coef=0.5,
        terminate_pos_weight=1.0,
        device=device
    )
    # Optional: resume from checkpoint
    # resume_path = "./checkpoints/refinenet_ppo_1000.pt"
    resume_path = "refinenet_ppo_500.pt"
    if os.path.exists(resume_path):
        try:
            sd = torch.load(resume_path, map_location=device)
            model.load_state_dict(sd, strict=False)
            print(f"Loaded checkpoint: {resume_path}")
        except Exception as e:
            print(f"[Warning] Failed to load checkpoint: {e}")

    print("Running determinism sanity check...")
    sanity_check_determinism(model, obs, device)

    # Training parameters
    rollout_steps = 4096
    ppo_epochs = 4
    batch_size = 128
    num_updates = 4000

    buffer = PPOBuffer()

    print("Starting PPO training with RefineNet backbone and decision heads...")

    obs, info = reset_compat(env)
    state = tensor01_from_obs(obs, device)
    episode_start_obs = obs

    global_step = 0
    episode_count = 0
    episode_return = 0.0
    episode_step_count = 0

    SAVE_EVERY_EPISODES = 500
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("episode_images", exist_ok=True)

    for update in range(num_updates):
        buffer.clear()

        model.train()

        last_done = 1.0

        for step in range(rollout_steps):
            global_step += 1
            episode_step_count += 1

            with torch.no_grad():
                action, log_prob, value, _, step_mode_logit, terminate_logit = \
                    model.get_action_and_value(state)
                action_item = action.item()

            next_obs, reward, terminated, truncated, info = env.step(action_item)
            done = terminated or truncated

            curr_mse = float(info.get("mse", np.inf))
            episode_return += float(reward)

            # Keep transitions unless MSE is non-finite
            bad = not np.isfinite(curr_mse)
            if not bad:
                transition = PPOTransition(
                    obs1=state[0].squeeze(0).cpu().numpy(),
                    obs2=state[1].squeeze(0).cpu().numpy(),
                    action=action_item,
                    reward=float(reward),
                    value=value.item(),
                    log_prob=log_prob.item(),
                    done=done,
                    is_success=bool(info.get("success", False)),
                    gt_step_mode=info.get("gt_step_mode", 0.0),
                    gt_terminate=info.get("gt_terminate", 0.0),
                )
                buffer.push(transition)

            # Handle episode end
            if done:
                success = info.get("success", False)
                mse = curr_mse
                reason = info.get("terminated_reason", "unknown")

                # Save episode images
                episode_end_obs = next_obs if next_obs is not None else (
                    state[0].squeeze(0).cpu(), state[1].squeeze(0).cpu())
                end_obs_for_save = (
                    episode_end_obs[0].squeeze(0).cpu() if isinstance(episode_end_obs[0], torch.Tensor) else
                    episode_end_obs[0],
                    episode_end_obs[1].squeeze(0).cpu() if isinstance(episode_end_obs[1], torch.Tensor) else
                    episode_end_obs[1]
                )
                # if (episode_count % SAVE_EVERY_EPISODES == 0) or success:
                #     save_episode_images(
                #         start_obs=episode_start_obs,
                #         end_obs=end_obs_for_save,
                #         episode_count=episode_count,
                #         success=success,
                #         mse=mse,
                #         steps=episode_step_count
                #     )

                print(f"[Update {update + 1} | Episode {episode_count + 1}] "
                      f"Return={episode_return:.3f} Success={success} "
                      f"MSE={mse:.6f} Steps={episode_step_count} Reason={reason}")

                if episode_count % SAVE_EVERY_EPISODES == 0:
                    torch.save(model.state_dict(), f"checkpoints/refinenet_ppo_{episode_count}.pt")
                    print(f"[Checkpoint] Saved checkpoints/refinenet_ppo_{episode_count}.pt")

                episode_count += 1
                episode_return = 0.0
                episode_step_count = 0
                last_done = 1.0

                obs, info = reset_compat(env)
                episode_start_obs = obs
                state = tensor01_from_obs(obs, device)
            else:
                # Continue episode
                state = tensor01_from_obs(next_obs, device)
                last_done = 0.0

        # Compute last value for proper GAE bootstrap
        with torch.no_grad():
            _, _, last_value, _, _, _ = model.get_action_and_value(state)

        # PPO update with proper bootstrap
        stats = ppo_agent.update(
            buffer,
            epochs=ppo_epochs,
            batch_size=batch_size,
            last_value=last_value.item(),
            last_done=last_done
        )

        # Calculate decision heads accuracy
        step_acc = 0.0
        term_acc = 0.0
        step_pos_ratio = 0.0
        term_pos_ratio = 0.0

        if len(buffer.transitions) > 0:
            step_mode_preds = []
            step_mode_gts = []
            terminate_preds = []
            terminate_gts = []

            with torch.no_grad():
                for trans in buffer.transitions:
                    obs1 = torch.from_numpy(trans.obs1).unsqueeze(0).float().to(device)
                    obs2 = torch.from_numpy(trans.obs2).unsqueeze(0).float().to(device)
                    _, _, _, _, step_logit, term_logit = model.get_action_and_value((obs1, obs2))

                    step_mode_preds.append(torch.sigmoid(step_logit).item())
                    terminate_preds.append(torch.sigmoid(term_logit).item())
                    step_mode_gts.append(trans.gt_step_mode)
                    terminate_gts.append(trans.gt_terminate)

            step_preds_np = np.array(step_mode_preds)
            step_gts_np = np.array(step_mode_gts)
            term_preds_np = np.array(terminate_preds)
            term_gts_np = np.array(terminate_gts)

            step_acc = ((step_preds_np > 0.5) == step_gts_np).mean()
            term_acc = ((term_preds_np > 0.5) == term_gts_np).mean()
            step_pos_ratio = step_gts_np.mean()
            term_pos_ratio = term_gts_np.mean()

            term_pred_binary = (term_preds_np > 0.5).astype(int)
            term_gt_binary = term_gts_np.astype(int)

            tp = ((term_pred_binary == 1) & (term_gt_binary == 1)).sum()
            fp = ((term_pred_binary == 1) & (term_gt_binary == 0)).sum()
            fn = ((term_pred_binary == 0) & (term_gt_binary == 1)).sum()
            tn = ((term_pred_binary == 0) & (term_gt_binary == 0)).sum()

            total = tp + fp + fn + tn

            print(f"\n  [Terminate Confusion Matrix]")
            print(f"                Predicted: Continue | Predicted: Stop")
            print(
                f"  Actual: Continue    TN={tn:4d} ({tn / total * 100:5.1f}%) | FP={fp:4d} ({fp / total * 100:5.1f}%)")
            print(
                f"  Actual: Stop        FN={fn:4d} ({fn / total * 100:5.1f}%) | TP={tp:4d} ({tp / total * 100:5.1f}%)")
            print(f"  ")
            print(f"  Key Metrics:")
            print(f"    - Precision: {tp / (tp + fp) * 100 if (tp + fp) > 0 else 0:.1f}%")
            print(f"    - Recall: {tp / (tp + fn) * 100 if (tp + fn) > 0 else 0:.1f}%")
            print(f"    - FN Rate: {fn / (fn + tp) * 100 if (fn + tp) > 0 else 0:.1f}%")

        # Entropy coefficient annealing
        frac = 1.0 - (update + 1) / num_updates
        start_c, end_c = 0.05, 0.0
        ppo_agent.entropy_coef = end_c + (start_c - end_c) * max(frac, 0.0)

        # Print training stats
        if stats and (update + 1) % 1 == 0:
            print(f"\n[Update {update + 1}] Loss={stats['loss']:.4f} "
                  f"Policy={stats['policy_loss']:.4f} Value={stats['value_loss']:.4f} "
                  f"Entropy={stats['entropy']:.4f}")
            print(f"  Decision Loss: StepMode={stats['step_mode_loss']:.4f} "
                  f"Terminate={stats['terminate_loss']:.4f}")

            if len(buffer.transitions) > 0:
                print(f"  Decision Accuracy: StepMode={step_acc:.6%} (pos={step_pos_ratio:.6%}) | "
                      f"Terminate={term_acc:.6%} (pos={term_pos_ratio:.6%})")

    torch.save(model.state_dict(), "checkpoints/refinenet_ppo_final.pt")
    print("Training completed. Final model saved to checkpoints/")
    print("Episode images saved to episode_images/ directory")
    print("\nKey settings:")
    print("• RefineNet backbone with pretrained weights")
    print("• Decision heads for step mode and termination prediction")
    print("• Fixed pair + handle_empty='undo' (light penalty)")
    print("• Reward='simple_step_with_final_bonus'")
    print("• Rollout uses model.train() (BN consistency)")
    print("• Entropy anneals 0.05 → 0.00")


if __name__ == "__main__":

    train()
