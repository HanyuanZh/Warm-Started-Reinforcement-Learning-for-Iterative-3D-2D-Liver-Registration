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
from matrices_tranversion_tensor1 import pytorch3d_to_surgvtk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical

# Image saving related imports
import cv2

# Project modules
from dataset import FixedPosePairDataset
from dataset import PosePairDataset
from env_auto import (
    SE3PoseEnvDiscrete, DiscreteStepSizes,
    VTKRenderer, load_vtk_points,
    get_default_surfaces, get_default_camera_params,
)

# Deterministic setup
def set_deterministic_training(seed=629):
    """Set up deterministic training for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Utils
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
        target_mask  = ((target_mask  - target_mask.min())  /
                        (target_mask.max()  - target_mask.min()  + 1e-8) * 255).astype(np.uint8)
        return current_mask, target_mask

    try:
        start_current, start_target = extract_liver_mask(start_obs)
        end_current,   end_target   = extract_liver_mask(end_obs)
        h, w = start_current.shape
        grid_img = np.zeros((h * 2, w * 2), dtype=np.uint8)
        grid_img[0:h,   0:w]   = start_current
        grid_img[0:h,   w:2*w] = start_target
        grid_img[h:2*h, 0:w]   = end_current
        grid_img[h:2*h, w:2*w] = end_target
        grid_img_color = cv2.cvtColor(grid_img, cv2.COLOR_GRAY2BGR)
        font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        color = (0, 255, 0) if success else (0, 0, 255)
        cv2.putText(grid_img_color, "Start Current", (5, 15), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(grid_img_color, "Start Target",  (w + 5, 15), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(grid_img_color, "End Current",   (5, h + 15), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(grid_img_color, "End Target",    (w + 5, h + 15), font, font_scale, (255, 255, 255), thickness)
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

# PPO Buffer
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
            # Decision labels
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
                'gt_step_modes': gt_step_modes,
                'gt_terminates': gt_terminates,
            }

# RefineNet-based actor-critic with decision heads
class RefineNetActorCritic(nn.Module):
    """
    Actor-Critic network with decision heads for step mode and termination prediction.
    """
    def __init__(
        self,
        num_actions: int = 12,
        in_channels: int = 6,
        dropout: float = 0.0,
        pretrained_weights: Optional[str] = None
    ):
        super().__init__()
        self.num_actions = num_actions

        # Shared backbone (RefineNet-style)
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet-style blocks
        self.layer1 = self._make_layer(64, 128, 2)
        self.layer2 = self._make_layer(128, 256, 2)
        self.layer3 = self._make_layer(256, 512, 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)

        # Actor head (policy)
        self.fc_actor = nn.Linear(512, num_actions)
        
        # Critic head (value)
        self.fc_critic = nn.Linear(512, 1)
        
        # Decision heads
        self.fc_step_mode = nn.Linear(512, 1)
        self.fc_terminate = nn.Linear(512, 1)

        # Load pretrained weights if provided
        if pretrained_weights and os.path.exists(pretrained_weights):
            self._load_pretrained_weights(pretrained_weights)

    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(num_blocks - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)

    def _load_pretrained_weights(self, path):
        """Load pretrained weights from checkpoint"""
        print(f"Loading pretrained weights from {path}")
        state_dict = torch.load(path, map_location='cpu')
        
        # Filter out keys that don't match (e.g., decision heads)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() 
                          if k in model_dict and v.shape == model_dict[k].shape}
        
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=False)
        print(f"Loaded {len(pretrained_dict)} parameters from pretrained model")

    def forward(self, obs):
        """
        Args:
            obs: tuple of (current_img, target_img), each (B, C, H, W)
        Returns:
            logits, value, step_mode_logit, terminate_logit
        """
        x = torch.cat(obs, dim=1)
        
        # Backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        
        # Heads
        logits = self.fc_actor(x)
        value = self.fc_critic(x)
        step_mode_logit = self.fc_step_mode(x)
        terminate_logit = self.fc_terminate(x)
        
        return logits, value, step_mode_logit, terminate_logit

    def get_action_and_value(self, obs, action=None):
        """
        Get action, log_prob, value, entropy, and decision head outputs.
        """
        logits, value, step_mode_logit, terminate_logit = self.forward(obs)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
        
        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        
        return action, log_prob, value.squeeze(-1), entropy, step_mode_logit.squeeze(-1), terminate_logit.squeeze(-1)

# PPO Agent
class PPOAgent:
    def __init__(
        self,
        model: RefineNetActorCritic,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: torch.device = torch.device("cpu"),
        step_mode_coef: float = 0.1,
        terminate_coef: float = 0.1,
    ):
        self.model = model
        self.optimizer = Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.step_mode_coef = step_mode_coef
        self.terminate_coef = terminate_coef

    def compute_gae(self, rewards, values, dones, last_value, last_done):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
                next_done = last_done
            else:
                next_value = values[t + 1]
                next_done = dones[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - next_done) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - next_done) * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = advantages + torch.tensor(values, dtype=torch.float32, device=self.device)
        
        return advantages, returns

    def update(self, buffer: PPOBuffer, epochs: int, batch_size: int, 
               last_value: float, last_done: float):
        """Perform PPO update"""
        if len(buffer.transitions) == 0:
            return None

        # Compute advantages
        rewards = [t.reward for t in buffer.transitions]
        values = [t.value for t in buffer.transitions]
        dones = [float(t.done) for t in buffer.transitions]
        
        advantages, returns = self.compute_gae(rewards, values, dones, last_value, last_done)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training loop
        stats = {'loss': 0, 'policy_loss': 0, 'value_loss': 0, 'entropy': 0,
                 'step_mode_loss': 0, 'terminate_loss': 0}
        n_batches = 0

        for epoch in range(epochs):
            for batch in buffer.get_batches(batch_size, self.device):
                obs = batch['obs']
                actions = batch['actions']
                old_log_probs = batch['old_log_probs']
                old_values = batch['old_values']
                
                batch_size_actual = len(actions)
                batch_advantages = advantages[:batch_size_actual]
                batch_returns = returns[:batch_size_actual]
                
                # Forward pass
                _, new_log_probs, new_values, entropy, step_mode_logits, terminate_logits = \
                    self.model.get_action_and_value(obs, actions)
                
                # Policy loss (PPO clipped)
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(new_values, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Decision head losses
                gt_step_modes = batch['gt_step_modes']
                gt_terminates = batch['gt_terminates']
                
                step_mode_loss = F.binary_cross_entropy_with_logits(
                    step_mode_logits, gt_step_modes
                )
                terminate_loss = F.binary_cross_entropy_with_logits(
                    terminate_logits, gt_terminates
                )
                
                # Total loss
                loss = (policy_loss + 
                       self.value_coef * value_loss + 
                       self.entropy_coef * entropy_loss +
                       self.step_mode_coef * step_mode_loss +
                       self.terminate_coef * terminate_loss)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track stats
                stats['loss'] += loss.item()
                stats['policy_loss'] += policy_loss.item()
                stats['value_loss'] += value_loss.item()
                stats['entropy'] += entropy.mean().item()
                stats['step_mode_loss'] += step_mode_loss.item()
                stats['terminate_loss'] += terminate_loss.item()
                n_batches += 1

        # Average stats
        for key in stats:
            stats[key] /= max(n_batches, 1)

        return stats

# Training function
def train():
    # Hyperparameters
    TOTAL_TIMESTEPS = 500_000
    ROLLOUT_STEPS = 2048
    PPO_EPOCHS = 4
    BATCH_SIZE = 64
    LEARNING_RATE = 3e-4
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_EPSILON = 0.2
    VALUE_COEF = 0.5
    ENTROPY_COEF_START = 0.05
    MAX_GRAD_NORM = 0.5
    SAVE_EVERY_EPISODES = 100
    
    # Decision head coefficients
    STEP_MODE_COEF = 0.1
    TERMINATE_COEF = 0.1
    
    # Setup
    set_deterministic_training(seed=629)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataset
    base_dataset = PosePairDataset(
        pose_path="./data/Bartoli/training_pose.npy",
        id_path="./data/mat_id.npy",
        device=device,
        dtype=torch.float32,
        ensure_unique_ids=True
    )
    
    # Fixed pair for curriculum
    TARGET_IDS = (0, 50)
    dataset = FixedPosePairDataset(base_dataset, target_ids=TARGET_IDS)
    
    # Create environment
    env = SE3PoseEnvDiscrete(
        dataset=dataset,
        max_steps=50,
        step_sizes=DiscreteStepSizes(
            trans_step=5.0,
            rot_step_deg=2.0
        ),
        reward_mode="simple_step_with_final_bonus",
        reward_scale=1.0,
        success_threshold=30.0,
        success_bonus=50.0,
        final_bonus_scale=100.0,
        handle_empty="undo",
        empty_penalty=-5.0,
        out_size=128,
        device=device,
    )
    
    # Create model with pretrained weights
    model = RefineNetActorCritic(
        num_actions=12,
        in_channels=6,
        dropout=0.0,
        pretrained_weights="checkpoints/refinenet_pretrained.pt"
    ).to(device)
    
    # Create PPO agent
    ppo_agent = PPOAgent(
        model=model,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_epsilon=CLIP_EPSILON,
        value_coef=VALUE_COEF,
        entropy_coef=ENTROPY_COEF_START,
        max_grad_norm=MAX_GRAD_NORM,
        device=device,
        step_mode_coef=STEP_MODE_COEF,
        terminate_coef=TERMINATE_COEF,
    )
    
    # Training setup
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("episode_images", exist_ok=True)
    
    buffer = PPOBuffer()
    num_updates = TOTAL_TIMESTEPS // ROLLOUT_STEPS
    ppo_epochs = PPO_EPOCHS
    batch_size = BATCH_SIZE
    
    # Initialize
    obs, info = reset_compat(env)
    episode_start_obs = obs
    state = tensor01_from_obs(obs, device)
    
    episode_count = 0
    episode_return = 0.0
    episode_step_count = 0
    global_step = 0
    
    print("=" * 60)
    print("Starting PPO Training with RefineNet + Decision Heads")
    print("=" * 60)
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Rollout steps: {ROLLOUT_STEPS}")
    print(f"PPO epochs: {PPO_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Device: {device}")
    print(f"Target IDs: {TARGET_IDS}")
    print("=" * 60)
    
    # Training loop
    for update in range(num_updates):
        buffer.clear()

        model.train()

        last_done = 1.0

        for step in range(ROLLOUT_STEPS):
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
                episode_end_obs = next_obs if next_obs is not None else (state[0].squeeze(0).cpu(), state[1].squeeze(0).cpu())
                end_obs_for_save = (
                    episode_end_obs[0].squeeze(0).cpu() if isinstance(episode_end_obs[0], torch.Tensor) else episode_end_obs[0],
                    episode_end_obs[1].squeeze(0).cpu() if isinstance(episode_end_obs[1], torch.Tensor) else episode_end_obs[1]
                )

                print(f"[Update {update+1} | Episode {episode_count+1}] "
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
        
        # Decision accuracy calculation
        if len(buffer.transitions) > 0:
            step_mode_preds = []
            step_mode_gts = []
            terminate_preds = []
            terminate_gts = []
            
            # Extract predictions and labels
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
            
            # Calculate accuracy
            step_acc = ((step_preds_np > 0.5) == step_gts_np).mean()
            term_acc = ((term_preds_np > 0.5) == term_gts_np).mean()
            step_pos_ratio = step_gts_np.mean()
            term_pos_ratio = term_gts_np.mean()
            
            # Confusion matrix calculation
            term_pred_binary = (term_preds_np > 0.5).astype(int)
            term_gt_binary = term_gts_np.astype(int)
            
            tp = ((term_pred_binary == 1) & (term_gt_binary == 1)).sum()  # True Positive
            fp = ((term_pred_binary == 1) & (term_gt_binary == 0)).sum()  # False Positive
            fn = ((term_pred_binary == 0) & (term_gt_binary == 1)).sum()  # False Negative
            tn = ((term_pred_binary == 0) & (term_gt_binary == 0)).sum()  # True Negative
            
            total = tp + fp + fn + tn
            
            print(f"\n  [Terminate Confusion Matrix]")
            print(f"                Predicted: Continue | Predicted: Stop")
            print(f"  Actual: Continue    TN={tn:4d} ({tn/total*100:5.1f}%) | FP={fp:4d} ({fp/total*100:5.1f}%)")
            print(f"  Actual: Stop        FN={fn:4d} ({fn/total*100:5.1f}%) | TP={tp:4d} ({tp/total*100:5.1f}%)")
            print(f"  ")
            print(f"  Key Metrics:")
            print(f"    - Precision (stop correctness): {tp/(tp+fp)*100 if (tp+fp)>0 else 0:.1f}%")
            print(f"    - Recall (stop detection rate): {tp/(tp+fn)*100 if (tp+fn)>0 else 0:.1f}%")
            print(f"    - FN Rate (early stop rate): {fn/(fn+tp)*100 if (fn+tp)>0 else 0:.1f}% (lower is better)")
    

        # Entropy coefficient annealing
        frac = 1.0 - (update + 1) / num_updates
        start_c, end_c = 0.05, 0.0
        ppo_agent.entropy_coef = end_c + (start_c - end_c) * max(frac, 0.0)

        # Print training stats
        if stats and (update + 1) % 1 == 0:
            print(f"\n[Update {update+1}] Loss={stats['loss']:.4f} "
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
