from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple, Union, Optional, Literal, Dict

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

# Gym/Gymnasium
try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    import gym as gym
    from gym import spaces

# VTK & Pytorch3D
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,
    SoftPhongShader,
    TexturesVertex,
    AmbientLights,
    BlendParams,
)

# Kornia
import kornia.morphology as morph


# =========================
#        Renderer
# =========================
class VTKRenderer:
    """
    Merges VTK meshes into PyTorch3D Meshes.
    Only performs RGB shading rendering when needed; normally uses rasterizer once to get fragments/depth/pix_to_face.
    """
    def __init__(
        self,
        surfaces: Dict[str, Dict],
        camera_params: Dict[str, float],
        extrinsic_matrix: Union[np.ndarray, torch.Tensor],
        device: Union[torch.device, str, None] = None,
        out_size: int = 128,
        faces_per_pixel: int = 2,
        blur_radius: float = 0.0,
        use_shading: bool = False,
        zfar: float = 2000.0,
    ):
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.surfaces = surfaces
        self.use_shading = use_shading

        # Original (full-size) intrinsics
        self.W_full = float(camera_params["W"])
        self.H_full = float(camera_params["H"])
        self.fx_full = float(camera_params["fx"])
        self.fy_full = float(camera_params["fy"])
        self.cx_full = float(camera_params["cx"])
        self.cy_full = float(camera_params["cy"])

        # Target rendering resolution
        self.out_W = int(out_size)
        self.out_H = int(out_size)

        # Rasterization parameters
        self.faces_per_pixel = int(faces_per_pixel)
        self.blur_radius = float(blur_radius)
        self.zfar = float(zfar)

        self._init_camera_and_rasterizer(extrinsic_matrix)
        self._load_meshes()

        # Cache
        self._rendered = False
        self._fragments = None
        self._depth_buffer = None
        self._pix_to_face = None
        self._masks: Dict[str, torch.Tensor] = {}
        self._rgb_image = None

    def _init_camera_and_rasterizer(self, extrinsic_matrix: Union[np.ndarray, torch.Tensor]):
        if isinstance(extrinsic_matrix, np.ndarray):
            extrinsic = torch.from_numpy(extrinsic_matrix).float().to(self.device)
        else:
            extrinsic = extrinsic_matrix.float().to(self.device)

        R = extrinsic[:3, :3].unsqueeze(0)
        T = extrinsic[:3, 3].unsqueeze(0)

        # Scale intrinsics to out_size
        scale_x = self.out_W / self.W_full
        scale_y = self.out_H / self.H_full
        fx_small = self.fx_full * scale_x
        fy_small = self.fy_full * scale_y
        cx_small = self.cx_full * scale_x
        cy_small = self.cy_full * scale_y

        self.cameras = PerspectiveCameras(
            focal_length=((fx_small, fy_small),),
            principal_point=((cx_small, cy_small),),
            image_size=((self.out_H, self.out_W),),
            R=R, T=T, in_ndc=False, device=self.device
        )

        self.raster_settings = RasterizationSettings(
            image_size=(self.out_H, self.out_W),
            blur_radius=self.blur_radius,
            faces_per_pixel=self.faces_per_pixel,
            bin_size=0,  # Use naive rasterization to avoid overflow
            max_faces_per_bin=None,  # Not needed when bin_size=0
        )
        self.rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=self.raster_settings)

        self.blend_params = BlendParams(background_color=(0.0, 0.0, 0.0))
        if self.use_shading:
            self.lights = AmbientLights(device=self.device)
            self.shader = SoftPhongShader(device=self.device, cameras=self.cameras,
                                          lights=self.lights, blend_params=self.blend_params)
            self.rgb_renderer = MeshRenderer(rasterizer=self.rasterizer, shader=self.shader)
        else:
            self.lights = None
            self.shader = None
            self.rgb_renderer = None

    def _load_vtk_mesh_data(self, path: str, color):
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(path)
        reader.Update()
        polydata = reader.GetOutput()

        pts = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float32)
        verts = torch.from_numpy(pts).to(self.device)

        polys = vtk_to_numpy(polydata.GetPolys().GetData())
        indices = []
        i = 0
        while i < len(polys):
            n = int(polys[i])
            idx = polys[i + 1: i + 1 + n]
            if n == 3:
                indices.append([int(idx[0]), int(idx[1]), int(idx[2])])
            else:
                for j in range(1, n - 1):
                    indices.append([int(idx[0]), int(idx[j]), int(idx[j + 1])])
            i += n + 1

        faces = torch.tensor(indices, dtype=torch.int64, device=self.device)

        color_tensor = torch.tensor(color, dtype=torch.float32, device=self.device)
        verts_rgb = color_tensor.unsqueeze(0).expand(verts.shape[0], -1)
        return verts, faces, verts_rgb

    def _load_meshes(self):
        verts_list, faces_list, colors_list = [], [], []
        self.mesh_info = {}
        vert_accum, face_accum, comp_id = 0, 0, 0

        for name, cfg in self.surfaces.items():
            v, f, c = self._load_vtk_mesh_data(cfg["file"], cfg.get("color", [1.0, 1.0, 1.0]))
            num_v, num_f = v.shape[0], f.shape[0]

            self.mesh_info[name] = {
                "vert_start": vert_accum,
                "vert_end": vert_accum + num_v,
                "face_start": face_accum,
                "face_end": face_accum + num_f,
                "comp_id": comp_id,
                "color": cfg.get("color", [1.0, 1.0, 1.0])
            }

            faces_list.append(f + vert_accum)
            verts_list.append(v)
            colors_list.append(c)

            vert_accum += num_v
            face_accum += num_f
            comp_id += 1

        merged_verts = torch.cat(verts_list, dim=0)
        merged_faces = torch.cat(faces_list, dim=0)
        merged_colors = torch.cat(colors_list, dim=0)

        total_faces = merged_faces.shape[0]
        face_to_component = torch.empty((total_faces,), dtype=torch.int64, device=self.device)
        for name, info in self.mesh_info.items():
            f_s, f_e = info["face_start"], info["face_end"]
            face_to_component[f_s:f_e] = info["comp_id"]

        self.merged_verts = merged_verts
        self.merged_faces = merged_faces
        self.merged_colors = merged_colors
        self.face_to_component = face_to_component

        textures = TexturesVertex(verts_features=[merged_colors])
        self.mesh = Meshes(verts=[merged_verts], faces=[merged_faces], textures=textures)

    def render(self, target_mask_name: Optional[str] = None) -> Tuple[torch.Tensor, ...]:
        """
        Returns: (rgb_image, pix_to_face, depth_buffer)
        If target_mask_name is not None, generate mask only for that component.
        """
        fragments = self.rasterizer(self.mesh)
        self._fragments = fragments
        self._pix_to_face = fragments.pix_to_face
        self._depth_buffer = fragments.zbuf

        depth_map = fragments.zbuf[0, :, :, 0]
        depth_buffer = depth_map.clone()

        pix_to_face_flat = self._pix_to_face[0, :, :, 0]

        if self.use_shading and self.rgb_renderer is not None:
            images = self.rgb_renderer(self.mesh)
            rgb_img = images[0, :, :, :3]
        else:
            rgb_img = torch.zeros((self.out_H, self.out_W, 3), device=self.device, dtype=torch.float32)

        if target_mask_name is not None:
            if target_mask_name in self.mesh_info:
                info = self.mesh_info[target_mask_name]
                comp_id = info["comp_id"]
                face_ids = self.face_to_component[pix_to_face_flat]
                mask_single = (face_ids == comp_id).float()
                self._masks[target_mask_name] = mask_single
            else:
                self._masks[target_mask_name] = torch.zeros((self.out_H, self.out_W), device=self.device)

        self._rgb_image = rgb_img
        self._rendered = True
        return rgb_img, self._pix_to_face, depth_buffer

    def get_component_mask(self, name: str) -> torch.Tensor:
        """
        Returns mask for a specific component name (1.0 = visible, 0.0 = background).
        """
        if name in self._masks:
            return self._masks[name]
        if self._pix_to_face is None:
            raise RuntimeError("Must call render() first.")

        if name in self.mesh_info:
            info = self.mesh_info[name]
            comp_id = info["comp_id"]
            pix_to_face_flat = self._pix_to_face[0, :, :, 0]
            face_ids = self.face_to_component[pix_to_face_flat]
            mask = (face_ids == comp_id).float()
            self._masks[name] = mask
            return mask
        else:
            mask = torch.zeros((self.out_H, self.out_W), device=self.device)
            self._masks[name] = mask
            return mask

    def clear_cache(self):
        self._rendered = False
        self._fragments = None
        self._depth_buffer = None
        self._pix_to_face = None
        self._masks.clear()
        self._rgb_image = None


# =============================
#     Registration Env
# =============================
@dataclass
class EnvConfig:
    out_size: int = 128
    surfaces: Optional[Dict[str, Dict]] = None
    camera_params: Optional[Dict[str, float]] = None

    max_steps: int = 50
    trans_step: float = 5.0
    rot_step_deg: float = 2.0

    target_mask_name: str = "bottom_liver"
    source_mask_name: str = "liver"

    add_initial_noise: bool = True
    noise_trans_range: Tuple[float, float] = (-30.0, 30.0)
    noise_rot_range_deg: Tuple[float, float] = (-10.0, 10.0)

    reward_mode: Literal[
        "improvement",
        "neg_mse",
        "improvement_ratio",
        "improvement_log",
        "progress_log_to_thresh",
        "log_progress_with_threshold",
        "curr_first_positive",
        "curr_delta",
        "simple_step_with_final_bonus"
    ] = "simple_step_with_final_bonus"
    reward_scale: float = 1.0
    success_threshold: float = 50.0
    success_bonus: float = 50.0
    success_crossing_bonus: float = 0.0
    clip_reward: Optional[Tuple[float, float]] = None

    mse_small_bonus_threshold: float = 30.0
    mse_small_bonus: float = 5.0

    empty_penalty: float = -10.0
    handle_empty: Literal["terminate", "rollback", "penalty"] = "terminate"

    curr_first_alpha: float = 0.5
    threshold_focus_coef: float = 0.0

    final_bonus_scale: float = 100.0

    obs_mode: Literal["rgb", "mask_overlay"] = "mask_overlay"
    obs_overlay_alpha: float = 0.5

    bad_streak_limit: int = 0

    num_pairs: int = 1
    extrinsics: Optional[list] = None

    use_random_seed: bool = False
    episode_seed: Optional[int] = None


class RegistrationEnv(gym.Env):
    """
    Reinforcement Learning Environment for 2D/3D Registration.
    
    Action space: 6D continuous (tx, ty, tz, rx, ry, rz).
    Observation: Two images (current rendered image, target image).
    Reward: Based on improvement of MSE between masks.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, config: EnvConfig):
        super().__init__()
        self.cfg = config

        if self.cfg.surfaces is None:
            self.cfg.surfaces = get_default_surfaces()
        if self.cfg.camera_params is None:
            self.cfg.camera_params = get_default_camera_params()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        C = 3 if self.cfg.obs_mode == "rgb" else 1
        obs_h = self.cfg.out_size
        obs_w = self.cfg.out_size
        single_obs = spaces.Box(low=0, high=1, shape=(C, obs_h, obs_w), dtype=np.float32)
        self.observation_space = spaces.Tuple((single_obs, single_obs))

        self._action_names = [
            "tx+", "tx-", "ty+", "ty-", "tz+", "tz-",
            "rx+", "rx-", "ry+", "ry-", "rz+", "rz-"
        ]

        self.target_pairs = self._build_target_pairs()
        self.current_pair_idx = 0

        self.extrinsic = None
        self.renderer_source = None
        self.renderer_target = None
        self.target_mask = None
        self.curr_img = None

        self.step_count = 0
        self.score = 0.0
        self.last_mse = None
        self.initial_mse = None
        self.best_mse = None
        self.best_extrinsic = None
        self.bad_streak = 0
        self.episode_log = []

    def _build_target_pairs(self):
        if self.cfg.extrinsics is not None and len(self.cfg.extrinsics) > 0:
            return self.cfg.extrinsics
        
        # Generate random extrinsics
        np.random.seed(42)
        pairs = []
        for _ in range(self.cfg.num_pairs):
            tx = np.random.uniform(-50, 50)
            ty = np.random.uniform(-50, 50)
            tz = np.random.uniform(-100, 100)
            rx = np.deg2rad(np.random.uniform(-15, 15))
            ry = np.deg2rad(np.random.uniform(-15, 15))
            rz = np.deg2rad(np.random.uniform(-15, 15))
            
            R = self._euler_to_rotation_matrix(rx, ry, rz)
            T = np.array([tx, ty, tz], dtype=np.float32)
            extrinsic_gt = np.eye(4, dtype=np.float32)
            extrinsic_gt[:3, :3] = R
            extrinsic_gt[:3, 3] = T
            pairs.append(extrinsic_gt)
        return pairs

    @staticmethod
    def _euler_to_rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ], dtype=np.float32)
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ], dtype=np.float32)
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ], dtype=np.float32)
        return Rz @ Ry @ Rx

    def _init_renderers(self, target_extrinsic: np.ndarray):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.renderer_target = VTKRenderer(
            surfaces=self.cfg.surfaces,
            camera_params=self.cfg.camera_params,
            extrinsic_matrix=target_extrinsic,
            device=device,
            out_size=self.cfg.out_size,
            faces_per_pixel=2,
            blur_radius=0.0,
            use_shading=False,
        )
        _, _, _ = self.renderer_target.render(target_mask_name=self.cfg.target_mask_name)
        self.target_mask = self.renderer_target.get_component_mask(self.cfg.target_mask_name)

        self.renderer_source = VTKRenderer(
            surfaces=self.cfg.surfaces,
            camera_params=self.cfg.camera_params,
            extrinsic_matrix=self.extrinsic.cpu().numpy(),
            device=device,
            out_size=self.cfg.out_size,
            faces_per_pixel=2,
            blur_radius=0.0,
            use_shading=False,
        )

    def _add_noise_to_extrinsic(self, extrinsic_gt: np.ndarray, rng: np.random.Generator) -> torch.Tensor:
        tx_noise = rng.uniform(*self.cfg.noise_trans_range)
        ty_noise = rng.uniform(*self.cfg.noise_trans_range)
        tz_noise = rng.uniform(*self.cfg.noise_trans_range)
        
        rx_noise = np.deg2rad(rng.uniform(*self.cfg.noise_rot_range_deg))
        ry_noise = np.deg2rad(rng.uniform(*self.cfg.noise_rot_range_deg))
        rz_noise = np.deg2rad(rng.uniform(*self.cfg.noise_rot_range_deg))

        noise_rot = self._euler_to_rotation_matrix(rx_noise, ry_noise, rz_noise)
        noise_trans = np.array([tx_noise, ty_noise, tz_noise], dtype=np.float32)

        R_gt = extrinsic_gt[:3, :3]
        T_gt = extrinsic_gt[:3, 3]

        R_new = R_gt @ noise_rot
        T_new = T_gt + noise_trans

        noisy_extrinsic = np.eye(4, dtype=np.float32)
        noisy_extrinsic[:3, :3] = R_new
        noisy_extrinsic[:3, 3] = T_new

        return torch.from_numpy(noisy_extrinsic).float()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if self.cfg.use_random_seed:
            super().reset(seed=seed)
            rng = np.random.default_rng(seed)
        else:
            episode_seed = self.cfg.episode_seed if self.cfg.episode_seed is not None else 42
            rng = np.random.default_rng(episode_seed)

        if options and "pair_idx" in options:
            self.current_pair_idx = options["pair_idx"]
        else:
            self.current_pair_idx = rng.integers(0, len(self.target_pairs))

        target_extrinsic = self.target_pairs[self.current_pair_idx]

        if self.cfg.add_initial_noise:
            self.extrinsic = self._add_noise_to_extrinsic(target_extrinsic, rng)
        else:
            self.extrinsic = torch.from_numpy(target_extrinsic).float()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.extrinsic = self.extrinsic.to(device)

        self._init_renderers(target_extrinsic)

        self.step_count = 0
        self.score = 0.0
        self.last_mse = None
        self.initial_mse = None
        self.best_mse = None
        self.best_extrinsic = None
        self.bad_streak = 0
        self.episode_log = []

        curr_img, tgt_img = self._render_obs()
        self.curr_img = (curr_img, tgt_img)

        mse = self._compute_mse(curr_img, tgt_img)
        self.last_mse = mse
        self.initial_mse = mse

        info = {
            "mse": mse,
            "best_mse": mse,
            "delta_mse": 0.0,
            "step": 0,
            "success": False,
            "score": 0.0,
            "is_empty": False,
            "terminated_reason": "reset",
            "pair_idx": self.current_pair_idx,
            "step_mode": "reset",
            "pair_indices": (self.current_pair_idx, len(self.target_pairs)),
        }

        self.episode_log.append({
            "step": 0,
            "action_id": -1,
            "action_name": "reset",
            "mse": mse,
            "delta_mse": 0.0,
            "reward": 0.0,
            "mse_bonus": 0.0,
            "is_empty": False,
            "pair_idx": self.current_pair_idx,
            "step_mode": "reset",
            "pair_indices": (self.current_pair_idx, len(self.target_pairs)),
        })

        obs = (curr_img.float().clamp(0, 1), tgt_img.float().clamp(0, 1))
        return obs, info

    def _render_obs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.renderer_source.clear_cache()
        self.renderer_source._init_camera_and_rasterizer(self.extrinsic.cpu().numpy())
        _, _, _ = self.renderer_source.render(target_mask_name=self.cfg.source_mask_name)
        source_mask = self.renderer_source.get_component_mask(self.cfg.source_mask_name)

        if self.cfg.obs_mode == "rgb":
            curr_rgb = self.renderer_source._rgb_image
            tgt_rgb = self.renderer_target._rgb_image
            curr_img = curr_rgb.permute(2, 0, 1)
            tgt_img = tgt_rgb.permute(2, 0, 1)
        else:
            alpha = self.cfg.obs_overlay_alpha
            curr_overlay = self._overlay_mask(source_mask, alpha)
            tgt_overlay = self._overlay_mask(self.target_mask, alpha)
            curr_img = curr_overlay.unsqueeze(0)
            tgt_img = tgt_overlay.unsqueeze(0)

        return curr_img, tgt_img

    @staticmethod
    def _overlay_mask(mask: torch.Tensor, alpha: float) -> torch.Tensor:
        return alpha * mask + (1 - alpha) * torch.ones_like(mask)

    def _compute_mse(self, curr_img: torch.Tensor, tgt_img: torch.Tensor) -> float:
        diff = (curr_img - tgt_img) ** 2
        mse_val = diff.mean().item()
        return mse_val

    def _is_empty_image(self, img: torch.Tensor, threshold: float = 1e-4) -> bool:
        return (img.abs().max().item() < threshold)

    @staticmethod
    def _reward_curr_first_positive_symmetric(
        prev_mse: float,
        curr_mse: float,
        m_ref: float,
        alpha: float = 0.5,
    ) -> float:
        if not np.isfinite(prev_mse) or not np.isfinite(curr_mse):
            return 0.0
        
        eps = 1e-9
        m_ref = max(m_ref, eps)
        
        if curr_mse < m_ref:
            w = alpha
        else:
            w = 1.0 - alpha
        
        delta = prev_mse - curr_mse
        r = w * delta / max(prev_mse, eps)
        return float(r)

    def step(self, action: Union[np.ndarray, int]):
        if isinstance(action, int):
            action_id = action
            step_mode = "discrete"
        else:
            action = np.asarray(action, dtype=np.float32)
            action_id = np.argmax(np.abs(action))
            step_mode = "continuous"

        delta = self._action_to_delta(action_id)
        self.extrinsic = torch.matmul(self.extrinsic, delta)
        self.step_count += 1

        curr_img, tgt_img = self._render_obs()
        self.curr_img = (curr_img, tgt_img)

        mse = self._compute_mse(curr_img, tgt_img)
        is_empty = self._is_empty_image(curr_img)

        delta_mse = (self.last_mse - mse) if (self.last_mse is not None) else 0.0

        terminated = False
        truncated = (self.step_count >= self.cfg.max_steps)
        reward = 0.0

        if delta_mse < 0:
            self.bad_streak += 1
        else:
            self.bad_streak = 0

        if is_empty:
            if self.cfg.handle_empty == "terminate":
                reward = self.cfg.empty_penalty
                terminated = True
            elif self.cfg.handle_empty == "rollback":
                if self.best_extrinsic is not None:
                    self.extrinsic = self.best_extrinsic.clone()
                    curr_img, tgt_img = self._render_obs()
                    self.curr_img = (curr_img, tgt_img)
                reward = self.cfg.empty_penalty * 0.5
                terminated = True
            else:
                reward = self.cfg.empty_penalty

            if (self.best_mse is None) or (mse < self.best_mse):
                self.best_mse = mse
                self.best_extrinsic = self.extrinsic.clone()

            if (self.bad_streak_limit > 0) and (self.bad_streak >= self.bad_streak_limit):
                if self.best_extrinsic is not None:
                    self.extrinsic = self.best_extrinsic.clone()
                    curr_img, tgt_img = self._render_obs()
                    self.curr_img = (curr_img, tgt_img)
                reward = self.cfg.empty_penalty * 0.5
                terminated = True

        else:
            base_reward = 0.0
            eps = 1e-6
            c = 1.0

            if self.cfg.reward_mode == "improvement":
                base_reward = delta_mse

            elif self.cfg.reward_mode == "neg_mse":
                base_reward = -mse

            elif self.cfg.reward_mode == "improvement_ratio":
                if (self.last_mse is not None) and np.isfinite(self.last_mse) and (self.last_mse > eps) and np.isfinite(mse):
                    base_reward = (self.last_mse - mse) / max(self.last_mse, eps)
                else:
                    base_reward = 0.0

            elif self.cfg.reward_mode == "improvement_log":
                if (self.last_mse is not None) and np.isfinite(self.last_mse) and np.isfinite(mse):
                    base_reward = float(np.log(self.last_mse + c) - np.log(mse + c))
                else:
                    base_reward = 0.0

            elif self.cfg.reward_mode == "progress_log_to_thresh":
                th = max(self.cfg.success_threshold, eps)
                if (self.last_mse is not None) and np.isfinite(self.last_mse) and np.isfinite(mse):
                    prev_term = float(np.log(max(self.last_mse, th)))
                    curr_term = float(np.log(max(mse, th)))
                    base_reward = prev_term - curr_term
                else:
                    base_reward = 0.0

            elif self.cfg.reward_mode == "log_progress_with_threshold":
                th = max(self.cfg.success_threshold, eps)
                if (self.last_mse is not None) and np.isfinite(self.last_mse) and np.isfinite(mse):
                    log_improve = float(np.log(self.last_mse + c) - np.log(mse + c))
                    prev_term = float(np.log(max(self.last_mse, th)))
                    curr_term = float(np.log(max(mse, th)))
                    toward_thresh = max(prev_term - curr_term, 0.0)
                else:
                    log_improve = 0.0
                    toward_thresh = 0.0
                base_reward = log_improve + self.cfg.threshold_focus_coef * toward_thresh

            elif self.cfg.reward_mode == "curr_first_positive":
                prev = self.last_mse if (self.last_mse is not None and np.isfinite(self.last_mse)) else mse
                base_reward = self._reward_curr_first_positive_symmetric(
                    prev_mse=prev,
                    curr_mse=mse,
                    m_ref=self.cfg.success_threshold,
                    alpha=self.cfg.curr_first_alpha,
                )

            elif self.cfg.reward_mode == "curr_delta":
                if (self.last_mse is not None) and np.isfinite(self.last_mse) and np.isfinite(mse):
                    base_reward = (self.last_mse - mse) / max(self.last_mse, eps)
                else:
                    base_reward = 0.0

            elif self.cfg.reward_mode == "simple_step_with_final_bonus":
                if (self.last_mse is not None) and np.isfinite(self.last_mse) and np.isfinite(mse):
                    if mse < self.last_mse:
                        base_reward = 1.0
                    elif mse > self.last_mse:
                        base_reward = -1.0
                    else:
                        base_reward = 0.0
                else:
                    base_reward = 0.0

            else:
                raise ValueError(f"Unknown reward_mode: {self.cfg.reward_mode}")

            reward = self.cfg.reward_scale * float(base_reward)

            mse_bonus = 0.0
            if np.isfinite(mse) and mse <= self.cfg.mse_small_bonus_threshold:
                reward += self.cfg.mse_small_bonus
                mse_bonus = self.cfg.mse_small_bonus

            reached = (mse < self.cfg.success_threshold)
            crossed = reached and ((self.last_mse is None) or (self.last_mse >= self.cfg.success_threshold))
            if reached:
                reward += self.cfg.success_bonus
                if crossed:
                    reward += self.cfg.success_crossing_bonus
                terminated = True

            # Final bonus (after terminated is set)
            if (terminated or truncated) and self.cfg.reward_mode == "simple_step_with_final_bonus":
                if (self.initial_mse is not None) and np.isfinite(self.initial_mse) and np.isfinite(mse):
                    improvement_ratio = (self.initial_mse - mse) / max(self.initial_mse, eps)
                    if improvement_ratio > 0:
                        final_bonus = self.cfg.final_bonus_scale * improvement_ratio
                        reward += final_bonus
                        
                        if mse < self.cfg.success_threshold:
                            reward += 0.0

            if self.cfg.clip_reward is not None:
                reward = float(np.clip(reward, self.cfg.clip_reward[0], self.cfg.clip_reward[1]))

            self.last_mse = mse
            if self.best_mse is None or mse < self.best_mse:
                self.best_mse = mse

        obs = (curr_img.float().clamp(0, 1), tgt_img.float().clamp(0, 1))
        self.score += float(reward)

        if is_empty and self.cfg.handle_empty in ("terminate", "rollback"):
            terminated_reason = "empty_image"
        elif (self.bad_streak_limit > 0) and (self.bad_streak >= self.bad_streak_limit):
            terminated_reason = "rollback"
        elif (not is_empty) and (mse < self.cfg.success_threshold):
            terminated_reason = "success"
        elif truncated:
            terminated_reason = "max_steps"
        else:
            terminated_reason = "ongoing"

        info = {
            "mse": mse,
            "best_mse": self.best_mse,
            "delta_mse": delta_mse,
            "step": self.step_count,
            "success": bool(not is_empty and np.isfinite(mse) and mse < self.cfg.success_threshold),
            "score": self.score,
            "is_empty": is_empty,
            "terminated_reason": terminated_reason,
            "pair_idx": self.current_pair_idx,
            "step_mode": step_mode,
        }

        t = delta[:3, 3]
        info["delta_t"] = (float(t[0]), float(t[1]), float(t[2]))
        info["pair_indices"] = self.episode_log[0].get("pair_indices", (-1, -1))
        
        self.episode_log.append({
            "step": self.step_count,
            "action_id": int(action_id),
            "action_name": self._action_names[int(action_id)],
            "mse": mse,
            "delta_mse": delta_mse,
            "reward": float(reward),
            "mse_bonus": float(mse_bonus),
            "is_empty": is_empty,
            "pair_idx": self.current_pair_idx,
            "step_mode": step_mode,
        })

        return obs, float(reward), terminated, truncated, info

    def _action_to_delta(self, action_id: int) -> torch.Tensor:
        delta = torch.eye(4, device=self.extrinsic.device, dtype=torch.float32)
        
        if action_id == 0:    # tx+
            delta[0, 3] = self.cfg.trans_step
        elif action_id == 1:  # tx-
            delta[0, 3] = -self.cfg.trans_step
        elif action_id == 2:  # ty+
            delta[1, 3] = self.cfg.trans_step
        elif action_id == 3:  # ty-
            delta[1, 3] = -self.cfg.trans_step
        elif action_id == 4:  # tz+
            delta[2, 3] = self.cfg.trans_step
        elif action_id == 5:  # tz-
            delta[2, 3] = -self.cfg.trans_step
        elif action_id == 6:  # rx+
            angle = np.deg2rad(self.cfg.rot_step_deg)
            delta[:3, :3] = torch.tensor([
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)]
            ], device=delta.device, dtype=torch.float32)
        elif action_id == 7:  # rx-
            angle = np.deg2rad(-self.cfg.rot_step_deg)
            delta[:3, :3] = torch.tensor([
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)]
            ], device=delta.device, dtype=torch.float32)
        elif action_id == 8:  # ry+
            angle = np.deg2rad(self.cfg.rot_step_deg)
            delta[:3, :3] = torch.tensor([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ], device=delta.device, dtype=torch.float32)
        elif action_id == 9:  # ry-
            angle = np.deg2rad(-self.cfg.rot_step_deg)
            delta[:3, :3] = torch.tensor([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ], device=delta.device, dtype=torch.float32)
        elif action_id == 10:  # rz+
            angle = np.deg2rad(self.cfg.rot_step_deg)
            delta[:3, :3] = torch.tensor([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ], device=delta.device, dtype=torch.float32)
        elif action_id == 11:  # rz-
            angle = np.deg2rad(-self.cfg.rot_step_deg)
            delta[:3, :3] = torch.tensor([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ], device=delta.device, dtype=torch.float32)
        
        return delta


# =============================
# Utilities for data/params
# =============================
def load_vtk_points(
        filename: str,
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cuda",
) -> torch.Tensor:
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()

    vtk_points = reader.GetOutput().GetPoints()
    n = vtk_points.GetNumberOfPoints()
    pts_np = np.array([vtk_points.GetPoint(i) for i in range(n)], dtype=np.float32)
    return torch.as_tensor(pts_np, dtype=dtype, device=device)


def get_default_surfaces():
    return {
        "liver":        {"file": "./data/upper_liver_new.vtk",  "color": [1.0, 0.8, 0.3], "opacity": 1.0},
        "left_ridge":   {"file": "./data/left_ridge.vtk",   "color": [0.7, 0.7, 0.7], "opacity": 1.0},
        "ligament":     {"file": "./data/ligament.vtk",     "color": [0.9, 0.6, 0.6], "opacity": 1.0},
        "right_ridge":  {"file": "./data/right_ridge.vtk",  "color": [0.6, 0.6, 0.9], "opacity": 1.0},
        "bottom_liver": {"file": "./data/bottom_liver_new.vtk", "color": [0.8, 0.8, 0.8], "opacity": 1.0},
    }


def get_default_camera_params():
    return {
        'W': 3840, 'H': 2160,
        'fx': 2606.479298, 'fy': 2606.479298,
        'cx': 1925.19481723, 'cy': 1038.7789224
    }
