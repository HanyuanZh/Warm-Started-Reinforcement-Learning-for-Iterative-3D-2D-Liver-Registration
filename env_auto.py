from __future__ import annotations
from PIL import Image  # NEW
import os
from dataclasses import dataclass
from typing import Tuple, Union, Optional, Literal, Dict

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from matrices_tranversion_tensor1 import pytorch3d_to_surgvtk,vtk_2_PyTorch3D,vtk_2_surgvtk,surgvtk_2_vtk
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
    Merge VTK meshes → PyTorch3D Meshes
    Only perform RGB shading rendering when needed; normally just use rasterizer once to get fragments / depth / pix_to_face.
    """
    def __init__(
        self,
        surfaces: Dict[str, Dict],
        camera_params: Dict[str, float],
        extrinsic_matrix: Union[np.ndarray, torch.Tensor],
        device: Union[torch.device, str, None] = None,
        out_size: int = 128,
        faces_per_pixel: int = 1,
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
        self.face_to_component = face_to_component
        self.comp_id_to_name = {info["comp_id"]: name for name, info in self.mesh_info.items()}

        textures = TexturesVertex(verts_features=merged_colors.unsqueeze(0))
        self.merged_mesh = Meshes(verts=[merged_verts], faces=[merged_faces], textures=textures)

    def _ensure_rasterized(self):
        if self._rendered and (self._fragments is not None):
            return
        fragments = self.rasterizer(self.merged_mesh)
        depth = fragments.zbuf[0, ..., 0]            # (H,W)
        pix2face = fragments.pix_to_face[0, ..., 0]  # (H,W)

        comp_ids = torch.full((self.out_H, self.out_W), -1, dtype=torch.int64, device=self.device)
        valid = (pix2face >= 0)
        comp_ids[valid] = self.face_to_component[pix2face[valid]]

        masks = {}
        for comp_id, name in self.comp_id_to_name.items():
            masks[name] = (comp_ids == comp_id).to(torch.uint8)

        self._fragments = fragments
        self._depth_buffer = depth
        self._pix_to_face = pix2face
        self._masks = masks
        self._rendered = True
        self._rgb_image = None

    def render_rgb(self):
        if not self.use_shading:
            raise RuntimeError("RGB renderer not constructed when use_shading=False.")
        if self._rgb_image is None:
            self._ensure_rasterized()
            img = self.rgb_renderer(self.merged_mesh)  # (1,H,W,3)
            self._rgb_image = img[0, ..., :3]          # (H,W,3)
        return self._rgb_image.clone()

    def get_depth(self, invert: bool = True, eps: float = 1e-6,
                  use_percentile: bool = True, p: float = 0.95,
                  component_name: str = None):
        self._ensure_rasterized()
        depth = self._depth_buffer.clone()
        pix2face = self._pix_to_face

        valid = (pix2face >= 0)
        if not invert:
            out = depth
        else:
            inv = torch.zeros_like(depth)
            inv[valid] = 1.0 / (depth[valid] + eps)
            if valid.any():
                if use_percentile:
                    scale = torch.quantile(inv[valid], q=p).clamp_min(eps)
                else:
                    scale = inv[valid].max().clamp_min(eps)
                inv = (inv / scale).clamp_(0, 1)
            out = inv

        if component_name is not None:
            mask = self.get_mask(component_name).float()
            out = out * mask
        return out

    def get_mask(self, component_name: str = None):
        self._ensure_rasterized()
        if component_name is None:
            return {n: m.clone() for n, m in self._masks.items()}
        if component_name not in self._masks:
            raise ValueError(f"Component '{component_name}' does not exist, available: {list(self._masks.keys())}")
        return self._masks[component_name].clone()

    def update_camera_matrix(self, new_extrinsic_matrix: Union[torch.Tensor, np.ndarray]):
        """
        Update camera extrinsic matrix (automatically check and correct rotation part to be orthogonal)
        """
        # Convert to tensor
        if isinstance(new_extrinsic_matrix, np.ndarray):
            extrinsic = torch.from_numpy(new_extrinsic_matrix).float().to(self.device)
        else:
            extrinsic = new_extrinsic_matrix.float().to(self.device)
    
        # ==============
        # Auto-orthogonalize R
        # ==============
        R_new = extrinsic[:3, :3]
        T_new = extrinsic[:3, 3]
    
        # Check if orthogonal
        with torch.no_grad():
            I = torch.eye(3, device=R_new.device, dtype=R_new.dtype)
            det_R = torch.det(R_new).item()
            orth_err = torch.norm(R_new.T @ R_new - I).item()
    
            if (abs(det_R - 1.0) > 1e-3) or (orth_err > 1e-3):
                # Fix using SVD
                U, _, Vt = torch.linalg.svd(R_new)
                R_new = U @ Vt
                # Ensure det(R)=+1
                if torch.det(R_new) < 0:
                    U[:, -1] *= -1
                    R_new = U @ Vt
    
        R_new = R_new.unsqueeze(0)
        T_new = T_new.unsqueeze(0)
    
        # Update camera
        self.cameras.R = R_new
        self.cameras.T = T_new
        self.rasterizer.cameras = self.cameras
        if self.shader is not None:
            self.shader.cameras = self.cameras
    
        # Reset cache
        self._rendered = False
        self._fragments = None
        self._depth_buffer = None
        self._pix_to_face = None
        self._masks = {}
        self._rgb_image = None



def process_liver_masks_tensor_all_torch_ultra_fast(
    liver_mask: torch.Tensor,
    left_ridge: torch.Tensor,
    right_ridge: torch.Tensor,
    ligament: torch.Tensor,
    bottom_liver: torch.Tensor,
    device: Union[torch.device, str] = "cuda",
    out_size: int = 128,
):
    device = torch.device(device)

    def _resize01(x):
        x = x.to(device).float()
        x = F.interpolate(x.unsqueeze(0).unsqueeze(0), size=(out_size, out_size), mode="nearest")
        return x.squeeze(0).squeeze(0)  # (H,W)

    liver = _resize01(liver_mask)
    left = _resize01(left_ridge)
    right = _resize01(right_ridge)
    lig = _resize01(ligament)
    bl = _resize01(bottom_liver)

    H, W = out_size, out_size
    if (left.sum() + right.sum() + lig.sum() + liver.sum()) / (H * W) < 1e-4:
        return None

    kernel3 = torch.ones((3, 3), device=device)
    dilated = morph.dilation(liver.unsqueeze(0).unsqueeze(0), kernel3).squeeze(0).squeeze(0)
    edges = ((dilated > 0) & (liver == 0)).float()

    kernel15 = torch.ones((15, 15), device=device)
    spine_dil = morph.dilation((left + right + lig + bl).unsqueeze(0).unsqueeze(0), kernel15).squeeze(0).squeeze(0)
    spine_dil = (spine_dil > 0).float()

    edges_clean = edges.clone()
    edges_clean[spine_dil > 0] = 0.0

    final = torch.zeros((4, H, W), dtype=torch.uint8, device=device)
    final[3][edges_clean > 0] = 255  # Edge
    final[2][left > 0] = 255        # R
    final[1][right > 0] = 255       # G
    final[0][lig > 0] = 255         # B

    return final.permute(1, 2, 0)  # (H,W,4)


def re_rendering(
    matrix: Union[torch.Tensor, np.ndarray],
    renderer: VTKRenderer,
    device: Union[torch.device, str] = "cuda",
    out_size: int = 128,
) -> torch.Tensor:
    """
    Returns (6, H, W), channels:
      [ B(ligament), G(right_ridge), R(left_ridge), Edge, invDepth(normalized), liverMask ]
    """
    device = torch.device(device)
    renderer.update_camera_matrix(matrix)

    parts = ["liver", "left_ridge", "right_ridge", "ligament", "bottom_liver"]
    masks = {p: renderer.get_mask(p) for p in parts}  # (H,W), uint8

    if masks["liver"].sum() == 0:
        return torch.zeros((6, out_size, out_size), device=device)

    sem_hwC = process_liver_masks_tensor_all_torch_ultra_fast(
        liver_mask=masks["liver"],
        left_ridge=masks["left_ridge"],
        right_ridge=masks["right_ridge"],
        ligament=masks["ligament"],
        bottom_liver=masks["bottom_liver"],
        device=device,
        out_size=out_size
    )
    if sem_hwC is None:
        return torch.zeros((6, out_size, out_size), device=device)

    sem = (sem_hwC.permute(2, 0, 1).float() / 255.0).to(device)  # (4,H,W)

    depth_inv = renderer.get_depth(invert=True, component_name="liver")  # (H,W)
    depth_inv = F.interpolate(depth_inv.unsqueeze(0).unsqueeze(0), size=(out_size, out_size), mode="nearest").squeeze(0).squeeze(0)

    liver_mask = masks["liver"].float()
    liver_mask = F.interpolate(liver_mask.unsqueeze(0).unsqueeze(0), size=(out_size, out_size), mode="nearest").squeeze(0).squeeze(0)

    out = torch.cat((sem, depth_inv.unsqueeze(0), liver_mask.unsqueeze(0)), dim=0)  # (6,H,W)
    return out


# =============================
#   Lie utilities: SO(3), SE(3)
# =============================
def _skew(w: Tensor) -> Tensor:
    wx, wy, wz = w.unbind()
    zeros = torch.zeros_like(wx)
    row1 = torch.stack([zeros, -wz, wy])
    row2 = torch.stack([wz, zeros, -wx])
    row3 = torch.stack([-wy, wx, zeros])
    return torch.stack([row1, row2, row3])


def so3_exp(w: Tensor) -> Tensor:
    theta = torch.linalg.norm(w)
    I = torch.eye(3, dtype=w.dtype, device=w.device)
    K = _skew(w)
    K2 = K @ K
    if theta < 1e-8:
        return I + K + 0.5 * K2
    s, c = torch.sin(theta), torch.cos(theta)
    return I + (s / theta) * K + ((1 - c) / (theta * theta)) * K2


def se3_exp(xi: Tensor) -> Tensor:
    assert xi.numel() == 6
    v = xi[:3]
    w = xi[3:]
    R = so3_exp(w)
    theta = torch.linalg.norm(w)
    I = torch.eye(3, dtype=xi.dtype, device=xi.device)
    K = _skew(w)
    K2 = K @ K
    if theta < 1e-8:
        V = I + 0.5 * K + (1.0 / 6.0) * K2
    else:
        s, c = torch.sin(theta), torch.cos(theta)
        V = I + ((1 - c) / (theta * theta)) * K + ((theta - s) / (theta ** 3)) * K2
    t = V @ v
    Tm = torch.eye(4, dtype=xi.dtype, device=xi.device)
    Tm[:3, :3] = R
    Tm[:3, 3] = t
    return Tm


# =============================
#        Dataclasses
# =============================
@dataclass
class DiscreteStepSizes:
    trans_mm: float = 1.0
    rot_deg: float = 1.0

    @property
    def rot_rad(self) -> float:
        return float(np.deg2rad(self.rot_deg))


# =============================
#           Env (MSE reward)
# =============================
class SE3PoseEnvDiscrete(gym.Env):
    """
    Observation: Tuple( current_6ch, target_6ch )
    Reward (supports multiple modes):
      - "improvement": absolute improvement in MSE
      - "neg_mse": directly use -mse
      - "improvement_ratio": relative improvement (Δ/mse_prev)
      - "improvement_log": logarithmic improvement log(mse_prev+c)-log(mse+c)
      - "progress_log_to_thresh": logarithmic progress with threshold reference
      - "log_progress_with_threshold": log improvement + threshold approach reinforcement + first-time crossing bonus
      - "curr_first_positive": focus on current error with improvement as supplement (strictly symmetric negative penalty)
      - "curr_delta": allow negative penalty; normalized increment based on current step MSE change
      - "simple_step_with_final_bonus": ±1 per step, final bonus based on overall improvement ratio

    Optimization: target image is rendered only once per episode and cached
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        renderer: VTKRenderer,
        target_image_6ch: Optional[Tensor] = None,
        start_extrinsic: Optional[Tensor] = None,
        out_size: int = 128,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        step_sizes: DiscreteStepSizes = None,
        max_steps: int = 256,
        compose_mode: Literal["left", "right"] = "left",
        model_pts: Optional[Tensor] = None,
        target_extrinsic: Optional[Tensor] = None,
        empty_image_threshold: float = 1e-6,
        check_empty_image: bool = True,
        dataset: Optional[torch.utils.data.Dataset] = None,
        dataset_mode: Literal["sequential", "random", "fixed"] = "sequential",
        dataset_start_index: int = 0,
        fixed_mat_ids: Optional[Tuple[int, int]] = None,
        reward_mode: Literal[
            "improvement",
            "neg_mse",
            "improvement_ratio",
            "improvement_log",
            "progress_log_to_thresh",
            "log_progress_with_threshold",
            "curr_first_positive",
            "curr_delta",
            "simple_step_with_final_bonus",
        ] = "curr_first_positive",
        reward_scale: float = 1.0,
        success_threshold: float = 300.0,
        success_bonus: float = 15.0,
        success_crossing_bonus: float = 10.0,
        threshold_focus_coef: float = 1.0,
        empty_penalty: float = -5.0,
        mse_small_bonus_threshold: float = 1200.0,
        mse_small_bonus: float = 0.0,
        clip_reward: Optional[Tuple[float, float]] = None,
        final_bonus_scale: float = 100.0,
        curr_first_alpha: float = 0.1,
        handle_empty: Literal["terminate", "undo", "rollback"] = "undo",
        bad_streak_limit: int = 0,
        adaptive_step: bool = True,
        coarse_trans_mm: float = 5.0,
        coarse_rot_deg: float = 5.0,
        fine_trans_mm: float = 2.0,
        fine_rot_deg: float = 2.0,
        fine_threshold: float = 100.0,

        # ---- NEW: target binary mask related parameters ----
        target_mask_path: Optional[str] = None,
        target_mask_threshold: float = 0.5,
        target_mask_invert: bool = False,
    ) -> None:
        super().__init__()
        assert compose_mode in ("left", "right")

        if step_sizes is None:
            step_sizes = DiscreteStepSizes()

        self.renderer = renderer
        self.device = torch.device(device)
        self.out_size = int(out_size)
        self.compose_mode = compose_mode
        self.step_sizes = step_sizes
        self.max_steps = int(max_steps)
        self.target = None if target_image_6ch is None else target_image_6ch.to(self.device).float().clamp(0, 1)
        self.start_extrinsic = None if start_extrinsic is None else start_extrinsic.to(self.device).float()
        self.target_extrinsic = None if target_extrinsic is None else target_extrinsic.to(self.device).float()
        self.model_pts = None if model_pts is None else model_pts.to(self.device).float()
        self.empty_image_threshold = float(empty_image_threshold)
        self.check_empty_image = bool(check_empty_image)
        self.dataset = dataset
        self.dataset_mode = dataset_mode
        self.dataset_cursor = int(dataset_start_index)
        self.fixed_mat_ids = fixed_mat_ids
        self._idpair_to_dsindex: Optional[Dict[Tuple[int, int], int]] = None

        self.observation_space = spaces.Tuple((
            spaces.Box(low=0.0, high=1.0, shape=(6, self.out_size, self.out_size), dtype=np.float32),
            spaces.Box(low=0.0, high=1.0, shape=(6, self.out_size, self.out_size), dtype=np.float32),
        ))
        self.action_space = spaces.Discrete(12)
        self._action_names = ["+x", "-x", "+y", "-y", "+z", "-z", "+rx", "-rx", "+ry", "-ry", "+rz", "-rz"]

        self.extrinsic: Tensor = (self.start_extrinsic.clone() if self.start_extrinsic is not None else torch.eye(4))
        self.curr_img: Optional[Tensor] = None
        self.step_count = 0
        self.score: float = 0.0
        self.episode_log = []
        self.current_pair_idx: Optional[int] = None

        self.reward_mode = reward_mode
        self.reward_scale = float(reward_scale)
        self.success_threshold = float(success_threshold)
        self.success_bonus = float(success_bonus)
        self.success_crossing_bonus = float(success_crossing_bonus)
        self.threshold_focus_coef = float(threshold_focus_coef)
        self.empty_penalty = float(empty_penalty)
        self.mse_small_bonus_threshold = float(mse_small_bonus_threshold)
        self.mse_small_bonus = float(mse_small_bonus)
        self.clip_reward = clip_reward
        self.curr_first_alpha = float(curr_first_alpha)
        self.final_bonus_scale = float(final_bonus_scale)
        self.handle_empty = handle_empty
        self.bad_streak_limit = int(bad_streak_limit)
        self.adaptive_step = bool(adaptive_step)
        self.coarse_trans_mm = float(coarse_trans_mm)
        self.coarse_rot_deg = float(coarse_rot_deg)
        self.fine_trans_mm = float(fine_trans_mm)
        self.fine_rot_deg = float(fine_rot_deg)
        self.fine_threshold = float(fine_threshold)

        self.prev_extrinsic: Optional[Tensor] = None
        self.best_extrinsic: Optional[Tensor] = None
        self.bad_streak: int = 0
        self.last_mse: Optional[float] = None
        self.best_mse: Optional[float] = None
        self.initial_mse: Optional[float] = None

        # Target image cache (performance optimization)
        self.target_img_cached: Optional[Tensor] = None
        self.target_extrinsic_last: Optional[Tensor] = None

        if self.dataset is None:
            if self.start_extrinsic is None or self.target_extrinsic is None:
                raise ValueError("When `dataset` is None, you must provide both `start_extrinsic` and `target_extrinsic`.")

        # ---- NEW: load and cache target binary mask ----
        self.target_mask: Optional[Tensor] = None
        self.target_mask_path = target_mask_path
        self.target_mask_threshold = float(target_mask_threshold)
        self.target_mask_invert = bool(target_mask_invert)

        if self.target_mask_path is not None:
            mask_path = self.target_mask_path
            # Small fallback: if "./largest.n=png" doesn't exist, automatically try "./largest.png"
            if (not os.path.exists(mask_path)) and (mask_path.endswith(".n=png")):
                fallback = mask_path.replace(".n=png", ".png")
                if os.path.exists(fallback):
                    mask_path = fallback

            if os.path.exists(mask_path):
                try:
                    # Read as grayscale and resample to out_size
                    img = Image.open(mask_path).convert("L")
                    img = img.resize((self.out_size, self.out_size), resample=Image.NEAREST)
                    mask_np = np.array(img, dtype=np.float32)
                    mask_np = (mask_np >= self.target_mask_threshold).astype(np.float32)
                    if self.target_mask_invert:
                        mask_np = 1.0 - mask_np
                    self.target_mask = torch.from_numpy(mask_np).to(self.device).unsqueeze(0)  # (1,H,W)
                except Exception as e:
                    print(f"[WARN] Failed to load target mask from '{mask_path}': {e}")
                    self.target_mask = None
            else:
                print(f"[WARN] target_mask_path not found: {self.target_mask_path}")

    def _ensure_idpair_index(self):
        if self._idpair_to_dsindex is not None:
            return
        assert self.dataset is not None, "Dataset must be provided"
        self._idpair_to_dsindex = {}
        for i in range(len(self.dataset)):
            try:
                sample = self.dataset[i]
                ids = sample.get("index_pair", None)
                if ids is not None:
                    s_id = int(ids[0]); t_id = int(ids[1])
                    self._idpair_to_dsindex.setdefault((s_id, t_id), i)
            except Exception:
                continue

    def _action_id_to_xi(self, a_id: int) -> Tensor:
        xi = torch.zeros(6, dtype=torch.float32, device=self.device)

        if self.adaptive_step and (self.last_mse is not None) and np.isfinite(self.last_mse):
            if self.last_mse < self.fine_threshold:
                s_t = self.fine_trans_mm
                s_r = np.deg2rad(self.fine_rot_deg)
            else:
                s_t = self.coarse_trans_mm
                s_r = np.deg2rad(self.coarse_rot_deg)
        else:
            s_t = float(self.step_sizes.trans_mm)
            s_r = float(self.step_sizes.rot_rad)

        if   a_id == 0:  xi[0] = +s_t
        elif a_id == 1:  xi[0] = -s_t
        elif a_id == 2:  xi[1] = +s_t
        elif a_id == 3:  xi[1] = -s_t
        elif a_id == 4:  xi[2] = +s_t
        elif a_id == 5:  xi[2] = -s_t
        elif a_id == 6:  xi[3] = +s_r
        elif a_id == 7:  xi[3] = -s_r
        elif a_id == 8:  xi[4] = +s_r
        elif a_id == 9:  xi[4] = -s_r
        elif a_id == 10: xi[5] = +s_r
        elif a_id == 11: xi[5] = -s_r
        else:
            raise ValueError(f"Invalid action id: {a_id}")
        return xi

    def _apply_delta(self, delta_T: Tensor) -> None:
        if self.compose_mode == "left":
            temp_ex = pytorch3d_to_surgvtk(self.extrinsic)
            temp_dt = pytorch3d_to_surgvtk(delta_T)
            self.extrinsic = vtk_2_PyTorch3D(surgvtk_2_vtk(temp_dt @ temp_ex))
            print("left")
        else:
            temp_ex = pytorch3d_to_surgvtk(self.extrinsic)
            self.extrinsic = vtk_2_PyTorch3D(surgvtk_2_vtk(temp_ex @ delta_T))

    def _render_obs(self) -> Tuple[Tensor, Tensor]:
        """
        Only render current image, target uses cache
        Within the same episode, target is rendered only once
        """
        # Always render current image
        current_img = re_rendering(self.extrinsic, self.renderer, device=self.device, out_size=self.out_size)
    
        # === NEW: apply the same binary mask to current as done for target in reset ===
        if self.target_mask is not None:
            # Safety: resize mask if necessary to align with current
            if self.target_mask.shape[-2:] != current_img.shape[-2:]:
                resized_mask = F.interpolate(
                    self.target_mask.unsqueeze(0),  # (1,1,H,W)
                    size=current_img.shape[-2:],
                    mode="nearest"
                ).squeeze(0)  # (1,H,W)
            else:
                resized_mask = self.target_mask  # (1,H,W)
    
            # (6,H,W) * (1,H,W) broadcast
            current_img = current_img * resized_mask
    
        # Check if need to re-render target
        need_render_target = (
            self.target_img_cached is None or
            self.target_extrinsic_last is None or
            not torch.allclose(self.target_extrinsic, self.target_extrinsic_last, atol=1e-6)
        )
    
        if need_render_target:
            # The rendered output here is "un-augmented/un-masked" target; the actual masked version
            # for output/training will be completed and cached in reset() to self.target_img_cached
            target_img = re_rendering(self.target_extrinsic, self.renderer, device=self.device, out_size=self.out_size)
            self.target_img_cached = target_img.clone()  # Cache immediately, will be overwritten in reset() with augmented+masked version
            self.target_extrinsic_last = self.target_extrinsic.clone()
        else:
            target_img = self.target_img_cached
    
        return current_img.float().clamp(0, 1), target_img.float().clamp(0, 1)

    def _compute_mse(self) -> float:
        if self.model_pts is None:
            raise RuntimeError("`model_pts` is required for MSE reward but is None.")
        if self.target_extrinsic is None or self.start_extrinsic is None:
            raise RuntimeError("Both `target_extrinsic` and `start_extrinsic` are required.")
        mat2 = self.target_extrinsic
        mat2_pred = self.extrinsic  # Current predicted pose
        pose_error = torch.linalg.inv(pytorch3d_to_surgvtk(mat2)) @ pytorch3d_to_surgvtk(mat2_pred)
        pts = self.model_pts  # (N, 3)
        R = pose_error[:3, :3]
        t = pose_error[:3, 3]
        pts_transformed = pts @ R.T + t  # (N, 3)
        mse = torch.mean((pts - pts_transformed) ** 2).item()
        if not np.isfinite(mse):
            mse = float("inf")

        return float(mse)
    
    def _is_empty_mask(self, curr6: Tensor, min_frac: float = 1e-4) -> bool:
        area = (curr6[5] > 0.5).float().sum().item()
        H, W = curr6.shape[-2:]
        return bool(area / (H * W) < min_frac)

    def _pull_pair_from_dataset(self,
                                forced_idx: Optional[int] = None,
                                forced_mat_ids: Optional[Tuple[int, int]] = None):
        assert self.dataset is not None, "Dataset must be provided"

        if forced_mat_ids is not None:
            self._ensure_idpair_index()
            key = (int(forced_mat_ids[0]), int(forced_mat_ids[1]))
            if key not in self._idpair_to_dsindex:
                raise RuntimeError(f"Specified mat id combination not found in dataset: {key}")
            idx = self._idpair_to_dsindex[key]
        elif forced_idx is not None:
            idx = int(forced_idx) % len(self.dataset)
        else:
            if self.dataset_mode == "random":
                idx = np.random.randint(len(self.dataset))
            elif self.dataset_mode == "fixed":
                if self.fixed_mat_ids is None:
                    raise RuntimeError("dataset_mode='fixed' requires fixed_mat_ids=(start_id, target_id)")
                self._ensure_idpair_index()
                key = (int(self.fixed_mat_ids[0]), int(self.fixed_mat_ids[1]))
                if key not in self._idpair_to_dsindex:
                    raise RuntimeError(f"fixed_mat_ids not found in dataset: {key}")
                idx = self._idpair_to_dsindex[key]
            else:
                idx = self.dataset_cursor % len(self.dataset)
                self.dataset_cursor += 1

        sample = self.dataset[idx]
        p3d_pair: Tensor = sample["p3d"].to(self.device).float()
        indices: Tensor = sample.get("index_pair", torch.tensor([-1, -1]))
        return int(idx), indices, p3d_pair[0], p3d_pair[1]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        forced_idx = None
        if options is not None and "pair_idx" in options:
            forced_idx = int(options["pair_idx"])

        sample_idx = -1
        pair_ids = torch.tensor([-1, -1], device=self.device)

        if self.dataset is not None:
            sample_idx, pair_ids, start_extr, target_extr = self._pull_pair_from_dataset(
                forced_idx=forced_idx,
                forced_mat_ids=self.fixed_mat_ids
            )
            self.start_extrinsic = start_extr
            self.target_extrinsic = target_extr

            # Clear target cache as target_extrinsic may have changed
            self.target_img_cached = None
            self.target_extrinsic_last = None

            if forced_idx is not None:
                expected = int(forced_idx) % len(self.dataset)
                if sample_idx != expected:
                    raise RuntimeError(f"Index mismatch! Expected: {expected}, Actual: {sample_idx}")

        elif self.start_extrinsic is None or self.target_extrinsic is None:
            raise RuntimeError("No dataset and no manual start/target provided.")

        self.current_pair_idx = int(sample_idx)
        self.extrinsic = self.start_extrinsic.clone()
        self.step_count = 0
        self.score = 0.0

        # Render initial observation (this will cache un-augmented target)
        curr_img, tgt_img = self._render_obs()

        # ========= Modified: apply augmentation =========
        if hasattr(self, 'augment_target_on_reset') and self.augment_target_on_reset:
            if hasattr(self, 'target_augmenter') and self.target_augmenter is not None:
                tgt_img = self.target_augmenter.augment_target(tgt_img)

        # ========= NEW: multiply with binary mask after augmentation =========
        # tgt_img: (6,H,W) in [0,1]
        if self.target_mask is not None:
            # Ensure size match (usually already resized in init, but safety check)
            if self.target_mask.shape[-2:] != tgt_img.shape[-2:]:
                self.target_mask = F.interpolate(
                    self.target_mask.unsqueeze(0),  # (1,1,H,W)
                    size=tgt_img.shape[-2:],
                    mode="nearest"
                ).squeeze(0)
            # mask (1,H,W) broadcast to (6,H,W)
            tgt_img = tgt_img * self.target_mask

        # Update cache to "final version" of target (augmented+masked)
        self.target_img_cached = tgt_img.clone()

        self.curr_img = (curr_img, tgt_img)
        obs = (curr_img.float().clamp(0, 1), tgt_img.float().clamp(0, 1))

        mse0 = self._compute_mse()
        self.last_mse = mse0
        self.best_mse = mse0
        self.initial_mse = mse0
        self.prev_extrinsic = self.extrinsic.clone()
        self.best_extrinsic = self.extrinsic.clone()
        self.bad_streak = 0

        is_empty = self._is_empty_mask(curr_img)

        self.episode_log = [{
            "step": 0,
            "action_id": None,
            "action_name": "START",
            "mse": mse0,
            "reward": 0.0,
            "is_empty": is_empty,
            "dataset_sample_idx": sample_idx,
            "pair_indices": tuple(map(int, pair_ids.detach().cpu().tolist())),
            "pair_idx": int(sample_idx),
            "step_mode": "initial",
        }]

        info = {
            "mse": mse0,
            "best_mse": mse0,
            "success": bool(mse0 < self.success_threshold) and (not is_empty),
            "score": self.score,
            "is_empty": is_empty,
            "dataset_sample_idx": sample_idx,
            "pair_indices": tuple(map(int, pair_ids.detach().cpu().tolist())),
            "pair_idx": int(sample_idx),
        }

        return obs, info

    @staticmethod
    def _reward_curr_first_positive_raw(prev_mse: float,
                                        curr_mse: float,
                                        m_ref: float,
                                        alpha: float,
                                        eps: float = 1e-8) -> float:
        curr_score = m_ref / (curr_mse + eps)
        improv_score = alpha * float(np.log((prev_mse + eps) / (curr_mse + eps)))
        return curr_score + improv_score

    @staticmethod
    def _reward_curr_first_positive_symmetric(prev_mse: float,
                                              curr_mse: float,
                                              m_ref: float,
                                              alpha: float,
                                              eps: float = 1e-8) -> float:
        if not np.isfinite(prev_mse) or not np.isfinite(curr_mse):
            return -1.0
        if curr_mse < prev_mse:
            return SE3PoseEnvDiscrete._reward_curr_first_positive_raw(prev_mse, curr_mse, m_ref, alpha, eps)
        elif curr_mse > prev_mse:
            pos_if_improved = SE3PoseEnvDiscrete._reward_curr_first_positive_raw(curr_mse, prev_mse, m_ref, alpha, eps)
            return -pos_if_improved
        else:
            return -alpha * 0.01

    def step(self, action_id: int, predicted_step_mode: Optional[float] = None,
             predicted_terminate: Optional[float] = None,
             terminate_confidence: float = 0.7):
        """
        Args:
            action_id: action ID
            predicted_step_mode: model predicted step size mode (0-1, >0.5 indicates fine mode)
            predicted_terminate: model predicted termination probability (0-1)
            terminate_confidence: confidence threshold for termination decision
        """
        self.step_count += 1
        self.prev_extrinsic = self.extrinsic.clone()

        # ========= Use model prediction to decide step size mode =========
        if predicted_step_mode is not None:
            use_fine = (predicted_step_mode > 0.5)
            if use_fine:
                s_t = self.fine_trans_mm
                s_r = np.deg2rad(self.fine_rot_deg)
                step_mode = "fine"
            else:
                s_t = self.coarse_trans_mm
                s_r = np.deg2rad(self.coarse_rot_deg)
                step_mode = "coarse"
            gt_step_mode = 1.0 if use_fine else 0.0
        elif self.adaptive_step and (self.last_mse is not None) and np.isfinite(self.last_mse):
            # Fallback: use original hardcoded logic
            step_mode = "fine" if self.last_mse < self.fine_threshold else "coarse"
            gt_step_mode = 1.0 if self.last_mse < self.fine_threshold * 2 else 0.0
            if step_mode == "fine":
                s_t = self.fine_trans_mm
                s_r = np.deg2rad(self.fine_rot_deg)
            else:
                s_t = self.coarse_trans_mm
                s_r = np.deg2rad(self.coarse_rot_deg)
        else:
            s_t = float(self.step_sizes.trans_mm)
            s_r = float(self.step_sizes.rot_rad)
            step_mode = "default"
            gt_step_mode = 0.0

        # Construct action vector
        xi = torch.zeros(6, dtype=torch.float32, device=self.device)
        if   action_id == 0:  xi[0] = +s_t
        elif action_id == 1:  xi[0] = -s_t
        elif action_id == 2:  xi[1] = +s_t
        elif action_id == 3:  xi[1] = -s_t
        elif action_id == 4:  xi[2] = +s_t
        elif action_id == 5:  xi[2] = -s_t
        elif action_id == 6:  xi[3] = +s_r
        elif action_id == 7:  xi[3] = -s_r
        elif action_id == 8:  xi[4] = +s_r
        elif action_id == 9:  xi[4] = -s_r
        elif action_id == 10: xi[5] = +s_r
        elif action_id == 11: xi[5] = -s_r
        else:
            raise ValueError(f"Invalid action id: {action_id}")

        delta = se3_exp(xi)
        self._apply_delta(delta)

        # Render observation (target uses cache, only render current)
        curr_img, tgt_img = self._render_obs()
        self.curr_img = (curr_img, tgt_img)

        is_empty = self._is_empty_mask(curr_img)
        terminated = False
        truncated = (self.step_count >= self.max_steps)

        if is_empty:
            if self.handle_empty == "terminate":
                reward = self.empty_penalty
                mse = float('inf')
                terminated = True

            elif self.handle_empty == "undo":
                self.extrinsic = self.prev_extrinsic.clone()
                reward = min(-1.0, self.empty_penalty * 0.2)
                mse = float('inf')
                terminated = False
                curr_img, tgt_img = self._render_obs()
                self.curr_img = (curr_img, tgt_img)

            elif self.handle_empty == "rollback":
                if self.best_extrinsic is not None:
                    self.extrinsic = self.best_extrinsic.clone()
                    curr_img, tgt_img = self._render_obs()
                    self.curr_img = (curr_img, tgt_img)
                reward = self.empty_penalty
                mse = float('inf')
                terminated = True

            else:
                reward = self.empty_penalty
                mse = float('inf')
                terminated = True

            mse_bonus = 0.0
            delta_mse = 0.0
            gt_terminate = 0.0

        else:
            mse = self._compute_mse()
            delta_mse = (self.last_mse - mse) if (self.last_mse is not None and np.isfinite(self.last_mse)) else 0.0

            # ========= Generate ground truth termination label =========
            should_terminate = False
            if mse < self.success_threshold:
                should_terminate = True
            elif mse < self.success_threshold * 5:
                should_terminate = True

            gt_terminate = 1.0 if should_terminate else 0.0

            if (self.last_mse is not None) and np.isfinite(self.last_mse):
                if mse >= self.last_mse:
                    self.bad_streak += 1
                else:
                    self.bad_streak = 0
            else:
                self.bad_streak = 0

            if (self.best_mse is None) or (mse < self.best_mse):
                self.best_mse = mse
                self.best_extrinsic = self.extrinsic.clone()

            if (self.bad_streak_limit > 0) and (self.bad_streak >= self.bad_streak_limit):
                if self.best_extrinsic is not None:
                    self.extrinsic = self.best_extrinsic.clone()
                    curr_img, tgt_img = self._render_obs()
                    self.curr_img = (curr_img, tgt_img)
                reward = self.empty_penalty * 0.5
                terminated = True

            else:
                base_reward = 0.0
                eps = 1e-6
                c = 1.0

                if self.reward_mode == "improvement":
                    base_reward = delta_mse

                elif self.reward_mode == "neg_mse":
                    base_reward = -mse

                elif self.reward_mode == "improvement_ratio":
                    if (self.last_mse is not None) and np.isfinite(self.last_mse) and (self.last_mse > eps) and np.isfinite(mse):
                        base_reward = (self.last_mse - mse) / max(self.last_mse, eps)
                    else:
                        base_reward = 0.0

                elif self.reward_mode == "improvement_log":
                    if (self.last_mse is not None) and np.isfinite(self.last_mse) and np.isfinite(mse):
                        base_reward = float(np.log(self.last_mse + c) - np.log(mse + c))
                    else:
                        base_reward = 0.0

                elif self.reward_mode == "progress_log_to_thresh":
                    th = max(self.success_threshold, eps)
                    if (self.last_mse is not None) and np.isfinite(self.last_mse) and np.isfinite(mse):
                        prev_term = float(np.log(max(self.last_mse, th)))
                        curr_term = float(np.log(max(mse, th)))
                        base_reward = prev_term - curr_term
                    else:
                        base_reward = 0.0

                elif self.reward_mode == "log_progress_with_threshold":
                    th = max(self.success_threshold, eps)
                    if (self.last_mse is not None) and np.isfinite(self.last_mse) and np.isfinite(mse):
                        log_improve = float(np.log(self.last_mse + c) - np.log(mse + c))
                        prev_term = float(np.log(max(self.last_mse, th)))
                        curr_term = float(np.log(max(mse, th)))
                        toward_thresh = max(prev_term - curr_term, 0.0)
                    else:
                        log_improve = 0.0
                        toward_thresh = 0.0
                    base_reward = log_improve + self.threshold_focus_coef * toward_thresh

                elif self.reward_mode == "curr_first_positive":
                    prev = self.last_mse if (self.last_mse is not None and np.isfinite(self.last_mse)) else mse
                    base_reward = self._reward_curr_first_positive_symmetric(
                        prev_mse=prev,
                        curr_mse=mse,
                        m_ref=self.success_threshold,
                        alpha=self.curr_first_alpha,
                    )

                elif self.reward_mode == "curr_delta":
                    if (self.last_mse is not None) and np.isfinite(self.last_mse) and np.isfinite(mse):
                        base_reward = (self.last_mse - mse) / max(self.last_mse, eps)
                    else:
                        base_reward = 0.0

                elif self.reward_mode == "simple_step_with_final_bonus":
                    if (self.last_mse is not None) and np.isfinite(self.last_mse) and np.isfinite(mse):
                        if mse < self.last_mse:
                            base_reward = 1.0
                        elif mse > self.last_mse:
                            base_reward = -1.1
                        else:
                            base_reward = 0.0
                    else:
                        base_reward = 0.0

                else:
                    raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

                reward = self.reward_scale * float(base_reward)

                mse_bonus = 0.0
                if np.isfinite(mse) and mse <= self.mse_small_bonus_threshold:
                    reward += self.mse_small_bonus
                    mse_bonus = self.mse_small_bonus

                reached = (mse < self.success_threshold)
                crossed = reached and ((self.last_mse is None) or (self.last_mse >= self.success_threshold))
                if reached:
                    reward += self.success_bonus
                    if crossed:
                        reward += self.success_crossing_bonus
                    terminated = True

                if (terminated or truncated) and self.reward_mode == "simple_step_with_final_bonus":
                    if (self.initial_mse is not None) and np.isfinite(self.initial_mse) and np.isfinite(mse):
                        improvement_ratio = (self.initial_mse - mse) / max(self.initial_mse, 1e-6)
                        if improvement_ratio > 0:
                            final_bonus = self.final_bonus_scale * improvement_ratio
                            reward += final_bonus

                if self.clip_reward is not None:
                    reward = float(np.clip(reward, self.clip_reward[0], self.clip_reward[1]))

                self.last_mse = mse
                if self.best_mse is None or mse < self.best_mse:
                    self.best_mse = mse

        # ========= Use model prediction to decide whether to terminate =========
        model_wants_terminate = (
            predicted_terminate is not None and
            predicted_terminate > terminate_confidence
        )

        obs = (curr_img.float().clamp(0, 1), tgt_img.float().clamp(0, 1))
        self.score += float(reward)

        if is_empty and self.handle_empty in ("terminate", "rollback"):
            terminated_reason = "empty_image"
        elif (self.bad_streak_limit > 0) and (self.bad_streak >= self.bad_streak_limit):
            terminated_reason = "rollback"
        elif model_wants_terminate and not terminated:
            # Model decides to terminate
            terminated = True
            terminated_reason = "model_decision"
        elif (not is_empty) and (mse < self.success_threshold):
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
            "success": bool(not is_empty and np.isfinite(mse) and mse < self.success_threshold),
            "score": self.score,
            "is_empty": is_empty,
            "terminated_reason": terminated_reason,
            "pair_idx": self.current_pair_idx,
            "step_mode": step_mode,
            "gt_step_mode": float(gt_step_mode),
            "gt_terminate": float(gt_terminate),
            "model_terminate_pred": float(predicted_terminate) if predicted_terminate is not None else 0.0,
            "model_step_mode_pred": float(predicted_step_mode) if predicted_step_mode is not None else 0.0,
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
            "gt_step_mode": float(gt_step_mode),
            "gt_terminate": float(gt_terminate),
            "model_terminate_pred": float(predicted_terminate) if predicted_terminate is not None else 0.0,
            "model_step_mode_pred": float(predicted_step_mode) if predicted_step_mode is not None else 0.0,
        })

        return obs, float(reward), terminated, truncated, info



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
        "liver":        {"file": "./data/upper_new.vtk",  "color": [1.0, 0.8, 0.3], "opacity": 1.0},
        "left_ridge":   {"file": "./data/left_ridge.vtk",   "color": [0.7, 0.7, 0.7], "opacity": 1.0},
        "ligament":     {"file": "./data/ligament.vtk",     "color": [0.9, 0.6, 0.6], "opacity": 1.0},
        "right_ridge":  {"file": "./data/right_ridge.vtk",  "color": [0.6, 0.6, 0.9], "opacity": 1.0},
        "bottom_liver": {"file": "./data/bottom_new.vtk", "color": [0.8, 0.8, 0.8], "opacity": 1.0},
    }



def get_default_camera_params():
    return {
        'W': 1920, 'H': 1080,
        'fx': 1279.8129504, 'fy': 1279.8129504,
        'cx': 955.172960, 'cy': 499.166189
    }
