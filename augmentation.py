import torch
import numpy as np
import cv2
from typing import Optional, Literal
from scipy.ndimage import gaussian_filter


class TargetImageAugmenter:
    """
    Augments target images (6-channel format) for SE3 pose estimation training.
    Applies augmentation to semantic channels (0-3), depth channel (4), and mask channel (5).
    """
    
    def __init__(
        self,
        # Contour augmentation
        apply_contour_aug: bool = True,
        contour_aug_prob: float = 0.5,
        skeleton_dilate_range: tuple = (1, 5),
        occlusion_boxes: int = 3,
        occlusion_ratio: float = 0.2,
        elastic_alpha: float = 10.0,
        elastic_sigma: float = 4.0,
        
        # Depth augmentation
        apply_depth_aug: bool = True,
        depth_aug_prob: float = 1.0,
        occluder_probability: float = 1.0,
        random_erasing_probability: float = 0.8,
        depth_normalization_probability: float = 0.6,
        scale_perturbation_probability: float = 0.6,
        
        # Rectangle occluder for depth
        length_range: tuple = (60, 128),
        width_range: tuple = (8, 12),
        angle_range: tuple = (-45, 45),
        
        # Depth normalization
        norm_min_range: tuple = (0.0, 0.2),
        norm_max_range: tuple = (0.8, 1.0),
        
        # Scale perturbation
        scale_factor_range: tuple = (0.7, 1.3),
        depth_shift_range: tuple = (-0.3, 0.3),
        noise_std_range: tuple = (0.01, 0.05),
        
        # Mask augmentation
        apply_mask_aug: bool = True,
        mask_aug_prob: float = 0.5,
        mask_dilate_prob: float = 0.5,  # Probability of dilation vs erosion
        mask_kernel_range: tuple = (1, 3),  # Kernel size range (odd numbers)
        mask_iterations_range: tuple = (1, 1),  # Number of iterations
        
        device: str = "cuda"
    ):
        self.apply_contour_aug = apply_contour_aug
        self.contour_aug_prob = contour_aug_prob
        self.skeleton_dilate_range = skeleton_dilate_range
        self.occlusion_boxes = occlusion_boxes
        self.occlusion_ratio = occlusion_ratio
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        
        self.apply_depth_aug = apply_depth_aug
        self.depth_aug_prob = depth_aug_prob
        self.occluder_prob = occluder_probability
        self.random_erasing_prob = random_erasing_probability
        self.depth_norm_prob = depth_normalization_probability
        self.scale_pert_prob = scale_perturbation_probability
        
        self.length_range = length_range
        self.width_range = width_range
        self.angle_range = angle_range
        
        self.norm_min_range = norm_min_range
        self.norm_max_range = norm_max_range
        self.scale_factor_range = scale_factor_range
        self.depth_shift_range = depth_shift_range
        self.noise_std_range = noise_std_range
        
        self.apply_mask_aug = apply_mask_aug
        self.mask_aug_prob = mask_aug_prob
        self.mask_dilate_prob = mask_dilate_prob
        self.mask_kernel_range = mask_kernel_range
        self.mask_iterations_range = mask_iterations_range
        
        self.device = torch.device(device)
    
    def augment_target(self, target_6ch: torch.Tensor) -> torch.Tensor:
        """
        Augment a 6-channel target image.
        
        Args:
            target_6ch: (6, H, W) tensor with channels:
                [0-3]: Semantic (B/ligament, G/right_ridge, R/left_ridge, Edge)
                [4]: Inverse depth
                [5]: Liver mask
        
        Returns:
            Augmented (6, H, W) tensor
        """
        result = target_6ch.clone()
        
        # Convert to numpy for augmentation
        semantic = (result[:4] * 255).cpu().numpy().astype(np.uint8)  # (4, H, W)
        depth = (result[4] * 255).cpu().numpy().astype(np.uint8)  # (H, W)
        mask = (result[5] * 255).cpu().numpy().astype(np.uint8)  # (H, W)
        
        # Augment semantic channels (contours)
        if self.apply_contour_aug and np.random.rand() < self.contour_aug_prob:
            semantic = self._augment_semantic(semantic)
        
        # Augment depth channel
        if self.apply_depth_aug and np.random.rand() < self.depth_aug_prob:
            depth = self._augment_depth(depth, mask)
        
        # Augment mask channel
        if self.apply_mask_aug and np.random.rand() < self.mask_aug_prob:
            mask = self._augment_mask(mask)
        
        # Convert back to tensor
        semantic_t = torch.from_numpy(semantic).float().to(self.device) / 255.0
        depth_t = torch.from_numpy(depth).float().to(self.device) / 255.0
        mask_t = torch.from_numpy(mask).float().to(self.device) / 255.0
        
        result[:4] = semantic_t
        result[4] = depth_t
        result[5] = mask_t
        
        return result.clamp(0, 1)
    
    def _augment_semantic(self, semantic: np.ndarray) -> np.ndarray:
        """
        Augment semantic channels (contours).
        
        Args:
            semantic: (4, H, W) uint8 array
        
        Returns:
            Augmented (4, H, W) uint8 array
        """
        # Convert channels to BGR image for processing
        H, W = semantic.shape[1:]
        bgr_image = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Map channels: B=ligament, G=right_ridge, R=left_ridge
        bgr_image[:, :, 0] = semantic[0]  # B
        bgr_image[:, :, 1] = semantic[1]  # G
        bgr_image[:, :, 2] = semantic[2]  # R
        
        # Apply skeleton + elastic transform
        iterations = np.random.randint(self.skeleton_dilate_range[0], 
                                      self.skeleton_dilate_range[1] + 1)
        
        augmented = self._augment_contour_fast(
            bgr_image,
            skeleton_dilate_iter=iterations,
            occlusion_boxes=self.occlusion_boxes,
            occlusion_ratio=self.occlusion_ratio,
            elastic_alpha=self.elastic_alpha,
            elastic_sigma=self.elastic_sigma
        )
        
        # Convert back to channel format
        result = np.zeros_like(semantic)
        result[0] = augmented[:, :, 0]  # B
        result[1] = augmented[:, :, 1]  # G
        result[2] = augmented[:, :, 2]  # R
        result[3] = semantic[3]  # Keep edge channel unchanged
        
        return result
    
    def _augment_contour_fast(self, img, skeleton_dilate_iter, 
                             occlusion_boxes, occlusion_ratio,
                             elastic_alpha, elastic_sigma):
        """Fast contour augmentation from data_augmentation_fast.py"""
        try:
            import cv2.ximgproc as xip
        except ImportError:
            return img  # Skip if opencv-contrib not available
        
        H, W = img.shape[:2]
        skel_canvas = np.zeros_like(img)
        
        # Process each color channel
        colors = {
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'red': (0, 0, 255)
        }
        
        for color_bgr in colors.values():
            mask = cv2.inRange(img, np.array(color_bgr), np.array(color_bgr))
            if mask.sum() == 0:
                continue
            
            # Skeleton
            skel = xip.thinning(mask, thinningType=xip.THINNING_ZHANGSUEN)
            
            # Dilate
            if skeleton_dilate_iter > 0:
                ker = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
                skel = cv2.dilate(skel, ker, iterations=skeleton_dilate_iter)
            
            # Color the skeleton
            layer = np.zeros_like(img)
            layer[skel > 0] = color_bgr
            cv2.bitwise_or(skel_canvas, layer, skel_canvas)
        
        # Random occlusion
        for _ in range(np.random.randint(0, occlusion_boxes + 1)):
            ew = int(W * occlusion_ratio)
            eh = int(H * occlusion_ratio)
            if ew > 0 and eh > 0 and W > ew and H > eh:
                x1 = np.random.randint(0, W - ew)
                y1 = np.random.randint(0, H - eh)
                skel_canvas[y1:y1+eh, x1:x1+ew] = 0
        
        # Elastic deformation
        dx = gaussian_filter((np.random.rand(H, W) * 2 - 1), elastic_sigma) * elastic_alpha
        dy = gaussian_filter((np.random.rand(H, W) * 2 - 1), elastic_sigma) * elastic_alpha
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        
        result = cv2.remap(skel_canvas, map_x, map_y,
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)
        
        return np.where(result > 128, 255, 0).astype(np.uint8)
    
    def _augment_depth(self, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Augment depth channel.
        
        Args:
            depth: (H, W) uint8 array
            mask: (H, W) uint8 array (liver mask)
        
        Returns:
            Augmented (H, W) uint8 array
        """
        result = depth.copy()
        
        # Rectangle occlusion
        if np.random.rand() < self.occluder_prob:
            result = self._apply_rectangle_occluder(result)
        
        # Random erasing
        if np.random.rand() < self.random_erasing_prob:
            result = self._apply_random_erasing(result)
        
        # Depth normalization
        if np.random.rand() < self.depth_norm_prob:
            result = self._apply_depth_normalization(result)
        
        # Scale perturbation
        if np.random.rand() < self.scale_pert_prob:
            result = self._apply_scale_perturbation(result)
        
        return result
    
    def _augment_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Augment mask channel with random dilation or erosion.
        
        Args:
            mask: (H, W) uint8 array (binary mask)
        
        Returns:
            Augmented (H, W) uint8 array
        """
        if mask.sum() == 0:
            return mask
        
        result = mask.copy()
        
        # Random kernel size (must be odd)
        kernel_size = np.random.randint(self.mask_kernel_range[0], 
                                       self.mask_kernel_range[1] + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                          (kernel_size, kernel_size))
        
        # Random number of iterations
        iterations = np.random.randint(self.mask_iterations_range[0],
                                      self.mask_iterations_range[1] + 1)
        
        # Randomly choose dilation or erosion
        if np.random.rand() < self.mask_dilate_prob:
            result = cv2.dilate(result, kernel, iterations=iterations)
        else:
            result = cv2.erode(result, kernel, iterations=iterations)
        
        return result
    
    def _apply_rectangle_occluder(self, depth: np.ndarray) -> np.ndarray:
        """Apply rotated rectangle occlusion"""
        H, W = depth.shape
        result = depth.copy()
        
        num_rects = np.random.randint(1, 3)
        for _ in range(num_rects):
            length = np.random.randint(self.length_range[0], self.length_range[1])
            width = np.random.randint(self.width_range[0], self.width_range[1])
            angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
            
            cx = np.random.randint(width, W - width)
            cy = np.random.randint(width, H - width)
            
            # Create rotated rectangle
            rect = ((cx, cy), (length, width), angle)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            cv2.fillPoly(result, [box], 0)
        
        return result
    
    def _apply_random_erasing(self, depth: np.ndarray) -> np.ndarray:
        """Random rectangular erasing"""
        H, W = depth.shape
        result = depth.copy()
        
        num_patches = np.random.randint(0, 3)
        for _ in range(num_patches):
            pw = np.random.randint(W // 20, W // 8)
            ph = np.random.randint(H // 20, H // 8)
            if pw > 0 and ph > 0 and W > pw and H > ph:
                x = np.random.randint(0, W - pw)
                y = np.random.randint(0, H - ph)
                result[y:y+ph, x:x+pw] = 0
        
        return result
    
    def _apply_depth_normalization(self, depth: np.ndarray) -> np.ndarray:
        """Normalize depth to random range"""
        result = depth.copy().astype(np.float32)
        valid_mask = result > 0
        
        if not np.any(valid_mask):
            return result.astype(np.uint8)
        
        valid_depths = result[valid_mask]
        d_min, d_max = valid_depths.min(), valid_depths.max()
        
        if d_max == d_min:
            return result.astype(np.uint8)
        
        # Random target range
        t_min = np.random.uniform(self.norm_min_range[0], self.norm_min_range[1])
        t_max = np.random.uniform(self.norm_max_range[0], self.norm_max_range[1])
        t_max = max(t_max, t_min + 0.1)
        
        # Normalize
        normalized = (result - d_min) / (d_max - d_min)
        normalized = normalized * (t_max - t_min) + t_min
        result = normalized * 255
        result[~valid_mask] = 0
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _apply_scale_perturbation(self, depth: np.ndarray) -> np.ndarray:
        """Apply scale and noise perturbation"""
        result = depth.copy().astype(np.float32)
        valid_mask = result > 0
        
        if not np.any(valid_mask):
            return result.astype(np.uint8)
        
        scale = np.random.uniform(self.scale_factor_range[0], self.scale_factor_range[1])
        shift = np.random.uniform(self.depth_shift_range[0], self.depth_shift_range[1]) * 255
        noise_std = np.random.uniform(self.noise_std_range[0], self.noise_std_range[1]) * 255
        
        result[valid_mask] = result[valid_mask] * scale + shift
        noise = np.random.normal(0, noise_std, result.shape)
        result[valid_mask] += noise[valid_mask]
        
        result = np.clip(result, 0, 255)
        result[~valid_mask] = 0
        
        return result.astype(np.uint8)


# Integration with SE3PoseEnvDiscrete
def integrate_augmentation_into_env(env_class):
    """
    Add target augmentation capability to SE3PoseEnvDiscrete.
    
    Usage:
        env = SE3PoseEnvDiscrete(...)
        env.target_augmenter = TargetImageAugmenter(device=env.device)
        env.augment_target_on_reset = True
    """
    
    original_reset = env_class.reset
    
    def augmented_reset(self, *, seed=None, options=None):
        obs, info = original_reset(self, seed=seed, options=options)
        
        # Apply augmentation if enabled
        if hasattr(self, 'augment_target_on_reset') and self.augment_target_on_reset:
            if hasattr(self, 'target_augmenter') and self.target_augmenter is not None:
                curr_img, tgt_img = obs
                augmented_target = self.target_augmenter.augment_target(tgt_img)
                obs = (curr_img, augmented_target)
                self.curr_img = (curr_img, augmented_target)
        
        return obs, info
    
    env_class.reset = augmented_reset
    return env_class