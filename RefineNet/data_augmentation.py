import cv2
import numpy as np
import random
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage import morphology

# 你的标签颜色字典
LABEL_COLORS = {
    'red'   : (0,   0, 255),
    'green' : (0, 255,   0),
    'blue'  : (255, 0,   0),
    'white' : (255, 255, 255)
}

def elastic_deformation(image, alpha, sigma):
    """弹性形变"""
    shape = image.shape[:2]
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="reflect") * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="reflect") * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    deformed = np.zeros_like(image)
    for c in range(image.shape[2]):
        deformed[..., c] = map_coordinates(image[..., c], indices,
                                           order=1, mode='reflect').reshape(shape)
    return deformed

def random_occlusion(image, max_boxes=3, box_ratio=0.2):
    """随机擦除若干矩形区域"""
    h, w = image.shape[:2]
    for _ in range(random.randint(0, max_boxes)):
        ew, eh = int(w * box_ratio), int(h * box_ratio)
        x1 = random.randint(0, w - ew)
        y1 = random.randint(0, h - eh)
        image[y1:y1+eh, x1:x1+ew] = 0
    return image

def augment_contour_map(a,
                        skeleton_dilate_iter=1,
                        occlusion_boxes=3,
                        occlusion_ratio=0.2,
                        elastic_alpha=10,
                        elastic_sigma=4):
    H, W = a.shape[:2]
    # 1) per-label skeleton + dilation
    skel_canvas = np.zeros_like(a)
    for name, bgr in LABEL_COLORS.items():
        # 提取当前颜色的掩码
        mask = cv2.inRange(a, np.array(bgr), np.array(bgr))
        # Skeletonize
        bin_mask = mask > 0
        skel_bool = morphology.skeletonize(bin_mask)
        skel = np.zeros((H, W), dtype=np.uint8)
        skel[skel_bool] = 255
        # 膨胀骨架
        ker = cv2.getStructuringElement(cv2.MORPH_CROSS, (15, 15))
        skel = cv2.dilate(skel, ker, iterations=skeleton_dilate_iter)
        # 还原成彩色骨架
        color_layer = np.zeros_like(a)
        color_layer[skel > 0] = bgr
        skel_canvas = cv2.bitwise_or(skel_canvas, color_layer)

    # 2) 随机擦除
    occluded = random_occlusion(skel_canvas.copy(),
                                max_boxes=occlusion_boxes,
                                box_ratio=occlusion_ratio)

    # 3) 最后弹性形变
    result = elastic_deformation(occluded,
                                 alpha=elastic_alpha,
                                 sigma=elastic_sigma)

    result = np.where(result > 128, 255, 0).astype(np.uint8)

    return result
