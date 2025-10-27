import cv2, cv2.ximgproc as xip
import numpy as np, random
from scipy.ndimage import gaussian_filter

LABEL_COLORS = {
    'red'   : (0,   0, 255),
    'green' : (0, 255,   0),
    'blue'  : (255, 0,   0),
    'white' : (255, 255, 255)
}

def fast_skeleton(mask, iterations=1):
    """
    mask: uint8 二值图
    iterations: 膨胀次数，>=0
    """
    # 1) 薄化
    skel = xip.thinning(mask, thinningType=xip.THINNING_ZHANGSUEN)

    # 2) 膨胀
    if iterations > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 1))
        skel = cv2.dilate(skel, ker, iterations=iterations)

    return skel


def random_occlusion(image, max_boxes=3, box_ratio=0.2):
    h, w = image.shape[:2]
    for _ in range(random.randint(0, max_boxes)):
        ew, eh = int(w*box_ratio), int(h*box_ratio)
        x1 = random.randint(0, w-ew); y1 = random.randint(0, h-eh)
        image[y1:y1+eh, x1:x1+ew] = 0
    return image

def elastic_deformation_fast(img, alpha, sigma):
    H, W = img.shape[:2]
    dx = gaussian_filter((np.random.rand(H,W)*2-1), sigma, mode='reflect') * alpha
    dy = gaussian_filter((np.random.rand(H,W)*2-1), sigma, mode='reflect') * alpha
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    return cv2.remap(img, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)

def augment_contour_map_fast(a,
        skeleton_dilate_iter=None,
        occlusion_boxes=3,
        occlusion_ratio=0.2,
        elastic_alpha=10,
        elastic_sigma=4):
    H, W = a.shape[:2]

    # 在这里动态选择随机值
    if skeleton_dilate_iter is None:
        skeleton_dilate_iter = random.randint(1, 5)
    skel_canvas = np.zeros_like(a)
    for bgr in LABEL_COLORS.values():
        mask = cv2.inRange(a, np.array(bgr), np.array(bgr))
        skel = fast_skeleton(mask, iterations=skeleton_dilate_iter)
        # 上色
        layer = np.zeros_like(a)
        layer[skel>0] = bgr
        cv2.bitwise_or(skel_canvas, layer, skel_canvas)
    # occlusion
    occ = random_occlusion(skel_canvas.copy(),
                           max_boxes=occlusion_boxes,
                           box_ratio=occlusion_ratio)
    # elastic
    out = elastic_deformation_fast(occ, alpha=elastic_alpha,
                                   sigma=elastic_sigma)
    return np.where(out>128,255,0).astype(np.uint8)
