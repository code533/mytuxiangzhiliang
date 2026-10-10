"""Frame-only quality predictors. Invalid pixels are excluded using a fixed ROI."""
import cv2
import numpy as np


def _mask_array(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if image.ndim != 2 or image.dtype != np.uint8 or not all(image.shape):
        raise ValueError("Expected nonempty uint8 grayscale image (H, W)")
    if mask is None:
        return np.ones(image.shape, dtype=bool)
    if mask.shape != image.shape:
        raise ValueError("Mask shape differs from input image")
    valid = np.asarray(mask, dtype=bool)
    if not valid.any():
        raise ValueError("Mask excludes the entire image")
    return valid


def frame_metrics(gray: np.ndarray, grid=(8, 6), fast_threshold: int = 20,
                  mask: np.ndarray | None = None) -> dict:
    valid = _mask_array(gray, mask)
    h, w = gray.shape
    gx, gy = grid
    if gx < 1 or gy < 1:
        raise ValueError("Grid dimensions must be positive")
    # Exclude mask boundaries from derivative features to prevent artificial edges.
    inside = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    if not inside.any():
        raise ValueError("Valid ROI is too small after 5x5 erosion")
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(dx, dy)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    mask8 = (inside.astype(np.uint8) * 255)
    fast = cv2.FastFeatureDetector_create(threshold=fast_threshold, nonmaxSuppression=True)
    points = fast.detect(gray, mask8)
    occupied = {(min(gx-1, int(k.pt[0]*gx/w)), min(gy-1, int(k.pt[1]*gy/h))) for k in points}
    ys, xs = np.nonzero(inside)
    valid_cells_count = np.unique(np.minimum(gy-1, ys * gy // h) * gx
                                  + np.minimum(gx-1, xs * gx // w)).size
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=1000, qualityLevel=0.01,
                                      minDistance=7, mask=mask8)
    return {
        "grad_p90": float(np.percentile(magnitude[inside], 90)),
        "usable_gradient_ratio": float(np.mean(((gray > 5) & (gray < 250) & (magnitude > 20))[inside])),
        "laplacian_variance": float(np.var(lap[inside])),
        "dark_ratio": float(np.mean((gray <= 5)[inside])),
        "saturated_ratio": float(np.mean((gray >= 250)[inside])),
        "fast_count": len(points),
        "fast_grid_occupancy": len(occupied) / valid_cells_count,
        "shi_tomasi_count_capped": 0 if corners is None else len(corners),
    }
