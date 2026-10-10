"""Low-cost current-frame quality predictors (8-bit grayscale inputs)."""
import cv2
import numpy as np


def frame_metrics(gray: np.ndarray, grid=(8, 6), fast_threshold: int = 20) -> dict:
    if gray.ndim != 2 or gray.dtype != np.uint8 or not all(gray.shape):
        raise ValueError("Expected nonempty uint8 grayscale image (H, W)")
    height, width = gray.shape
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(dx, dy)
    usable = (gray > 5) & (gray < 250) & (magnitude > 20.0)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    fast = cv2.FastFeatureDetector_create(threshold=fast_threshold, nonmaxSuppression=True)
    points = fast.detect(gray)
    gx, gy = grid
    if gx < 1 or gy < 1:
        raise ValueError("Grid dimensions must be positive")
    occupied = {
        (min(gx-1, int(kp.pt[0] * gx / width)),
         min(gy-1, int(kp.pt[1] * gy / height)))
        for kp in points
    }
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=1000, qualityLevel=0.01, minDistance=7)
    return {
        "grad_p90": float(np.percentile(magnitude, 90)),
        "usable_gradient_ratio": float(np.mean(usable)),
        "laplacian_variance": float(np.var(lap)),
        "dark_ratio": float(np.mean(gray <= 5)),
        "saturated_ratio": float(np.mean(gray >= 250)),
        "fast_count": len(points),
        "fast_grid_occupancy": len(occupied) / (gx * gy),
        "shi_tomasi_count_capped": 0 if corners is None else len(corners),
    }
