"""Next-frame KLT tracking labels, kept separate from image predictors."""
import cv2
import numpy as np


def track_pair(gray0: np.ndarray, gray1: np.ndarray, fb_threshold: float = 1.0) -> dict:
    if gray0.shape != gray1.shape or gray0.ndim != 2 or gray0.dtype != np.uint8 or gray1.dtype != np.uint8:
        raise ValueError("Expected same-sized uint8 grayscale images")
    if fb_threshold < 0:
        raise ValueError("fb_threshold must be nonnegative")
    pts0 = cv2.goodFeaturesToTrack(gray0, maxCorners=1000, qualityLevel=0.01, minDistance=7)
    n0 = 0 if pts0 is None else len(pts0)
    empty = {
        "klt_initial_count": n0,
        "klt_fb_good_count": 0,
        "klt_fb_success_ratio": float("nan") if n0 == 0 else 0.0,
        "klt_good_per_1000px": 0.0,
        "klt_fb_median_error_px": float("nan"),
    }
    if n0 == 0:
        return empty
    opts = dict(winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    pts1, forward, _ = cv2.calcOpticalFlowPyrLK(gray0, gray1, pts0, None, **opts)
    if pts1 is None or forward is None:
        return empty
    # Invalid LK endpoints are replaced before backward LK to keep OpenCV stable.
    fwd = pts1.reshape(-1, 2).copy()
    forward_ok = forward.ravel() == 1
    finite_fwd = np.isfinite(fwd).all(axis=1)
    safe_fwd = fwd.copy()
    safe_fwd[~finite_fwd] = pts0.reshape(-1, 2)[~finite_fwd]
    returned, backward, _ = cv2.calcOpticalFlowPyrLK(
        gray1, gray0, safe_fwd.reshape(-1, 1, 2), None, **opts
    )
    if returned is None or backward is None:
        return empty
    back = returned.reshape(-1, 2)
    orig = pts0.reshape(-1, 2)
    h, w = gray0.shape
    finite = finite_fwd & np.isfinite(back).all(axis=1)
    bounds = (fwd[:, 0] >= 0) & (fwd[:, 0] < w) & (fwd[:, 1] >= 0) & (fwd[:, 1] < h)
    valid = forward_ok & (backward.ravel() == 1) & finite & bounds
    errors = np.linalg.norm(orig - back, axis=1)
    passed = valid & (errors <= fb_threshold)
    good = int(np.sum(passed))
    return {
        "klt_initial_count": n0,
        "klt_fb_good_count": good,
        "klt_fb_success_ratio": good / n0,
        "klt_good_per_1000px": 1000.0 * good / gray0.size,
        "klt_fb_median_error_px": float(np.median(errors[valid])) if np.any(valid) else float("nan"),
    }
