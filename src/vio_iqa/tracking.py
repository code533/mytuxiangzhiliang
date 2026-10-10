"""Two-frame KLT labels; mask applies to detections and tracked endpoints."""
import cv2
import numpy as np
from vio_iqa.metrics import _mask_array


def track_pair(gray0: np.ndarray, gray1: np.ndarray,
               fb_threshold: float = 1.0, mask: np.ndarray | None = None) -> dict:
    if gray0.shape != gray1.shape or gray0.dtype != np.uint8 or gray1.dtype != np.uint8:
        raise ValueError("Expected same-sized uint8 grayscale images")
    if fb_threshold < 0:
        raise ValueError("fb_threshold must be nonnegative")
    valid_roi = _mask_array(gray0, mask)
    # Keep a safety margin away from invalid pixels when calculating optical flow.
    inside = cv2.erode(valid_roi.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if not inside.any():
        raise ValueError("Valid ROI is too small after erosion")
    pts0 = cv2.goodFeaturesToTrack(gray0, maxCorners=1000, qualityLevel=0.01,
                                  minDistance=7, mask=(inside.astype(np.uint8)*255))
    n = 0 if pts0 is None else len(pts0)
    failed = dict(klt_initial_count=n, klt_fb_good_count=0,
                  klt_fb_success_ratio=0.0 if n else float("nan"),
                  klt_good_per_1000px=0.0, klt_fb_median_error_px=float("nan"))
    if n == 0:
        return failed
    opts = dict(winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    pts1, st1, _ = cv2.calcOpticalFlowPyrLK(gray0, gray1, pts0, None, **opts)
    if pts1 is None or st1 is None:
        return failed
    fwd = pts1.reshape(-1, 2)
    finite_fwd = np.isfinite(fwd).all(axis=1)
    safe = fwd.copy()
    safe[~finite_fwd] = pts0.reshape(-1, 2)[~finite_fwd]
    returned, st2, _ = cv2.calcOpticalFlowPyrLK(gray1, gray0, safe.reshape(-1, 1, 2), None, **opts)
    if returned is None or st2 is None:
        return failed
    back = returned.reshape(-1, 2)
    orig = pts0.reshape(-1, 2)
    h, w = gray0.shape
    in_bounds = (fwd[:, 0] >= 0) & (fwd[:, 0] < w) & (fwd[:, 1] >= 0) & (fwd[:, 1] < h)
    finite = finite_fwd & np.isfinite(back).all(axis=1)
    valid = (st1.ravel() == 1) & (st2.ravel() == 1) & finite & in_bounds
    idx = np.nonzero(valid)[0]
    valid[idx] &= inside[fwd[idx, 1].astype(int), fwd[idx, 0].astype(int)]
    errors = np.linalg.norm(orig - back, axis=1)
    passed = valid & (errors <= fb_threshold)
    good = int(passed.sum())
    return {
        "klt_initial_count": n,
        "klt_fb_good_count": good,
        "klt_fb_success_ratio": good / n,
        "klt_good_per_1000px": 1000.0 * good / int(inside.sum()),
        "klt_fb_median_error_px": float(np.median(errors[valid])) if valid.any() else float("nan"),
    }
