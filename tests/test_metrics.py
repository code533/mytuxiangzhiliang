import numpy as np
import pytest

from vio_iqa.metrics import frame_metrics
from vio_iqa.tracking import track_pair


def test_flat_image_has_zero_gradient_and_corners():
    img = np.full((96, 128), 127, dtype=np.uint8)
    m = frame_metrics(img)
    assert m["grad_p90"] == 0
    assert m["usable_gradient_ratio"] == 0
    assert m["fast_count"] == 0


def test_textured_image_has_features_and_stable_self_track():
    img = np.random.default_rng(42).integers(0, 256, (128, 128), dtype=np.uint8)
    m = frame_metrics(img)
    assert m["shi_tomasi_count_capped"] > 0
    assert m["fast_grid_occupancy"] > 0
    t = track_pair(img, img)
    assert t["klt_initial_count"] > 0
    assert t["klt_fb_success_ratio"] > 0.8


def test_flat_pair_has_missing_success_fraction():
    img = np.zeros((64, 64), dtype=np.uint8)
    t = track_pair(img, img)
    assert t["klt_initial_count"] == 0
    assert np.isnan(t["klt_fb_success_ratio"])


def test_wrong_grayscale_dtype_rejected():
    with pytest.raises(ValueError):
        frame_metrics(np.ones((40, 40), dtype=np.float32))
