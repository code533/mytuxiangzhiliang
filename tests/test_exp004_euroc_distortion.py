"""Regression tests for the official EuRoC camera distortion model label."""
import cv2
import numpy as np
import pytest
import yaml

from vio_iqa.exp004 import read_calibration, normalized_rays


@pytest.mark.parametrize("model", ["radtan", "radial-tangential"])
def test_euroc_radtan_alias_matches_opencv_geometry(tmp_path, model):
    sensor = tmp_path / "sensor.yaml"
    sensor.write_text(yaml.safe_dump({
        "camera_model": "pinhole",
        "resolution": [752, 480],
        "intrinsics": [458.654, 457.296, 367.215, 248.375],
        "distortion_model": model,
        "distortion_coefficients": [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-5],
    }))
    K, D, canonical = read_calibration(sensor, (480, 752))
    assert canonical == "radtan"
    assert D.shape == (4,)

    rays_3d = np.array([
        [0.4, -0.2, 3.0], [-1.0, 0.5, 4.0], [0.7, 0.8, 2.4], [-0.8, -0.4, 3.3],
    ], dtype=np.float64)
    distorted_px, _ = cv2.projectPoints(rays_3d, np.zeros(3), np.zeros(3), K, D)
    normalized = normalized_rays(distorted_px.reshape(-1, 2), K, D, canonical)
    np.testing.assert_allclose(normalized, rays_3d[:, :2] / rays_3d[:, 2, None], atol=1e-4)


def test_unknown_distortion_model_still_rejected(tmp_path):
    sensor = tmp_path / "sensor.yaml"
    sensor.write_text(yaml.safe_dump({
        "camera_model": "pinhole", "resolution": [752, 480],
        "intrinsics": [458.654, 457.296, 367.215, 248.375],
        "distortion_model": "unsupported-distortion",
        "distortion_coefficients": [-.28, .07, .0002, 0.0],
    }))
    with pytest.raises(ValueError, match="Unsupported distortion model"):
        read_calibration(sensor, (480, 752))
