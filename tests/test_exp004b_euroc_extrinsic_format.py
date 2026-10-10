"""EXP004-B regression: official EuRoC OpenCV-matrix T_BS representation."""
import numpy as np
import pytest
import yaml

from vio_iqa.exp004b import camera_extrinsic


def test_official_euroc_t_bs_rows_cols_data(tmp_path):
    # Nontrivial R and t ensure row-major and translation are preserved.
    T = np.array([
        [0., -1., 0., 0.12],
        [1.,  0., 0., -0.03],
        [0.,  0., 1., 0.08],
        [0.,  0., 0., 1.],
    ])
    path = tmp_path / "sensor.yaml"
    path.write_text(yaml.safe_dump({
        "T_BS": {"rows": 4, "cols": 4, "data": T.reshape(-1).tolist()},
    }), encoding="utf8")
    np.testing.assert_allclose(camera_extrinsic(path), T)


def test_explicit_nested_list_backward_compatible(tmp_path):
    path = tmp_path / "sensor.yaml"
    T = np.eye(4)
    T[0, 3] = 0.23
    path.write_text(yaml.safe_dump({"T_BS": T.tolist()}), encoding="utf8")
    np.testing.assert_allclose(camera_extrinsic(path), T)


@pytest.mark.parametrize("value", [
    {"rows": 3, "cols": 4, "data": [0.] * 12},
    {"rows": 4, "cols": 4, "data": [0.] * 15},
    {"rows": 4, "cols": 4},
    {"rows": 4, "cols": 4, "data": [0.] * 15 + [float("nan")]},
])
def test_reject_invalid_euroc_matrix_mapping(tmp_path, value):
    path = tmp_path / "sensor.yaml"
    path.write_text(yaml.safe_dump({"T_BS": value}), encoding="utf8")
    with pytest.raises(ValueError, match="T_BS"):
        camera_extrinsic(path)
