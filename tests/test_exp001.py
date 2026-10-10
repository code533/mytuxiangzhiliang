from argparse import Namespace
import json

import cv2
import numpy as np
import pandas as pd
import pytest

from vio_iqa.exp001 import apply_degradation, image_to_u8, make_roi, report, run
from vio_iqa.metrics import frame_metrics
from vio_iqa.tracking import track_pair


def _dataset(tmp_path, bit16=False):
    root = tmp_path / "synthetic" / "mav0" / "cam0"
    (root / "data").mkdir(parents=True)
    rng = np.random.default_rng(111)
    frame = rng.integers(20, 220, (96, 96), dtype=np.uint8)
    rows = []
    for i in range(13):
        shifted = np.roll(frame, (0, i), axis=(0, 1))
        if bit16:
            shifted = shifted.astype(np.uint16) * 200
        name = f"{i}.png"
        assert cv2.imwrite(str(root / "data" / name), shifted)
        rows.append(f"{1_000_000_000 + i * 50_000_000},{name}")
    (root / "data.csv").write_text("#timestamp [ns],filename\n" + "\n".join(rows) + "\n")
    return root.parent.parent


def test_mask_excludes_black_border_and_limits_count():
    rng = np.random.default_rng(9)
    img = rng.integers(15, 240, (96, 96), dtype=np.uint8)
    roi = make_roi(img.shape, None, 0.35)
    im2 = img.copy()
    im2[~roi] = 0
    base = frame_metrics(img, mask=roi)
    changed = frame_metrics(im2, mask=roi)
    assert np.isclose(base["dark_ratio"], changed["dark_ratio"])
    assert np.isclose(base["grad_p90"], changed["grad_p90"])
    a = track_pair(img, img, mask=roi)
    b = track_pair(im2, im2, mask=roi)
    assert a["klt_initial_count"] > 0
    assert b["klt_fb_good_count"] > 0
    assert abs(a["klt_initial_count"] - b["klt_initial_count"]) <= 3


def test_16bit_requires_explicit_mapping(tmp_path):
    x = np.full((8, 8), 8000, np.uint16)
    p = tmp_path / "image.png"
    cv2.imwrite(str(p), x)
    with pytest.raises(ValueError, match="uint16-white"):
        image_to_u8(p, None)
    mapped, typ = image_to_u8(p, 16000)
    assert typ == "uint16"
    assert np.all(mapped == 128)


def test_gaussian_noise_deterministic_per_frame():
    img = np.full((50, 50), 127, np.uint8)
    cond = dict(type="gaussian_noise", sigma=10)
    assert np.array_equal(apply_degradation(img, cond, 123), apply_degradation(img, cond, 123))
    assert not np.array_equal(apply_degradation(img, cond, 123), apply_degradation(img, cond, 124))


def test_exp001_end_to_end_8bit(tmp_path):
    sequence = _dataset(tmp_path)
    cfg = tmp_path / "conditions.json"
    cfg.write_text(json.dumps([{"name": "clean", "type": "clean"},
                               {"name": "blur", "type": "gaussian_blur", "sigma": 2.0},
                               {"name": "low", "type": "lowlight", "factor": 0.2}]))
    out = tmp_path / "out"
    params = dict(sequence=str(sequence), conditions=str(cfg), output_dir=str(out),
                  max_frames=0, stride=1, fb_threshold=1.0, seed=533,
                  uint16_white=None, mask=None, circle_radius=0.46)
    run(Namespace(**params))
    results = pd.read_csv(out / "frame_pairs.csv")
    assert len(results) == 12 * 3
    assert results.groupby("condition").size().eq(12).all()
    assert set(results["source_dtype"]) == {"uint8"}
    assert results["roi_coverage"].between(0.60, 0.70).all()
    report(Namespace(input_dir=str(out), output_dir=None))
    d = pd.read_csv(out / "paired_deltas.csv")
    assert len(d) == 24
    for name in ["natural_correlations.csv", "condition_summary.csv", "delta_correlations.csv", "paired_count_delta.png"]:
        assert (out / name).exists()
    assert pd.read_csv(out / "condition_summary.csv").shape[0] == 2
    original = (out / "frame_pairs.csv").read_bytes()
    run(Namespace(**params))
    assert (out / "frame_pairs.csv").read_bytes() == original


def test_exp001_end_to_end_16bit(tmp_path):
    sequence = _dataset(tmp_path, bit16=True)
    cfg = tmp_path / "conditions.json"
    cfg.write_text('[{"name":"clean","type":"clean"}, {"name":"noise","type":"gaussian_noise","sigma":5}]')
    out = tmp_path / "out"
    run(Namespace(sequence=str(sequence), conditions=str(cfg), output_dir=str(out),
                  max_frames=3, stride=1, fb_threshold=1.0, seed=3,
                  uint16_white=50000, mask=None, circle_radius=None))
    df = pd.read_csv(out / "frame_pairs.csv")
    assert len(df) == 4
    assert set(df.source_dtype) == {"uint16"}
