import json
from argparse import Namespace
import cv2
import numpy as np
import pandas as pd
import pytest

from vio_iqa.exp002 import extract, evaluate


def sequence(tmp_path, name, n=18, blank=False, bit16=False):
    path = tmp_path / name / "mav0" / "cam0"
    (path / "data").mkdir(parents=True)
    texture = np.random.default_rng(42).integers(20, 235, (100, 120), dtype=np.uint8)
    rows = ["#timestamp [ns],filename"]
    for k in range(n):
        img = np.zeros_like(texture) if blank else np.roll(texture, k, axis=1)
        if bit16:
            img = img.astype(np.uint16) * 200
        assert cv2.imwrite(str(path / "data" / f"{k}.png"), img)
        rows.append(f"{1_000_000_000+k*50_000_000},{k}.png")
    (path / "data.csv").write_text("\n".join(rows) + "\n")
    return path.parent.parent


def args(seq, out, **updates):
    a = dict(sequence=str(seq), output_dir=str(out), max_frames=0,
             stride=1, budget=72, candidates=300, fb_threshold=1.0,
             uint16_white=None, mask=None, circle_radius=None)
    a.update(updates)
    return Namespace(**a)


def test_survival_and_tail(tmp_path):
    out = tmp_path / "out"
    extract(args(sequence(tmp_path, "A"), out))
    df = pd.read_csv(out / "survival_labels.csv")
    assert len(df) == 18
    assert (df.iloc[:8].s_10 >= .8).all()
    assert df.iloc[:13].s_5.notna().all() and df.iloc[13:].s_5.isna().all()
    assert df.iloc[:8].s_10.notna().all() and df.iloc[8:].s_10.isna().all()
    assert (df.iloc[8:].censored_10 == 1).all()
    assert (df.n_initial <= 72).all()
    assert (df.s_10.dropna().to_numpy() <= df.s_5.iloc[:8].to_numpy()+1e-7).all()
    assert json.loads((out / "manifest.json").read_text())["n_valid_10"] == 8
    original = (out / "survival_labels.csv").read_bytes()
    extract(args(tmp_path / "A", out))
    assert (out / "survival_labels.csv").read_bytes() == original


def test_no_features_are_not_zero_survival(tmp_path):
    out = tmp_path / "out"
    extract(args(sequence(tmp_path, "Blank", n=12, blank=True), out))
    d = pd.read_csv(out / "survival_labels.csv")
    assert (d.no_features == 1).all()
    assert d.s_5.isna().all() and d.s_10.isna().all()
    assert (d.n_initial == 0).all()


def test_stride_counts_actual_frame_steps(tmp_path):
    out = tmp_path / "out"
    extract(args(sequence(tmp_path, "Skip", n=14), out, stride=3))
    d = pd.read_csv(out / "survival_labels.csv")
    assert d.frame_idx.tolist() == [0, 3, 6, 9, 12]
    assert d.censored_5.tolist() == [0, 0, 0, 1, 1]
    assert d.censored_10.tolist() == [0, 0, 1, 1, 1]


def test_16bit_explicit_mapping(tmp_path):
    seq = sequence(tmp_path, "U16", n=12, bit16=True)
    with pytest.raises(ValueError, match="uint16-white"):
        extract(args(seq, tmp_path / "bad"))
    out = tmp_path / "ok"
    extract(args(seq, out, uint16_white=50000, circle_radius=.42))
    d = pd.read_csv(out / "survival_labels.csv")
    assert set(d.source_dtype) == {"uint16"}


def test_heldout_split_and_metrics(tmp_path):
    src1 = sequence(tmp_path, "Train", n=20)
    src2 = sequence(tmp_path, "Test", n=20)
    out1, out2 = tmp_path / "a", tmp_path / "b"
    extract(args(src1, out1))
    extract(args(src2, out2))
    a, b = out1 / "survival_labels.csv", out2 / "survival_labels.csv"
    dest = tmp_path / "eval"
    evaluate(Namespace(train=[a], test=[b], horizon=5, output_dir=str(dest)))
    score = pd.read_csv(dest / "heldout_metrics.csv")
    assert len(score) == 9 and score.mae.ge(0).all()
    assert "single_fast_count" in score.model.values
    with pytest.raises(ValueError, match="overlap|Same input CSV"):
        evaluate(Namespace(train=[a], test=[a], horizon=5, output_dir=str(dest)))
    duplicate = tmp_path / "copy.csv"
    duplicate.write_bytes(a.read_bytes())
    with pytest.raises(ValueError, match="overlap"):
        evaluate(Namespace(train=[a], test=[duplicate], horizon=5, output_dir=str(dest)))
