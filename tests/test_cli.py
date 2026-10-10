from argparse import Namespace

import cv2
import numpy as np
import pandas as pd

from vio_iqa.cli import load_index, report, run


def test_euroc_smoke_test(tmp_path):
    seq = tmp_path / "MH_01_easy"
    cam = seq / "mav0" / "cam0"
    (cam / "data").mkdir(parents=True)
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, (120, 160), dtype=np.uint8)
    cv2.imwrite(str(cam / "data" / "a.png"), img)
    cv2.imwrite(str(cam / "data" / "b.png"), img)
    (cam / "data.csv").write_text("#timestamp [ns],filename\n1000000000,a.png\n1050000000,b.png\n")
    directory, index = load_index(seq)
    assert directory == cam and len(index) == 2
    output = tmp_path / "results" / "test.csv"
    run(Namespace(sequence=str(seq), max_frames=0, fb_threshold=1.0, output=str(output)))
    df = pd.read_csv(output)
    assert len(df) == 1
    assert df["klt_fb_success_ratio"].iloc[0] > 0.8
    out = tmp_path / "report"
    report(Namespace(input=str(output), output_dir=str(out)))
    assert (out / "spearman.csv").exists()
    assert (out / "timeline.png").exists()
