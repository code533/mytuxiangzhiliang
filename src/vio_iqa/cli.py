"""CLI for EuRoC extraction and exploratory reporting."""
import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from vio_iqa.metrics import frame_metrics
from vio_iqa.tracking import track_pair

METRIC_COLUMNS = [
    "grad_p90", "usable_gradient_ratio", "laplacian_variance", "dark_ratio",
    "saturated_ratio", "fast_count", "fast_grid_occupancy", "shi_tomasi_count_capped",
]
TARGET_COLUMNS = ["klt_fb_success_ratio", "klt_fb_good_count", "klt_good_per_1000px"]


def load_index(sequence: Path) -> tuple[Path, pd.DataFrame]:
    cam = sequence / "mav0" / "cam0" if (sequence / "mav0").is_dir() else sequence / "cam0"
    index_file = cam / "data.csv"
    if not index_file.is_file():
        raise FileNotFoundError(f"EuRoC camera index not found: {index_file}")
    records = pd.read_csv(index_file, comment="#", header=None,
                          names=["timestamp_ns", "filename"],
                          dtype={"timestamp_ns": "int64", "filename": "string"})
    if records.empty or records["filename"].isna().any():
        raise ValueError("Empty or invalid EuRoC camera index")
    if not records.timestamp_ns.is_monotonic_increasing or records.timestamp_ns.duplicated().any():
        raise ValueError("EuRoC timestamps must be strictly increasing")
    return cam, records


def run(args):
    sequence = Path(args.sequence)
    cam, records = load_index(sequence)
    if args.max_frames < 0:
        raise ValueError("--max-frames cannot be negative")
    if args.max_frames:
        records = records.iloc[:args.max_frames]
    if len(records) < 2:
        raise ValueError("Need at least two frames")

    def image_at(i):
        path = cam / "data" / str(records.iloc[i].filename)
        im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise FileNotFoundError(f"Missing or unreadable image: {path}")
        return im

    prev = image_at(0)
    rows = []
    for i in range(len(records) - 1):
        nxt = image_at(i + 1)
        t0, t1 = int(records.iloc[i].timestamp_ns), int(records.iloc[i+1].timestamp_ns)
        rows.append({
            "sequence": sequence.name,
            "t_ns": t0,
            "t_next_ns": t1,
            "delta_t_s": (t1 - t0) / 1e9,
            **frame_metrics(prev),
            **track_pair(prev, nxt, fb_threshold=args.fb_threshold),
        })
        prev = nxt
        if (i+1) % 100 == 0:
            print(f"Processed {i+1}/{len(records)-1} pairs", flush=True)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Wrote {len(rows)} frame pairs to {path}")


def report(args):
    df = pd.read_csv(args.input)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for target in TARGET_COLUMNS:
        for metric in METRIC_COLUMNS:
            if target not in df or metric not in df:
                continue
            valid = df[[metric, target]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid) < 10 or valid[metric].nunique() < 2 or valid[target].nunique() < 2:
                rho = float("nan")
            else:
                rho = float(spearmanr(valid[metric].to_numpy(), valid[target].to_numpy()).statistic)
            rows.append(dict(target=target, metric=metric, spearman_rho=rho, n=len(valid)))
    stats = pd.DataFrame(rows, columns=["target", "metric", "spearman_rho", "n"])
    stats.to_csv(out / "spearman.csv", index=False)
    print(stats.to_string(index=False))
    if "klt_fb_success_ratio" in df and "fast_count" in df:
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        axes[0].plot(np.arange(len(df)), df["klt_fb_success_ratio"])
        axes[0].set(ylabel="FB success fraction", ylim=(-0.05, 1.05))
        axes[1].plot(np.arange(len(df)), df["fast_count"])
        axes[1].set(xlabel="Frame pair index", ylabel="FAST keypoints")
        fig.tight_layout()
        fig.savefig(out / "timeline.png", dpi=160)
        plt.close(fig)
    print(f"Report written to {out}")


def main():
    parser = argparse.ArgumentParser(description="VIO-IQA: image quality vs next-frame KLT")
    cmds = parser.add_subparsers(dest="command", required=True)
    p = cmds.add_parser("run")
    p.add_argument("--sequence", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-frames", type=int, default=500, help="0 means full sequence")
    p.add_argument("--fb-threshold", type=float, default=1.0)
    p.set_defaults(func=run)
    p = cmds.add_parser("report")
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    p.set_defaults(func=report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
