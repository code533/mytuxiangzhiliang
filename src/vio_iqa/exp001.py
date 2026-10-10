"""Experiment 001: paired image degradation versus next-frame KLT quality.

Only original frame I_t (degraded identically as its next frame) is used for
predictors. Tracking from (I_t, I_(t+1)) supplies *retrospective* labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from vio_iqa.cli import load_index, METRIC_COLUMNS, TARGET_COLUMNS
from vio_iqa.metrics import frame_metrics
from vio_iqa.tracking import track_pair


def load_conditions(filename: Path) -> list[dict]:
    data = json.loads(filename.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data or data[0].get("name") != "clean":
        raise ValueError("Condition list must start with name=clean")
    names = [d["name"] for d in data]
    if len(set(names)) != len(names):
        raise ValueError("Condition names must be unique")
    kinds = {"clean", "gaussian_blur", "motion_blur", "lowlight", "contrast_loss", "overexposure", "gaussian_noise"}
    for d in data:
        if d.get("type") not in kinds:
            raise ValueError(f"Unsupported degradation type: {d.get('type')}")
        if d["type"] == "clean" and d["name"] != "clean":
            raise ValueError("Only the clean reference can use type=clean")
    return data


def image_to_u8(path: Path, uint16_white: int | None) -> tuple[np.ndarray, str]:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.ndim != 2:
        raise ValueError(f"Image must be readable single-channel: {path}")
    if raw.dtype == np.uint8:
        return raw, "uint8"
    if raw.dtype == np.uint16:
        if uint16_white is None or not (1 <= uint16_white <= 65535):
            raise ValueError("16-bit image requires --uint16-white in 1..65535; never auto-scale frames")
        return np.clip(raw.astype(np.float32) * (255.0 / uint16_white), 0, 255).round().astype(np.uint8), "uint16"
    raise ValueError(f"Unsupported source dtype {raw.dtype} in {path}")


def make_roi(shape: tuple[int, int], mask_path: str | None,
             circle_radius: float | None) -> np.ndarray:
    if mask_path and circle_radius is not None:
        raise ValueError("Choose --mask or --circle-radius, not both")
    h, w = shape
    if mask_path:
        raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if raw is None or raw.ndim != 2 or raw.shape != shape:
            raise ValueError("ROI mask must be single-channel and exactly match image dimensions")
        roi = raw != 0
    elif circle_radius is not None:
        if not (0 < circle_radius <= 0.5):
            raise ValueError("Circle radius must be in (0, 0.5], relative to min image dimension")
        yy, xx = np.ogrid[:h, :w]
        roi = ((xx - (w-1)/2)**2 + (yy - (h-1)/2)**2) <= (circle_radius * min(h, w))**2
    else:
        roi = np.ones(shape, dtype=bool)
    if not roi.any():
        raise ValueError("ROI mask is empty")
    return roi


def apply_degradation(img: np.ndarray, condition: dict, seed: int) -> np.ndarray:
    kind = condition["type"]
    if kind == "clean":
        return img.copy()
    if kind == "gaussian_blur":
        sigma = float(condition["sigma"])
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    if kind == "motion_blur":
        length = int(condition["length"])
        if length < 3 or length % 2 != 1:
            raise ValueError("motion blur length must be odd and >=3")
        angle = np.deg2rad(float(condition.get("angle_deg", 0)))
        kernel = np.zeros((length, length), dtype=np.float32)
        cx = (length - 1) / 2
        cv2.line(kernel,
                 (round(cx - cx * np.cos(angle)), round(cx - cx * np.sin(angle))),
                 (round(cx + cx * np.cos(angle)), round(cx + cx * np.sin(angle))),
                 1.0, 1)
        kernel /= kernel.sum()
        return cv2.filter2D(img, -1, kernel)
    if kind == "gaussian_noise":
        sigma = float(condition["sigma"])
        if sigma <= 0:
            raise ValueError("noise sigma must be positive")
        noise = np.random.default_rng(seed).normal(0.0, sigma, size=img.shape)
        return np.clip(img.astype(np.float32) + noise, 0, 255).round().astype(np.uint8)
    if kind in {"lowlight", "contrast_loss", "overexposure"}:
        scale = float(condition["factor"])
        if scale <= 0:
            raise ValueError("factor must be positive")
        if kind == "lowlight":
            if scale >= 1:
                raise ValueError("lowlight factor must be <1")
            changed = img.astype(np.float32) * scale
        elif kind == "contrast_loss":
            if scale >= 1:
                raise ValueError("contrast_loss factor must be <1")
            changed = (img.astype(np.float32) - 127.5) * scale + 127.5
        else:
            if scale <= 1:
                raise ValueError("overexposure factor must be >1")
            changed = img.astype(np.float32) * scale
        return np.clip(changed, 0, 255).round().astype(np.uint8)
    raise ValueError(f"Unknown degradation: {kind}")


def run(args) -> None:
    conditions = load_conditions(Path(args.conditions))
    cam, records = load_index(Path(args.sequence))
    if args.max_frames < 0 or args.stride < 1:
        raise ValueError("--max-frames must be nonnegative, --stride >= 1")
    if args.max_frames:
        records = records.iloc[:args.max_frames]
    if len(records) < 2:
        raise ValueError("At least 2 frames required")

    first, dtype = image_to_u8(cam / "data" / str(records.iloc[0].filename), args.uint16_white)
    roi = make_roi(first.shape, args.mask, args.circle_radius)
    cache: dict[int, np.ndarray] = {0: first}

    def frame(i):
        if i not in cache:
            cache[i], typ = image_to_u8(cam / "data" / str(records.iloc[i].filename), args.uint16_white)
            if typ != dtype or cache[i].shape != first.shape:
                raise ValueError("Image shapes/dtypes vary within a sequence")
        return cache[i]

    rows = []
    for i in range(0, len(records) - 1, args.stride):
        x, y = frame(i), frame(i+1)
        t0, t1 = int(records.iloc[i].timestamp_ns), int(records.iloc[i + 1].timestamp_ns)
        for cidx, condition in enumerate(conditions):
            # Deterministic independent seeds per frame and condition; shared frame retains same noise
            # regardless of whether it occurs as the first or the second frame of a pair.
            a = apply_degradation(x, condition, seed=args.seed + cidx * 1_000_003 + i)
            b = apply_degradation(y, condition, seed=args.seed + cidx * 1_000_003 + i + 1)
            rows.append({
                "sequence": Path(args.sequence).name,
                "frame_idx": i,
                "t_ns": t0, "t_next_ns": t1, "delta_t_s": (t1 - t0)/1e9,
                "condition": condition["name"], "degradation": condition["type"],
                "source_dtype": dtype, "uint16_white": args.uint16_white if dtype == "uint16" else None,
                "roi_coverage": float(roi.mean()),
                **frame_metrics(a, mask=roi),
                **track_pair(a, b, fb_threshold=args.fb_threshold, mask=roi),
            })
        cache = {i+1: y}
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "frame_pairs.csv", index=False)
    manifest = {
        "experiment": "exp001", "sequence": str(Path(args.sequence).resolve()),
        "n_pairs": len(range(0, len(records) - 1, args.stride)),
        "n_conditions": len(conditions), "conditions": conditions, "stride": args.stride,
        "max_frames": args.max_frames, "fb_threshold": args.fb_threshold,
        "seed": args.seed, "source_dtype": dtype,
        "uint16_white": args.uint16_white, "roi_fraction": float(roi.mean()),
        "mask": args.mask, "circle_radius": args.circle_radius,
        "notes": "Fixed global grayscale mapping and one fixed ROI per sequence. No VIO pose ground truth.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Created {len(rows)} rows from {manifest['n_pairs']} pairs x {len(conditions)} conditions in {out}")


def _rho(frame: pd.DataFrame, a: str, b: str) -> tuple[float, int]:
    data = frame[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
    n = len(data)
    if n < 10 or data[a].nunique() < 2 or data[b].nunique() < 2:
        return float("nan"), n
    return float(spearmanr(data[a], data[b]).statistic), n


def report(args):
    df = pd.read_csv(Path(args.input_dir) / "frame_pairs.csv")
    out = Path(args.output_dir or args.input_dir)
    out.mkdir(parents=True, exist_ok=True)
    if df.duplicated(["sequence", "frame_idx", "condition"]).any():
        raise ValueError("Duplicate sequence/frame/condition; do not concatenate same-run outputs")
    keys = ["sequence", "frame_idx"]
    clean = df[df.condition == "clean"]
    if len(clean) == 0:
        raise ValueError("Missing clean reference")
    base = clean.set_index(keys)
    # Natural-scene baseline: no artificial degradation, using real motion/scene variation.
    natural = []
    for seq, group in clean.groupby("sequence", sort=True):
        for target in TARGET_COLUMNS:
            for metric in METRIC_COLUMNS:
                rho, n = _rho(group, metric, target)
                natural.append(dict(sequence=seq, metric=metric, target=target,
                                    spearman_rho=rho, n=n))
    pd.DataFrame(natural).to_csv(out / "natural_correlations.csv", index=False)
    to_compare = METRIC_COLUMNS + TARGET_COLUMNS
    extra = df[df.condition != "clean"].copy()
    for col in to_compare:
        extra[f"delta_{col}"] = extra[col].to_numpy() - base.reindex(pd.MultiIndex.from_frame(extra[keys]))[col].to_numpy()
    extra.to_csv(out / "paired_deltas.csv", index=False)

    summary = []
    for (seq, condition, kind), group in extra.groupby(["sequence", "condition", "degradation"], sort=True):
        record = dict(sequence=seq, condition=condition, degradation=kind, n_pairs=len(group))
        for target in TARGET_COLUMNS:
            record[f"mean_delta_{target}"] = float(group[f"delta_{target}"].mean())
            record[f"median_delta_{target}"] = float(group[f"delta_{target}"].median())
        summary.append(record)
    pd.DataFrame(summary).to_csv(out / "condition_summary.csv", index=False)

    correlations = []
    # Stratify by sequence and degradation type. In each stratum all levels are pooled:
    # exploratory correlations only; no frame-level p-values due to temporal dependence.
    for (seq, kind), group in extra.groupby(["sequence", "degradation"], sort=True):
        for target in TARGET_COLUMNS:
            for metric in METRIC_COLUMNS:
                rho, n = _rho(group, f"delta_{metric}", f"delta_{target}")
                correlations.append(dict(sequence=seq, degradation=kind, metric=metric,
                                         target=target, spearman_delta_rho=rho, n=n))
    pd.DataFrame(correlations).to_csv(out / "delta_correlations.csv", index=False)

    if not extra.empty:
        fig, ax = plt.subplots(figsize=(11, 5))
        group = extra.groupby("condition", sort=True)["delta_klt_fb_good_count"].agg(["mean", "median"])
        ax.bar(range(len(group)), group["mean"], label="Mean paired delta")
        ax.set_xticks(range(len(group)), group.index, rotation=45, ha="right")
        ax.set_ylabel("Δ forward-backward surviving feature count")
        ax.set_title("Controlled degradation vs clean (same frame pairs)")
        fig.tight_layout()
        fig.savefig(out / "paired_count_delta.png", dpi=150)
        plt.close(fig)
    print(f"Reports saved in {out}")


def main():
    parser = argparse.ArgumentParser(description="Experiment 001: paired degradations and KLT")
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--sequence", required=True, help="EuRoC ASL layout including TUM-VI exported euroc/mav0/cam0")
    r.add_argument("--conditions", default="experiments/exp001/conditions.json")
    r.add_argument("--output-dir", required=True)
    r.add_argument("--max-frames", type=int, default=300, help="0 = all frames")
    r.add_argument("--stride", type=int, default=1, help="sample every Nth starting frame; still track adjacent pair")
    r.add_argument("--fb-threshold", type=float, default=1.0)
    r.add_argument("--seed", type=int, default=533)
    r.add_argument("--uint16-white", type=int, help="Required for 16-bit source; fixed source level mapped to 255")
    r.add_argument("--mask", help="static 1-channel nonzero=valid image path")
    r.add_argument("--circle-radius", type=float, help="static central circular ROI radius / min(width,height)")
    r.set_defaults(func=run)
    p = sub.add_parser("report")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir")
    p.set_defaults(func=report)
    a = parser.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
