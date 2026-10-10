"""EXP002: causal image predictors and retrospective K-frame survival labels.

Frames observed through t can be predictors; future images are used for labels ONLY.
A cohort of Shi–Tomasi corners starts at each sampled t and is never replenished.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from vio_iqa.cli import METRIC_COLUMNS, load_index
from vio_iqa.exp001 import image_to_u8, make_roi
from vio_iqa.metrics import frame_metrics

HORIZONS = (5, 10)
# Candidate selection and evaluation follow pre-registered defaults.
LK = dict(winSize=(21, 21), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def _inside(mask: np.ndarray) -> np.ndarray:
    return cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) != 0


def select_spread(gray: np.ndarray, inside: np.ndarray, budget=240,
                  candidates=3000, grid=(8, 6)) -> np.ndarray:
    """Shi–Tomasi ranked candidates, spatial round-robin; output Nx2 float32."""
    if budget < 1 or candidates < budget:
        raise ValueError("Must have 1 <= budget <= candidates")
    h, w = gray.shape
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=candidates, qualityLevel=0.01,
                                  minDistance=7, mask=inside.astype(np.uint8) * 255)
    if pts is None:
        return np.empty((0, 2), np.float32)
    buckets = [[] for _ in range(grid[0] * grid[1])]
    for p in pts.reshape(-1, 2):
        bx = min(grid[0] - 1, int(p[0] * grid[0] / w))
        by = min(grid[1] - 1, int(p[1] * grid[1] / h))
        buckets[by * grid[0] + bx].append(p)
    selected = []
    while len(selected) < budget:
        changed = False
        for b in buckets:
            if b and len(selected) < budget:
                selected.append(b.pop(0))
                changed = True
        if not changed:
            break
    return np.asarray(selected, dtype=np.float32).reshape(-1, 2)


def _advance(cohorts: list[dict], prev: np.ndarray, current: np.ndarray,
             inside: np.ndarray, threshold: float) -> None:
    """One batched forward/backward KLT call for all active cohorts."""
    sizes = [len(c["points"]) for c in cohorts]
    n = sum(sizes)
    passed = np.zeros(n, dtype=bool)
    next_pts = np.empty((n, 2), dtype=np.float32)
    if n:
        pts = np.concatenate([c["points"] for c in cohorts]).reshape(-1, 1, 2)
        fwd, status1, _ = cv2.calcOpticalFlowPyrLK(prev, current, pts, None, **LK)
        if fwd is not None and status1 is not None:
            fwd = fwd.reshape(-1, 2)
            good_finite = np.isfinite(fwd).all(axis=1)
            safe = fwd.copy()
            safe[~good_finite] = pts.reshape(-1, 2)[~good_finite]
            back, status2, _ = cv2.calcOpticalFlowPyrLK(
                current, prev, safe.reshape(-1, 1, 2), None, **LK)
            if back is not None and status2 is not None:
                back = back.reshape(-1, 2)
                h, w = prev.shape
                in_bounds = (fwd[:, 0] >= 0) & (fwd[:, 0] < w) & (fwd[:, 1] >= 0) & (fwd[:, 1] < h)
                valid = (status1.ravel() == 1) & (status2.ravel() == 1)
                valid &= good_finite & np.isfinite(back).all(axis=1) & in_bounds
                ids = np.flatnonzero(valid)
                valid[ids] &= inside[fwd[ids, 1].astype(int), fwd[ids, 0].astype(int)]
                error = np.linalg.norm(pts.reshape(-1, 2) - back, axis=1)
                passed = valid & (error <= threshold)
                next_pts[:] = safe
    offset = 0
    for c, size in zip(cohorts, sizes):
        use = passed[offset:offset + size]
        if size:
            c["points"] = next_pts[offset:offset + size][use]
            c["ids"] = c["ids"][use]
        c["counts"].append(len(c["points"]))
        offset += size


def extract(args) -> None:
    if args.max_frames < 0 or args.stride < 1 or args.budget < 1 or args.candidates < args.budget:
        raise ValueError("Invalid frame count, stride, budget, or candidate count")
    if args.fb_threshold < 0:
        raise ValueError("--fb-threshold must be nonnegative")
    cam, index = load_index(Path(args.sequence))
    if args.max_frames:
        index = index.iloc[:args.max_frames]
    if index.empty:
        raise ValueError("No images in sequence")
    def read(i):
        return image_to_u8(cam / "data" / str(index.iloc[i].filename), args.uint16_white)
    prev, dtype = read(0)
    roi = make_roi(prev.shape, args.mask, args.circle_radius)
    inside = _inside(roi)
    if not inside.any():
        raise ValueError("Mask too small after erosion")
    active, complete = [], []
    for i in range(len(index)):
        current, typ = (prev, dtype) if i == 0 else read(i)
        if typ != dtype or current.shape != prev.shape:
            raise ValueError("Input image dtype/shape must remain fixed within sequence")
        if i > 0:
            _advance(active, prev, current, inside, args.fb_threshold)
            expired = [c for c in active if len(c["counts"]) == max(HORIZONS)]
            complete.extend(expired)
            active = [c for c in active if len(c["counts"]) < max(HORIZONS)]
        if i % args.stride == 0:
            pts = select_spread(current, inside, args.budget, args.candidates)
            row = dict(sequence=Path(args.sequence).name, frame_idx=i,
                       t_ns=int(index.iloc[i].timestamp_ns), source_dtype=dtype,
                       roi_coverage=float(roi.mean()),
                       n_initial=len(pts), no_features=int(len(pts) == 0),
                       **frame_metrics(current, mask=roi))
            active.append(dict(row=row, points=pts, ids=np.arange(len(pts)), counts=[]))
        prev = current
    complete.extend(active)
    rows = []
    for c in sorted(complete, key=lambda x: x["row"]["frame_idx"]):
        row = c["row"]
        n = row["n_initial"]
        row["observed_steps"] = len(c["counts"])
        for k in HORIZONS:
            observed = len(c["counts"]) >= k
            row[f"censored_{k}"] = int(not observed)
            row[f"n_survive_{k}"] = c["counts"][k - 1] if observed else np.nan
            row[f"s_{k}"] = (c["counts"][k - 1] / n) if observed and n else np.nan
        rows.append(row)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "survival_labels.csv", index=False)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    manifest = dict(experiment="EXP002", sequence=str(Path(args.sequence).resolve()),
                    n_images=len(index), n_start_frames=len(rows), stride=args.stride,
                    max_frames=args.max_frames, horizons=list(HORIZONS),
                    n_valid_5=int(pd.notna(pd.DataFrame(rows)["s_5"]).sum()),
                    n_valid_10=int(pd.notna(pd.DataFrame(rows)["s_10"]).sum()),
                    budget=args.budget, candidates=args.candidates,
                    fb_threshold=args.fb_threshold, source_dtype=dtype,
                    uint16_white=args.uint16_white, mask=args.mask,
                    circle_radius=args.circle_radius, roi_fraction=float(roi.mean()),
                    git_commit=sha.stdout.strip() if sha.returncode == 0 else None,
                    label_only_future=True,
                    zero_features="s_K = NaN with no_features=1",
                    short_tail="s_K = NaN with censored_K=1; not a failure")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(rows)} rows, valid S5={manifest['n_valid_5']}, S10={manifest['n_valid_10']} to {out}")


def evaluate(args) -> None:
    """Held-out whole-sequence regression; all model choices are fixed a priori."""
    from sklearn.dummy import DummyRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    if not args.train or not args.test:
        raise ValueError("Provide both train and test CSV paths")
    if set(map(str, args.train)) & set(map(str, args.test)):
        raise ValueError("Same input CSV cannot be in train and test")
    def read_group(paths):
        arr = []
        for p in paths:
            x = pd.read_csv(p)
            if x.empty or x["sequence"].nunique() != 1:
                raise ValueError("Each input must contain one nonempty sequence")
            arr.append(x)
        d = pd.concat(arr, ignore_index=True)
        if d.duplicated(["sequence", "frame_idx"]).any():
            raise ValueError("Duplicate sequence/frame, possible synthetic copy leakage")
        return d
    train, test = read_group(args.train), read_group(args.test)
    tr, te = set(train.sequence), set(test.sequence)
    if tr & te:
        raise ValueError(f"Train/test sequence overlap: {tr & te}")
    ycol = f"s_{args.horizon}"
    train = train.loc[(train.no_features == 0) & train[ycol].notna()].copy()
    test = test.loc[(test.no_features == 0) & test[ycol].notna()].copy()
    if len(train) < 10 or len(test) < 5:
        raise ValueError("Not enough valid labeled rows; use more images/sequences")
    # Do not use motion/next-frame observations, or derived survival counts as features.
    features = [c for c in METRIC_COLUMNS if c != "shi_tomasi_count_capped"]
    models = {"constant_train_mean": ([], DummyRegressor(strategy="mean"))}
    for col in features:
        models[f"single_{col}"] = ([col], make_pipeline(SimpleImputer(strategy="median"),
                                                       StandardScaler(), Ridge(alpha=10.0)))
    models["image_multimetric_ridge"] = (features, make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)))
    ytrain = train[ycol].to_numpy(float)
    ytest = test[ycol].to_numpy(float)
    scores, predictions = [], []
    for name, (cols, model) in models.items():
        if cols:
            model.fit(train[cols], ytrain)
            pred = np.clip(model.predict(test[cols]), 0, 1)
        else:
            model.fit(np.zeros((len(train), 1)), ytrain)
            pred = model.predict(np.zeros((len(test), 1)))
        for seq in sorted(test.sequence.unique()):
            sel = test.sequence.to_numpy() == seq
            yy, pp = ytest[sel], pred[sel]
            rho = float(spearmanr(yy, pp).statistic) if np.unique(yy).size > 1 and np.unique(pp).size > 1 else np.nan
            scores.append(dict(model=name, target=ycol, test_sequence=seq,
                               train_sequences=";".join(sorted(tr)), n_test=len(yy),
                               mae=mean_absolute_error(yy, pp),
                               rmse=np.sqrt(mean_squared_error(yy, pp)), spearman=rho))
        predictions.extend(dict(sequence=s, frame_idx=int(i), target=ycol,
                                actual=float(yy), predicted=float(pp), model=name)
                           for s, i, yy, pp in zip(test.sequence, test.frame_idx, ytest, pred))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scores).to_csv(out / "heldout_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(out / "heldout_predictions.csv", index=False)
    (out / "split.json").write_text(json.dumps(dict(train=list(map(str,args.train)), test=list(map(str,args.test)),
        train_sequences=sorted(tr), test_sequences=sorted(te), horizon=args.horizon,
        ridge_alpha=10.0, predictors=features, train_rows=len(train), test_rows=len(test),
        note="Exploratory if only one held-out sequence; no frame-level shuffling or future-frame predictors."), indent=2))
    print(pd.DataFrame(scores).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="EXP002: predict K-frame feature survival")
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("extract", help="Generate retrospective S5/S10 labels")
    p.add_argument("--sequence", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-frames", type=int, default=300, help="0 = entire sequence")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--budget", type=int, default=240)
    p.add_argument("--candidates", type=int, default=3000)
    p.add_argument("--fb-threshold", type=float, default=1.0)
    p.add_argument("--uint16-white", type=int)
    p.add_argument("--mask")
    p.add_argument("--circle-radius", type=float)
    p.set_defaults(func=extract)
    p = subs.add_parser("evaluate", help="Compare models on disjoint whole sequences")
    p.add_argument("--train", nargs="+", required=True, type=Path)
    p.add_argument("--test", nargs="+", required=True, type=Path)
    p.add_argument("--horizon", choices=HORIZONS, default=5, type=int)
    p.add_argument("--output-dir", required=True)
    p.set_defaults(func=evaluate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
