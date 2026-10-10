"""EXP003: past-only IMU features and image/IMU ablation experiments.

Input labels are from EXP002; future frames were used only to make these labels.
IMU feature windows are (t-window, t], never (t, t+K].
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

IMAGE_FEATURES = [
    "grad_p90", "usable_gradient_ratio", "laplacian_variance", "dark_ratio",
    "saturated_ratio", "fast_count", "fast_grid_occupancy",
]
WINDOWS_S = (0.20, 0.50)
IMU_FIELDS = ("wx", "wy", "wz", "ax", "ay", "az")
IMU_FEATURES = [
    f"imu_{w}ms_{metric}"
    for w in (200, 500)
    for metric in ("gyro_mean", "gyro_p90", "gyro_max", "angle_integral",
                   "gyro_net_integral", "acc_mean", "acc_norm_std", "acc_axes_std")
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read_euroc_imu(path: Path) -> pd.DataFrame:
    """EuRoC imu0/data.csv: timestamp, omega_xyz [rad/s], accel_xyz [m/s^2]."""
    if not path.is_file():
        raise FileNotFoundError(f'IMU file not found: {path}')
    names = ['imu_t_ns', *IMU_FIELDS]
    data = pd.read_csv(path, comment='#', header=None, names=names)
    if data.empty or data[names].isnull().any().any():
        raise ValueError('IMU data empty or contains missing values')
    for col in names:
        data[col] = pd.to_numeric(data[col], errors='raise')
    data['imu_t_ns'] = data['imu_t_ns'].astype(np.int64)
    if not data.imu_t_ns.is_monotonic_increasing or data.imu_t_ns.duplicated().any():
        raise ValueError('IMU timestamps must be strictly increasing')
    if not np.isfinite(data[list(IMU_FIELDS)].to_numpy(float)).all():
        raise ValueError('IMU data contains NaN or Inf')
    return data


def _window_features(ts_ns: int, imu_ns: np.ndarray, values: np.ndarray,
                     window_s: float, max_gap_s: float) -> dict[str, float] | None:
    """Require timestamp coverage near both ends, excluding future samples."""
    begin = ts_ns - round(window_s * 1e9)
    left = int(np.searchsorted(imu_ns, begin, side='left'))
    right = int(np.searchsorted(imu_ns, ts_ns, side='right'))
    if right - left < 3:
        return None
    times = imu_ns[left:right]
    # Require the full past window; no silently partial windows at sequence start.
    if (times[0] - begin) / 1e9 > max_gap_s or (ts_ns - times[-1]) / 1e9 > max_gap_s:
        return None
    dt = np.diff(times).astype(np.float64) * 1e-9
    if np.any(dt <= 0) or np.max(dt) > max_gap_s:
        return None
    vals = values[left:right]
    gyro = vals[:, :3]
    accel = vals[:, 3:]
    gnorm = np.linalg.norm(gyro, axis=1)
    anorm = np.linalg.norm(accel, axis=1)
    # Raw gyro bias remains; this is not integrated calibrated orientation.
    return {
        'gyro_mean': float(gnorm.mean()),
        'gyro_p90': float(np.percentile(gnorm, 90)),
        'gyro_max': float(np.max(gnorm)),
        'angle_integral': float(np.sum((gnorm[:-1] + gnorm[1:]) * 0.5 * dt)),
        'gyro_net_integral': float(np.linalg.norm(np.sum((gyro[:-1] + gyro[1:]) * (0.5 * dt[:, None]), axis=0))),
        'acc_mean': float(anorm.mean()),
        'acc_norm_std': float(anorm.std()),
        'acc_axes_std': float(np.linalg.norm(np.std(accel, axis=0))),
    }


def imu_features_at_t(frame_times_ns: np.ndarray, imu: pd.DataFrame,
                      max_gap_s: float = 0.03) -> pd.DataFrame:
    if max_gap_s <= 0:
        raise ValueError('max_gap_s must be positive')
    timestamps = np.asarray(frame_times_ns, dtype=np.int64)
    imu_ns = imu.imu_t_ns.to_numpy(dtype=np.int64)
    vals = imu[list(IMU_FIELDS)].to_numpy(dtype=np.float64)
    rows = []
    for timestamp in timestamps:
        out: dict[str, float | int] = {}
        for w in WINDOWS_S:
            key = int(round(w * 1000))
            stats = _window_features(int(timestamp), imu_ns, vals, w, max_gap_s)
            out[f'imu_{key}ms_valid'] = int(stats is not None)
            for field in ('gyro_mean', 'gyro_p90', 'gyro_max', 'angle_integral',
                          'gyro_net_integral', 'acc_mean', 'acc_norm_std', 'acc_axes_std'):
                out[f'imu_{key}ms_{field}'] = stats[field] if stats is not None else float('nan')
        rows.append(out)
    return pd.DataFrame(rows)


def augment(args: argparse.Namespace) -> None:
    labels_path = Path(args.labels)
    imu_path = Path(args.imu_csv)
    labels = pd.read_csv(labels_path, dtype={'t_ns': 'int64'})
    required = set(IMAGE_FEATURES + ['sequence', 'frame_idx', 't_ns', 's_5', 's_10', 'no_features'])
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f'Not EXP002 survival_labels.csv; missing columns: {missing}')
    if labels.empty or labels.sequence.nunique() != 1 or labels[['sequence', 'frame_idx']].duplicated().any():
        raise ValueError('Expected one sequence of unique nonempty EXP002 frames')
    if not labels.t_ns.is_monotonic_increasing:
        raise ValueError('Camera timestamps should be increasing')
    imu = read_euroc_imu(imu_path)
    feats = imu_features_at_t(labels.t_ns.to_numpy(np.int64), imu, args.max_gap_s)
    out = pd.concat([labels.reset_index(drop=True), feats], axis=1)
    dest = Path(args.output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest / 'survival_imu_features.csv', index=False)
    complete = out['imu_200ms_valid'].eq(1) & out['imu_500ms_valid'].eq(1)
    summary = {
        'experiment': 'EXP003', 'label_csv': str(labels_path.resolve()),
        'label_sha256': _sha256(labels_path), 'imu_csv': str(imu_path.resolve()),
        'imu_sha256': _sha256(imu_path), 'time_policy': 'past_only_imu_t_le_camera_t',
        'camera_to_imu_time_offset_ns': 0, 'imu_window_seconds': list(WINDOWS_S),
        'imu_max_gap_s': args.max_gap_s, 'imu_units': {'gyro':'rad/s','accel':'m/s^2'},
        'raw_imu_bias_corrected': False, 'imu_time_calibration_applied': False,
        'source_sequence': str(labels.sequence.iloc[0]),
        'rows': len(out), 'n_imu_complete': int(complete.sum()),
        'n_imu_incomplete': int((~complete).sum()),
        'notes': 'Pre-frame features from past raw IMU only. S5/S10 remain retrospective EXP002 labels.',
    }
    (dest / 'manifest.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote {len(out)} frames ({int(complete.sum())} complete IMU windows) to {dest}')


def _load_group(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if frame.empty or frame.sequence.nunique() != 1:
            raise ValueError(f'One nonempty sequence per CSV required: {path}')
        frames.append(frame)
    full = pd.concat(frames, ignore_index=True)
    if full[['sequence','frame_idx']].duplicated().any():
        raise ValueError('Overlapping or duplicate sequence/frame rows')
    return full


def evaluate(args: argparse.Namespace) -> None:
    """Compare image, IMU and fused models on identical held-out frames."""
    from sklearn.dummy import DummyRegressor, DummyClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                                 roc_auc_score, average_precision_score, brier_score_loss)

    tr = _load_group(list(map(Path, args.train)))
    te = _load_group(list(map(Path, args.test)))
    train_seqs = sorted(tr.sequence.unique())
    test_seqs = sorted(te.sequence.unique())
    if set(train_seqs) & set(test_seqs):
        raise ValueError('Training and test share a sequence; disjoint sequences required')
    horizon = args.horizon
    ycol = f's_{horizon}'
    necessary = set(IMAGE_FEATURES + IMU_FEATURES + [ycol, 'no_features', 'imu_200ms_valid', 'imu_500ms_valid'])
    for group in (tr, te):
        if not necessary.issubset(group.columns):
            raise ValueError(f'Missing feature/label columns: {sorted(necessary - set(group.columns))}')
    original_counts = {'train':len(tr), 'test':len(te)}

    def eligible(frame):
        return frame.loc[(frame.no_features == 0) & frame[ycol].notna() &
                         frame.imu_200ms_valid.eq(1) & frame.imu_500ms_valid.eq(1)].copy()
    tr = eligible(tr)
    te = eligible(te)
    if len(tr) < 20 or len(te) < 10:
        raise ValueError('Insufficient valid labeled frames with complete past IMU windows')
    if not (0 < args.risk_threshold < 1):
        raise ValueError('--risk-threshold must lie between 0 and 1')
    if not (0 < args.flag_fraction < 1):
        raise ValueError('--flag-fraction must lie between 0 and 1')
    yr = tr[ycol].to_numpy(float)
    yt = te[ycol].to_numpy(float)
    low_train = (yr < args.risk_threshold).astype(int)
    low_test = (yt < args.risk_threshold).astype(int)
    if np.unique(low_train).size < 2:
        raise ValueError('Training sequence lacks both low-survival and normal events; add training sequences')

    features = {'image': IMAGE_FEATURES, 'imu': IMU_FEATURES, 'image_plus_imu': IMAGE_FEATURES + IMU_FEATURES}
    results, per_frame = [], []
    for family in ('constant', *features):
        columns = features.get(family, [])
        if family == 'constant':
            xtrain = np.zeros((len(tr), 1))
            xtest = np.zeros((len(te), 1))
            reg = DummyRegressor(strategy='mean')
            classifier = DummyClassifier(strategy='prior')
        else:
            xtrain, xtest = tr[columns], te[columns]
            reg = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=10.0))
            classifier = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                                       LogisticRegression(C=1.0, max_iter=2000, random_state=0))
        reg.fit(xtrain, yr)
        reg_pred = np.clip(reg.predict(xtest), 0, 1)
        classifier.fit(xtrain, low_train)
        pcol = list(classifier.classes_) if family == 'constant' else list(classifier[-1].classes_)
        prisk = classifier.predict_proba(xtest)[:, pcol.index(1)]

        for seq in test_seqs:
            mask = te.sequence.to_numpy() == seq
            yy, pp, ee, rr = yt[mask], reg_pred[mask], low_test[mask], prisk[mask]
            rho = float(spearmanr(yy, pp).statistic) if np.unique(yy).size > 1 and np.unique(pp).size > 1 else float('nan')
            auc = float(roc_auc_score(ee,rr)) if np.unique(ee).size > 1 else float('nan')
            ap = float(average_precision_score(ee,rr)) if np.unique(ee).size > 1 else float('nan')
            k = max(1, int(np.ceil(len(rr)*args.flag_fraction)))
            flagged = np.argsort(-rr, kind='mergesort')[:k]
            recall = float(ee[flagged].sum() / ee.sum()) if ee.sum() else float('nan')
            precision = float(ee[flagged].mean())
            results.append(dict(model=family, target=ycol, test_sequence=seq, n_test=len(yy),
                                n_low=int(ee.sum()), event_prevalence=float(ee.mean()),
                                mae=float(mean_absolute_error(yy, pp)), rmse=float(np.sqrt(mean_squared_error(yy, pp))),
                                spearman=rho, risk_auroc=auc, risk_auprc=ap,
                                risk_brier=float(brier_score_loss(ee, rr)),
                                flagged_fraction=k/len(ee), flagged_precision=precision,
                                flagged_recall=recall))
        per_frame.extend(dict(sequence=seq, frame_idx=int(fi), actual=float(real),
                              predicted_survival=float(p), true_low_event=int(event), predicted_low_risk=float(risk),
                              horizon=horizon, model=family)
                         for seq,fi,real,p,event,risk in zip(te.sequence,te.frame_idx,yt,reg_pred,low_test,prisk))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out/'ablation_metrics.csv', index=False)
    pd.DataFrame(per_frame).to_csv(out/'ablation_predictions.csv', index=False)
    (out/'evaluation_manifest.json').write_text(json.dumps(dict(
        train=list(map(str,args.train)),test=list(map(str,args.test)),
        train_sequences=train_seqs,test_sequences=test_seqs,
        input_rows=original_counts, train_eligible=len(tr),test_eligible=len(te),
        horizon=horizon, risk_threshold=args.risk_threshold,flag_fraction=args.flag_fraction,
        risk_definition=f'{ycol} < {args.risk_threshold}',
        model_groups=features, imputer='train_median',normalizer='train_standard_scaler',
        ridge_alpha=10.0,logistic_C=1.0,model_selection='none',
        notes='All model families evaluated on same full-IMU test frames; no target leakage.'),
        ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(results).round(4).to_string(index=False))


def main():
    parser=argparse.ArgumentParser(description='EXP003: past-only IMU + current-frame image ablations')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('augment',help='Add past IMU-window predictors to existing EXP002 labels')
    p.add_argument('--labels',required=True,type=Path)
    p.add_argument('--imu-csv',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--max-gap-s',type=float,default=0.03)
    p.set_defaults(func=augment)
    p=sub.add_parser('evaluate',help='Whole-sequence heldout image-vs-IMU-vs-fusion')
    p.add_argument('--train',required=True,nargs='+',type=Path)
    p.add_argument('--test',required=True,nargs='+',type=Path)
    p.add_argument('--horizon',type=int,choices=(5,10),default=5)
    p.add_argument('--risk-threshold',type=float,default=0.7)
    p.add_argument('--flag-fraction',type=float,default=0.10)
    p.add_argument('--output-dir',required=True,type=Path)
    p.set_defaults(func=evaluate)
    args=parser.parse_args()
    args.func(args)


if __name__=='__main__':
    main()
