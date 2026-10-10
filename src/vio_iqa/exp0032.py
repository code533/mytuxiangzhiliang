"""EXP003.2: sequence-heldout IMU compression, causal trends and risk calibration.

Prediction inputs use I_t and IMU history up to t only. Train sequences fit
models, validation sequences fit calibration/select model, test is evaluation only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             mean_absolute_error, mean_squared_error, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from vio_iqa.exp003 import IMAGE_FEATURES, IMU_FEATURES

COMPACT_IMU = ["imu_200ms_gyro_mean", "imu_500ms_acc_norm_std"]
TREND_FEATURES = ["imu_gyro_mean_recent_minus_long", "imu_acc_std_recent_minus_long"]
GROUPS = {
    "constant": [],
    "image": list(IMAGE_FEATURES),
    "imu_full": list(IMU_FEATURES),
    "imu_compact": list(COMPACT_IMU),
    "image_imu_full": list(IMAGE_FEATURES) + list(IMU_FEATURES),
    "image_imu_compact": list(IMAGE_FEATURES) + list(COMPACT_IMU),
    "image_imu_compact_trend": list(IMAGE_FEATURES) + list(COMPACT_IMU) + list(TREND_FEATURES),
}
EPS = 1e-6


def causal_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Signed 200ms-minus-500ms historical window contrasts, not derivatives."""
    df = df.copy()
    needed = ["imu_200ms_gyro_mean", "imu_500ms_gyro_mean",
              "imu_200ms_acc_norm_std", "imu_500ms_acc_norm_std"]
    missing = sorted(set(needed) - set(df.columns))
    if missing:
        raise ValueError(f"Missing EXP003 features: {missing}")
    df[TREND_FEATURES[0]] = df[needed[0]] - df[needed[1]]
    df[TREND_FEATURES[1]] = df[needed[2]] - df[needed[3]]
    return df


def read_seq(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"t_ns": "int64"})
    must = {"sequence", "frame_idx", "t_ns", "no_features", "s_5", "s_10",
            "imu_200ms_valid", "imu_500ms_valid"} | set(IMAGE_FEATURES) | set(IMU_FEATURES)
    absent = sorted(must - set(df.columns))
    if absent:
        raise ValueError(f"{path}: missing columns: {absent}")
    if df.empty or df.sequence.nunique() != 1:
        raise ValueError("One nonempty sequence per feature CSV required")
    if df.frame_idx.duplicated().any() or not df.t_ns.is_monotonic_increasing:
        raise ValueError("Frame index must be unique and timestamp monotonic")
    return causal_derived_features(df)


def read_split(paths: list[Path]) -> pd.DataFrame:
    df = pd.concat([read_seq(p) for p in paths], ignore_index=True)
    if df[["sequence", "frame_idx"]].duplicated().any():
        raise ValueError("Duplicate sequence/frame across CSVs")
    return df


def sha256(path: Path) -> str:
    hh = hashlib.sha256()
    with path.open("rb") as fh:
        for buf in iter(lambda: fh.read(1048576), b""):
            hh.update(buf)
    return hh.hexdigest()


def check_data(train, validation, test, horizon, threshold):
    all_paths = train + validation + test
    if len({str(p.resolve()) for p in all_paths}) != len(all_paths):
        raise ValueError("Train/validation/test input CSV paths must be distinct")
    hashes = [sha256(p) for p in all_paths]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Identical CSV files reused under different paths")
    frames = (read_split(train), read_split(validation), read_split(test))
    names = [set(df.sequence) for df in frames]
    if (names[0] & names[1]) or (names[0] & names[2]) or (names[1] & names[2]):
        raise ValueError("Train, validation and test must use disjoint whole sequences")
    target = f"s_{horizon}"
    prepared = []
    counts = []
    for name, df in zip(("train", "validation", "test"), frames):
        valid = ((df.no_features == 0) & df[target].notna() &
                 df.imu_200ms_valid.eq(1) & df.imu_500ms_valid.eq(1))
        eligible = df[valid].copy()
        if len(eligible) < 20:
            raise ValueError(f"{name} has fewer than 20 valid labels with complete IMU")
        low = (eligible[target] < threshold).astype(int)
        if low.nunique() < 2:
            raise ValueError(f"{name} needs both low-survival and normal events")
        if not np.isfinite(eligible[IMAGE_FEATURES + IMU_FEATURES].to_numpy(float)).all():
            raise ValueError(f"{name} contains nonfinite predictors in complete IMU rows")
        counts.append(dict(split=name, n_input=len(df), n_eligible=len(eligible),
                           n_low=int(low.sum()), event_rate=float(low.mean()),
                           n_censored_or_invalid=len(df)-len(eligible)))
        prepared.append(eligible)
    return prepared, names, counts, dict(zip(map(str, all_paths), hashes))


def monotone_platt_fit(uncalibrated_p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Validation-only monotonically nondecreasing probability calibration."""
    x = logit(np.clip(np.asarray(uncalibrated_p, dtype=float), EPS, 1-EPS))
    yy = np.asarray(y, dtype=float)
    def objective(params):
        a, b = params
        z = a*x+b
        return np.mean(np.logaddexp(0,z) - yy*z) + .01*(a-1)**2 + .01*b**2
    fit = minimize(objective, x0=[1.,0.], bounds=[(0.,20.),(-12.,12.)], method="L-BFGS-B")
    if not fit.success:
        raise RuntimeError(f"Calibration failed: {fit.message}")
    return float(fit.x[0]), float(fit.x[1])


def apply_platt(p: np.ndarray, ab: tuple[float, float]) -> np.ndarray:
    return expit(ab[0]*logit(np.clip(np.asarray(p), EPS, 1-EPS))+ab[1])


def tie_aware_top_fraction(y: np.ndarray, risk: np.ndarray, fraction: float) -> tuple[float, float]:
    """Expected top-fraction precision/recall with randomized equal-risk ties.

    Constant predictor's expected recall equals the flag fraction, not the
    accidental recall of the initial contiguous video frames.
    """
    labels = np.asarray(y, dtype=int)
    score = np.asarray(risk, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    sorted_scores, sorted_y = score[order], labels[order]
    quota, remaining, expected_tp = fraction*len(labels), fraction*len(labels), 0.
    start = 0
    while start < len(labels) and remaining > 0:
        end = start+1
        while end < len(labels) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        taken = min(remaining, end-start)
        expected_tp += taken/(end-start)*np.sum(sorted_y[start:end])
        remaining -= taken
        start = end
    return float(expected_tp/quota), float(expected_tp/labels.sum()) if labels.sum() else float("nan")


def risk_metrics(y: np.ndarray, p: np.ndarray, flag_fraction: float) -> dict:
    labels = np.asarray(y, int)
    score = np.asarray(p, float)
    precision, recall = tie_aware_top_fraction(labels, score, flag_fraction)
    return dict(auroc=float(roc_auc_score(labels, score)),
                auprc=float(average_precision_score(labels, score)),
                brier=float(brier_score_loss(labels, score)),
                logloss=float(log_loss(labels, np.clip(score, EPS, 1-EPS), labels=[0,1])),
                risk_mean=float(score.mean()), flagged_precision=precision,
                flagged_recall=recall, flagged_fraction=flag_fraction)


def _new_models(cols: list[str]):
    if not cols:
        return DummyRegressor(strategy="mean"), DummyClassifier(strategy="prior")
    return (make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)),
            make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(C=1.0, max_iter=2000, random_state=0)))


def evaluate(args: argparse.Namespace):
    train = list(map(Path, args.train))
    val = list(map(Path, args.validation))
    test = list(map(Path, args.test))
    if args.horizon not in (5,10) or not (0 < args.risk_threshold < 1) or not (0 < args.flag_fraction < 1):
        raise ValueError("Invalid horizon, risk threshold or flag fraction")
    (tr, va, te), names, split_counts, checksums = check_data(train, val, test, args.horizon, args.risk_threshold)
    ycol = f"s_{args.horizon}"
    yt, yv, ye = (frame[ycol].to_numpy(float) for frame in (tr,va,te))
    event_t, event_v, event_e = ((yy < args.risk_threshold).astype(int) for yy in (yt,yv,ye))
    rows, preds, val_scores, calibrations = [], [], [], {}
    for name, columns in GROUPS.items():
        Xt = tr[columns] if columns else np.zeros((len(tr),1))
        Xv = va[columns] if columns else np.zeros((len(va),1))
        Xe = te[columns] if columns else np.zeros((len(te),1))
        reg, clf = _new_models(columns)
        reg.fit(Xt, yt)
        clf.fit(Xt, event_t)
        predicted_survival = np.clip(reg.predict(Xe), 0, 1)
        classes = clf[-1].classes_ if columns else clf.classes_
        positive_col = int(np.flatnonzero(classes == 1)[0])
        pval = clf.predict_proba(Xv)[:,positive_col]
        raw = clf.predict_proba(Xe)[:,positive_col]
        if name == "constant":
            a, b = 0., float(logit(np.clip(event_v.mean(), EPS, 1-EPS)))
        else:
            a, b = monotone_platt_fit(pval, event_v)
        calibrated = apply_platt(raw, (a,b))
        calibrations[name] = dict(a=a, b=b, validation_event_rate=float(event_v.mean()))
        val_scores.append(dict(model=name, val_auprc=float(average_precision_score(event_v,pval)),
                               val_brier=float(brier_score_loss(event_v,pval)),
                               val_event_rate=float(event_v.mean())))
        for seq in sorted(te.sequence.unique()):
            keep = te.sequence.to_numpy() == seq
            actual, pred, low = ye[keep], predicted_survival[keep], event_e[keep]
            rho = float(spearmanr(actual,pred).statistic) if np.unique(actual).size>1 and np.unique(pred).size>1 else float("nan")
            base = dict(model=name,target=ycol,test_sequence=seq,n_test=len(actual),
                        n_low=int(low.sum()),event_rate=float(low.mean()),
                        mae=float(mean_absolute_error(actual,pred)),
                        rmse=float(np.sqrt(mean_squared_error(actual,pred))),spearman=rho)
            for label, scores in (("raw",raw[keep]), ("calibrated",calibrated[keep])):
                rows.append(dict(**base, probability_type=label,
                                 **risk_metrics(low,scores,args.flag_fraction)))
        preds.extend(dict(sequence=seq, frame_idx=int(fi), model=name,
                          horizon=args.horizon, actual=float(actual), predicted_survival=float(pred),
                          true_low_event=int(low), raw_risk=float(prisk), calibrated_risk=float(pcal))
                     for seq,fi,actual,pred,low,prisk,pcal in
                     zip(te.sequence,te.frame_idx,ye,predicted_survival,event_e,raw,calibrated))
    validation = pd.DataFrame(val_scores)
    # Selection is mathematically independent of all test outcomes.
    selected = validation.sort_values(["val_auprc","val_brier","model"],
                                      ascending=[False,True,True]).iloc[0]["model"]
    out = Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/"heldout_ablation.csv", index=False)
    pd.DataFrame(preds).to_csv(out/"heldout_predictions.csv", index=False)
    validation.to_csv(out/"validation_model_selection.csv", index=False)
    (out/"selected_from_validation.json").write_text(json.dumps(dict(
        selected_model=selected, selection_metric="validation_AUPRC",
        tie_break="validation_Brier", selected_using_validation_only=True,
        normalization="fit_on_training_only",
        probability_calibration="nonnegative_slope_Platt_on_validation_only",
        calibration_params=calibrations),indent=2),encoding="utf-8")
    (out/"protocol_manifest.json").write_text(json.dumps(dict(
        experiment="EXP003.2", input_files_sha256=checksums,
        train_sequences=sorted(names[0]),validation_sequences=sorted(names[1]),
        test_sequences=sorted(names[2]),train_paths=list(map(str,train)),
        validation_paths=list(map(str,val)),test_paths=list(map(str,test)),
        sample_counts=split_counts,horizon=args.horizon,threshold=args.risk_threshold,
        flag_fraction=args.flag_fraction, model_groups=GROUPS,
        ridge_alpha=10.,logistic_C=1.,calibration_regularization=.01,
        note="Compact groups are pre-specified using previously seen MH_01/MH_05; only newly untouched sequence sets support unbiased validation/test.",
        label="EXP002 FB KLT survival proxy, not real VIO update reliability"),
        indent=2,ensure_ascii=False),encoding="utf-8")
    print("Selected by validation AUPRC:",selected)
    print(pd.DataFrame(rows).query('probability_type=="calibrated"')[
        ["model","test_sequence","mae","rmse","auprc","brier","flagged_recall"]
    ].round(4).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="EXP003.2: independent train/validation/test IMU ablations")
    parser.add_argument("--train", nargs="+",type=Path,required=True)
    parser.add_argument("--validation",nargs="+",type=Path,required=True)
    parser.add_argument("--test",nargs="+",type=Path,required=True)
    parser.add_argument("--horizon",choices=(5,10),type=int,default=5)
    parser.add_argument("--risk-threshold",type=float,default=.7)
    parser.add_argument("--flag-fraction",type=float,default=.1)
    parser.add_argument("--output-dir",type=Path,required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
