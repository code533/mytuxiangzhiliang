"""EXP004-B.1: threshold sensitivity and multi-cause geometry diagnostics.

Only EXP004-B GT-evaluated matches have saved GT Sampson errors. Increasing
the baseline requirement is legitimate; lowering below the original limit
requires re-running EXP004-B from original EuRoC GT and camera files.
Categories are *descriptive*, not causal diagnoses or VIO update labels.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_BASELINES=(.01,.02,.05,.10)
DEFAULT_EPI=(.5,1.,1.5,2.,3.)


def load_outputs(dirs):
    frames=[];points=[];manifests=[]
    for root in map(Path,dirs):
        f=pd.read_csv(root/"gt_frame_geometry.csv",
                      dtype={"start_t_ns":"string","end_t_ns":"string"})
        m=pd.read_csv(root/"gt_match_geometry.csv")
        meta=json.loads((root/"manifest.json").read_text(encoding="utf8"))
        name=Path(meta["sequence"]).name
        if f.empty or set(f.sequence)!={name} or not set(m.sequence).issubset({name}):
            raise ValueError(f"EXP004-B sequence/manifest mismatch: {root}")
        if f.duplicated(["sequence","frame_idx","horizon"]).any() or m.duplicated(
            ["sequence","frame_idx","horizon","feature_id"]).any():
            raise ValueError(f"Duplicate frame/horizon/feature records: {root}")
        frames.append(f);points.append(m);manifests.append(meta)
    return pd.concat(frames,ignore_index=True),pd.concat(points,ignore_index=True),manifests


def check_inputs(frame,points,manifests):
    cf={"sequence","frame_idx","horizon","exp004a_status","gt_status",
        "survival_rate","n_survive","n_initial","baseline_m",
        "gt_inlier_ratio_survivors","self_ransac_inlier_ratio"}
    cp={"sequence","frame_idx","horizon","gt_status","gt_sampson_px","self_ransac_inlier"}
    if cf-set(frame.columns) or cp-set(points.columns):
        raise ValueError("EXP004-B CSV missing required columns")
    source_base={float(m["min_baseline_m"]) for m in manifests}
    if len(source_base)!=1:
        raise ValueError("Input runs used different minimum GT baselines")
    original=next(iter(source_base))
    evaluated=frame[frame.gt_status.eq("gt_evaluated")]
    match=points[points.gt_status.eq("gt_evaluated")]
    if evaluated.empty or match.empty or not np.isfinite(match.gt_sampson_px.to_numpy(float)).all():
        raise ValueError("Need nonempty GT-evaluated frame/point residuals")
    keys=["sequence","frame_idx","horizon"]
    size=match.groupby(keys).size()
    index=evaluated.set_index(keys)
    if not index.index.isin(size.index).all() or not np.array_equal(
        index.n_survive.to_numpy(int),size.loc[index.index].to_numpy(int)):
        raise ValueError("Mismatch between saved GT-matched features and survival counts")
    if (evaluated.baseline_m < original-1e-10).any():
        raise ValueError("GT-evaluated frames contradict original baseline threshold")
    return original


def sweep(frame,points,baselines,epi_thresholds,original_base):
    baselines=np.asarray(baselines,float)
    epis=np.asarray(epi_thresholds,float)
    if np.any(baselines<original_base-1e-12):
        raise ValueError("Cannot lower baseline below source run without original GT")
    if np.any(baselines<=0) or np.any(epis<=0):
        raise ValueError("All thresholds must be positive")
    cols=["sequence","frame_idx","horizon"]
    evalf=frame[frame.gt_status.eq("gt_evaluated")].copy()
    match=points[points.gt_status.eq("gt_evaluated")]
    report=[];perframe=[]
    for ep in epis:
        score=match.assign(inlier=lambda d:d.gt_sampson_px<=ep).groupby(cols,as_index=False).agg(
            n_matches=("inlier","size"),n_inliers=("inlier","sum"))
        score["gt_ratio"]=score.n_inliers/score.n_matches
        joint=evalf.merge(score,on=cols,how="left",validate="one_to_one")
        if joint.gt_ratio.isna().any():
            raise ValueError("Missing residual-based GT evaluation")
        for b in baselines:
            g=joint[joint.baseline_m>=b].copy()
            g["epi_threshold_px"]=ep
            g["baseline_min_m"]=b
            perframe.append(g[cols+["epi_threshold_px","baseline_min_m",
                                     "n_matches","n_inliers","gt_ratio"]])
            for (sequence,k),allf in frame.groupby(["sequence","horizon"]):
                complete=allf[allf.gt_status.ne("tail_censored")]
                missing=complete[~complete.gt_status.isin(["gt_evaluated","low_gt_baseline"])]
                low=complete[complete.gt_status.isin(["gt_evaluated","low_gt_baseline"])
                             &complete.baseline_m.lt(b)]
                v=g[(g.sequence==sequence)&(g.horizon==k)]
                report.append(dict(sequence=sequence,horizon=int(k),
                    epi_threshold_px=float(ep),baseline_min_m=float(b),
                    n_complete=len(complete),n_gt_unavailable=len(missing),
                    n_low_baseline=len(low),n_evaluable=len(v),
                    evaluable_fraction=len(v)/len(complete),
                    mean_frame_gt_inlier=float(v.gt_ratio.mean()) if len(v) else np.nan,
                    pooled_point_gt_inlier=float(v.n_inliers.sum()/v.n_matches.sum()) if len(v) else np.nan,
                    n_self_high_gt_low=int(((v.self_ransac_inlier_ratio>=.8)&
                                            (v.gt_ratio<.6)).sum()),
                    n_self_pose_fail_gt_good=int(((v.exp004a_status=="insufficient_positive_depth")&
                                                  (v.gt_ratio>=.8)).sum())))
    return pd.DataFrame(report),pd.concat(perframe,ignore_index=True)


def classify(frame,points,epi_px):
    keys=["sequence","frame_idx","horizon"]
    pm=(points[points.gt_status.eq("gt_evaluated")]
          .assign(good=lambda x:x.gt_sampson_px<=epi_px)
          .groupby(keys).good.mean().rename("independent_gt_ratio").reset_index())
    merged=frame.merge(pm,on=keys,how="left",validate="one_to_one")
    labels=[]
    for row in merged.itertuples(index=False):
        if row.gt_status=="tail_censored":
            label="tail_censored"
        elif row.gt_status in ("gt_outside_time_range","gt_interpolation_gap"):
            label="missing_gt"
        elif row.gt_status=="low_gt_baseline":
            label="gt_low_baseline"
        elif row.gt_status!="gt_evaluated":
            label="other_unresolved"
        elif pd.isna(row.independent_gt_ratio):
            label="missing_gt_residuals"
        elif row.self_ransac_inlier_ratio>=.8 and row.independent_gt_ratio<.6:
            label="self_ransac_gt_conflict"
        elif row.independent_gt_ratio<.6:
            label="gt_epipolar_low_consistency"
        elif (row.exp004a_status=="insufficient_positive_depth"
              and row.independent_gt_ratio>=.8):
            label="self_pose_fail_gt_compatible"
        elif row.independent_gt_ratio>=.8:
            label="gt_epi_compatible"
        else:
            label="intermediate_gt_consistency"
        labels.append(label)
    merged["failure_class"]=labels
    return merged


def analyze(args):
    frame,points,manifests=load_outputs(args.input_dirs)
    source_min=check_inputs(frame,points,manifests)
    if args.reference_epi not in args.epi_thresholds:
        raise ValueError("--reference-epi must be included in --epi-thresholds")
    sensitivity,perframe=sweep(frame,points,args.baselines,args.epi_thresholds,source_min)
    labeled=classify(frame,points,args.reference_epi)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    sensitivity.to_csv(out/"threshold_sensitivity.csv",index=False)
    perframe.to_csv(out/"frame_threshold_sensitivity.csv",index=False)
    labeled.to_csv(out/"failure_classification.csv",index=False)
    labeled.groupby(["sequence","horizon","failure_class"]).size().rename("n").reset_index().to_csv(
        out/"failure_class_counts.csv",index=False)
    evalf=labeled[labeled.gt_status.eq("gt_evaluated")].copy()
    evalf["self_minus_gt"]=evalf.self_ransac_inlier_ratio-evalf.independent_gt_ratio
    evalf.nlargest(35,"self_minus_gt").to_csv(
        out/"largest_self_vs_gt_disagreements.csv",index=False)
    labeled[(labeled.survival_rate>=.8)&labeled.failure_class.isin(
        ["gt_low_baseline","self_ransac_gt_conflict",
         "self_pose_fail_gt_compatible","gt_epipolar_low_consistency"])].to_csv(
        out/"high_survival_warning_cases.csv",index=False)
    (out/"manifest.json").write_text(json.dumps(dict(experiment="EXP004-B.1",
        input_dirs=list(map(str,args.input_dirs)),source_baseline_min_m=source_min,
        baselines_m=args.baselines,epi_thresholds_px=args.epi_thresholds,
        reference_epi_px=args.reference_epi,
        limitations="Do not evaluate points with unavailable GT or from original low-baseline frames. Classes are descriptive, not VIO update labels."),indent=2,ensure_ascii=False),encoding="utf8")
    print(labeled.groupby(["sequence","horizon","failure_class"]).size().to_string())


def main():
    pa=argparse.ArgumentParser(description="EXP004-B.1: independent GT geometry sensitivity")
    pa.add_argument("--input-dirs",type=Path,nargs="+",required=True)
    pa.add_argument("--output-dir",type=Path,required=True)
    pa.add_argument("--baselines",nargs="+",type=float,default=list(DEFAULT_BASELINES))
    pa.add_argument("--epi-thresholds",nargs="+",type=float,default=list(DEFAULT_EPI))
    pa.add_argument("--reference-epi",type=float,default=1.5)
    analyze(pa.parse_args())


if __name__=="__main__":
    main()
