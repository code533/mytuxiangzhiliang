import json
from argparse import Namespace
import numpy as np
import pandas as pd
import pytest
from vio_iqa.exp004b1 import check_inputs, sweep, classify, analyze


def source(tmp_path, name):
    folder=tmp_path/name;folder.mkdir()
    rows=[];points=[]
    for i, (status,b) in enumerate([
        ("gt_evaluated",.03),("gt_evaluated",.08),
        ("low_gt_baseline",.003),("gt_outside_time_range",np.nan),
        ("tail_censored",np.nan),
    ]):
        row=dict(sequence=name,frame_idx=i*5,horizon=5,
                 gt_status=status,exp004a_status="estimated",
                 survival_rate=.9,n_survive=10,n_initial=12,
                 baseline_m=b,gt_inlier_ratio_survivors=.9,
                 self_ransac_inlier_ratio=.92)
        rows.append(row)
        if status=="gt_evaluated":
            for fid in range(10):
                points.append(dict(sequence=name,frame_idx=i*5,horizon=5,
                    feature_id=fid,gt_status=status,
                    gt_sampson_px=float(fid)/10,
                    self_ransac_inlier=1))
    pd.DataFrame(rows).to_csv(folder/"gt_frame_geometry.csv",index=False)
    pd.DataFrame(points).to_csv(folder/"gt_match_geometry.csv",index=False)
    (folder/"manifest.json").write_text(json.dumps(dict(sequence=f"/EuRoC/{name}",
        min_baseline_m=.01)))
    return folder


def test_sweep_counts_and_coverage(tmp_path):
    folder=source(tmp_path,"Seq")
    from vio_iqa.exp004b1 import load_outputs
    f,p,m=load_outputs([folder])
    bound=check_inputs(f,p,m)
    assert bound==.01
    r,per=sweep(f,p,[.01,.05],[.5,1.5],bound)
    assert len(r)==4
    assert r[(r.baseline_min_m==.01)&(r.epi_threshold_px==1.5)].iloc[0].n_evaluable==2
    assert r[(r.baseline_min_m==.05)&(r.epi_threshold_px==1.5)].iloc[0].n_evaluable==1
    assert r.n_gt_unavailable.eq(1).all()
    assert r.n_complete.eq(4).all()
    assert len(per)==6


def test_never_lower_original_baseline(tmp_path):
    from vio_iqa.exp004b1 import load_outputs
    f,p,m=load_outputs([source(tmp_path,"Seq")])
    with pytest.raises(ValueError,match="Cannot lower"):
        sweep(f,p,[.005],[1.5],.01)


def test_classes_keep_unknown_separate(tmp_path):
    from vio_iqa.exp004b1 import load_outputs
    f,p,_=load_outputs([source(tmp_path,"Seq")])
    out=classify(f,p,1.5)
    assert set(out.failure_class)=={"gt_epi_compatible","gt_low_baseline",
        "missing_gt","tail_censored"}
    assert out[out.gt_status=="gt_outside_time_range"].independent_gt_ratio.isna().all()


def test_no_invented_gt_errors(tmp_path):
    from vio_iqa.exp004b1 import load_outputs
    folder=source(tmp_path,"Seq")
    f,p,m=load_outputs([folder])
    p.loc[0,"gt_sampson_px"]=np.nan
    with pytest.raises(ValueError,match="GT-evaluated"):
        check_inputs(f,p,m)


def test_integration_writes_manifest_and_outputs(tmp_path):
    a=source(tmp_path,"First")
    b=source(tmp_path,"Second")
    out=tmp_path/"output"
    analyze(Namespace(input_dirs=[a,b],output_dir=out,
        baselines=[.01,.02,.05],epi_thresholds=[.5,1.5,3.0],reference_epi=1.5))
    for name in ("threshold_sensitivity.csv","failure_classification.csv",
                 "failure_class_counts.csv","high_survival_warning_cases.csv","manifest.json"):
        assert (out/name).exists()
    assert pd.read_csv(out/"failure_classification.csv").shape[0]==10
