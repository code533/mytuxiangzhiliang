import json
from argparse import Namespace
import numpy as np
import pandas as pd
import pytest
from vio_iqa.exp003 import IMAGE_FEATURES, IMU_FEATURES
from vio_iqa.exp005a import merge_sequence, build, PREDICTORS


def fixture(tmp_path,seq='S',gt_missing=False):
    base=1403636579763555584
    feats=[]
    for i in range(40):
        row=dict(sequence=seq,frame_idx=i,t_ns=str(base+i*50000000),
                 no_features=0,n_initial=240,imu_200ms_valid=1,
                 imu_500ms_valid=1,s_5=.9,s_10=(np.nan if i==20 else .8))
        for j,f in enumerate(PREDICTORS):row[f]=j+.1+i*.01
        feats.append(row)
    feat=tmp_path/(seq+'_f.csv')
    pd.DataFrame(feats).to_csv(feat,index=False)
    rows=[]
    for frame_idx in (0,5,10,15,20):
        for h in (5,10):
            status=('gt_outside_time_range' if gt_missing and frame_idx==0 else
                    'tail_censored' if frame_idx==20 and h==10 else
                    'low_gt_baseline' if frame_idx==5 else 'gt_evaluated')
            rows.append(dict(sequence=seq,frame_idx=frame_idx,horizon=h,
                  gt_status=status,baseline_m=(np.nan if status.startswith('gt_outside') or status=='tail_censored' else
                                                .004 if frame_idx==5 else .15),
                  gt_inlier_ratio_survivors=(.55 if frame_idx==10 else .9) if status=='gt_evaluated' else np.nan,
                  n_initial=240,n_survive=np.nan if status=='tail_censored' else int(240*(.9 if h==5 else .8)),
                  survival_rate=np.nan if status=='tail_censored' else (.9 if h==5 else .8),
                  start_t_ns=str(base+frame_idx*50000000),
                  gt_parallax_deg=.3,gt_positive_depth_fraction=1.))
    geom=tmp_path/(seq+'_g.csv')
    pd.DataFrame(rows).to_csv(geom,index=False)
    return feat,geom


def test_label_masks_and_horizon_specific(tmp_path):
    f,g=fixture(tmp_path,gt_missing=True)
    df=merge_sequence(f,g)
    assert len(df)==10
    one=df[df.frame_idx==5]
    assert one.label_low_baseline.eq(1).all()
    assert one.label_low_epipolar.isna().all()
    assert one.epipolar_label_status.eq('low_baseline_excluded').all()
    two=df[df.frame_idx==10]
    assert two.label_low_baseline.eq(0).all()
    assert two.label_low_epipolar.eq(1).all()
    assert df[(df.frame_idx==0)].label_low_baseline.isna().all()
    assert df[(df.frame_idx==20)&(df.horizon==10)].baseline_label_status.eq('tail_censored').all()
    assert all(c in df.columns for c in IMAGE_FEATURES+IMU_FEATURES)


def test_never_use_future_metrics_as_predictors():
    assert not set(['survival_rate','s_5','s_10','baseline_m','gt_parallax_deg',
                    'gt_inlier_ratio_survivors'])&set(PREDICTORS)


def test_no_float_timestamps_or_nearest_frame_join(tmp_path):
    f,g=fixture(tmp_path)
    d=pd.read_csv(g,dtype={'start_t_ns':'string'})
    d.loc[0,'start_t_ns']='1.4036365797635556e+18';d.to_csv(g,index=False)
    with pytest.raises(ValueError,match='exact integer nanoseconds'):
        merge_sequence(f,g)
    f,g=fixture(tmp_path)
    d=pd.read_csv(g,dtype={'start_t_ns':'string'})
    d.loc[0,'start_t_ns']=str(int(d.loc[0,'start_t_ns'])+1)
    d.to_csv(g,index=False)
    with pytest.raises(ValueError,match='timestamp mismatch'):
        merge_sequence(f,g)


def test_survival_mismatch_is_hard_error(tmp_path):
    f,g=fixture(tmp_path)
    x=pd.read_csv(g);x.loc[0,'survival_rate']=.02;x.to_csv(g,index=False)
    with pytest.raises(ValueError,match='survival mismatch'):
        merge_sequence(f,g)


def test_build_multi_sequence_outputs(tmp_path):
    a,b=fixture(tmp_path,'A');c,d=fixture(tmp_path,'B')
    out=tmp_path/'out'
    build(Namespace(features=[c,a],geometry=[b,d],output_dir=out))
    manifest=json.loads((out/'dataset_manifest.json').read_text())
    assert manifest['sequences']==['A','B']
    assert manifest['feature_columns']==IMAGE_FEATURES+IMU_FEATURES
    s=pd.read_csv(out/'dataset_summary.csv')
    assert len(s)==4
    assert s.label_low_baseline_n_events.eq(1).all()
    assert s.label_low_epipolar_n_events.eq(1).all()
    assert (out/'A'/'geometry_prediction_labels.csv').exists()
    assert (out/'feature_event_associations.csv').exists()


def test_require_sequence_pairs(tmp_path):
    a,b=fixture(tmp_path,'A');c,d=fixture(tmp_path,'B')
    with pytest.raises(ValueError,match='Sequence pairing mismatch'):
        build(Namespace(features=[a],geometry=[d],output_dir=tmp_path/'out'))
    with pytest.raises(ValueError,match='Duplicate features'):
        build(Namespace(features=[a,a],geometry=[b,b],output_dir=tmp_path/'out'))


def test_out_of_range_gt_inliers_rejected(tmp_path):
    f,g=fixture(tmp_path)
    d=pd.read_csv(g)
    d.loc[0,'gt_inlier_ratio_survivors']=1.1;d.to_csv(g,index=False)
    with pytest.raises(ValueError,match='Out of range'):
        merge_sequence(f,g)
