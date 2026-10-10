import json
from argparse import Namespace
import numpy as np
import pandas as pd
import pytest
from vio_iqa.exp003 import IMAGE_FEATURES, IMU_FEATURES
from vio_iqa.exp0032 import (GROUPS,TREND_FEATURES,read_seq,evaluate,
                             tie_aware_top_fraction,monotone_platt_fit,apply_platt)


def source(tmp_path,name,seed=1,n=130,missing=False):
    rng=np.random.default_rng(seed)
    xx=rng.normal(size=n)
    bad=xx>.4
    s5=np.where(bad,.5,.91)+rng.normal(0,.01,n)
    data=dict(sequence=name,frame_idx=np.arange(n),
              t_ns=1_000_000_000+np.arange(n,dtype=np.int64)*50_000_000,
              s_5=s5,s_10=s5-.12,no_features=np.zeros(n,int),
              imu_200ms_valid=np.ones(n,int),imu_500ms_valid=np.ones(n,int))
    for j,k in enumerate(IMAGE_FEATURES):
        data[k]=3+xx*.15+rng.normal(0,.4,n)+.01*j
    for j,k in enumerate(IMU_FEATURES):
        data[k]=.8+xx*.04+rng.normal(0,.04,n)+j*.003
    df=pd.DataFrame(data)
    if missing:
        df.loc[:4,'imu_500ms_valid']=0
        for f in IMU_FEATURES:
            if '500ms' in f:
                df.loc[:4,f]=np.nan
    path=tmp_path/f'{name}.csv'
    df.to_csv(path,index=False)
    return path


def params(a,b,c,out,**kw):
    x=dict(train=[a],validation=[b],test=[c],horizon=5,
           risk_threshold=.7,flag_fraction=.1,output_dir=out)
    x.update(kw)
    return Namespace(**x)


def test_causal_trends(tmp_path):
    d=read_seq(source(tmp_path,'one'))
    assert set(TREND_FEATURES).issubset(d.columns)
    assert np.allclose(d[TREND_FEATURES[0]],d.imu_200ms_gyro_mean-d.imu_500ms_gyro_mean)
    assert np.allclose(d[TREND_FEATURES[1]],d.imu_200ms_acc_norm_std-d.imu_500ms_acc_norm_std)
    assert len(GROUPS['imu_compact'])==2


def test_tie_aware_constant():
    y=np.array([1]*15+[0]*85)
    p=np.ones(100)*.12
    precision,recall=tie_aware_top_fraction(y,p,.1)
    assert precision==pytest.approx(.15)
    assert recall==pytest.approx(.1)
    assert tie_aware_top_fraction(np.roll(y,20),p,.1)==pytest.approx((precision,recall))


def test_calibration_monotone():
    p=np.linspace(.03,.85,120)
    yy=(np.arange(120)%3==0).astype(int)
    ab=monotone_platt_fit(p,yy)
    z=apply_platt(p,ab)
    assert ab[0]>=0 and np.all(np.diff(z)>=-1e-10)
    assert z.min()>0 and z.max()<1


def test_three_way_evaluation(tmp_path):
    a=source(tmp_path,'Train',12);b=source(tmp_path,'Valid',8);c=source(tmp_path,'Test',17)
    out=tmp_path/'out'
    evaluate(params(a,b,c,out))
    score=pd.read_csv(out/'heldout_ablation.csv')
    pred=pd.read_csv(out/'heldout_predictions.csv')
    assert len(score)==len(GROUPS)*2
    assert len(pred)==130*len(GROUPS)
    assert score.auprc.between(0,1).all()
    assert score.flagged_recall.between(0,1).all()
    assert np.allclose(score[score.model=='constant'].flagged_recall.to_numpy(),.1)
    meta=json.loads((out/'protocol_manifest.json').read_text())
    sel=json.loads((out/'selected_from_validation.json').read_text())
    assert meta['train_sequences']==['Train']
    assert meta['validation_sequences']==['Valid']
    assert meta['test_sequences']==['Test']
    assert sel['selected_model'] in GROUPS


def test_no_split_overlap(tmp_path):
    a=source(tmp_path,'Train');b=source(tmp_path,'Valid',2);c=source(tmp_path,'Test',3)
    with pytest.raises(ValueError,match='distinct|disjoint'):
        evaluate(params(a,b,a,tmp_path/'bad1'))
    copy=tmp_path/'copy.csv';copy.write_bytes(a.read_bytes())
    with pytest.raises(ValueError,match='Identical CSV'):
        evaluate(params(a,b,copy,tmp_path/'bad2'))


def test_test_labels_cannot_influence_selection_or_prediction(tmp_path):
    a=source(tmp_path,'Train',12);b=source(tmp_path,'Valid',8);c=source(tmp_path,'Test',17)
    x=tmp_path/'run1';evaluate(params(a,b,c,x))
    changed=pd.read_csv(c)
    changed['s_5']=np.where(changed.s_5<.7,.91,.4)
    c2=tmp_path/'modified_test.csv';changed.to_csv(c2,index=False)
    y=tmp_path/'run2';evaluate(params(a,b,c2,y))
    assert json.loads((x/'selected_from_validation.json').read_text())==json.loads((y/'selected_from_validation.json').read_text())
    p=pd.read_csv(x/'heldout_predictions.csv')
    q=pd.read_csv(y/'heldout_predictions.csv')
    assert np.allclose(p.predicted_survival,q.predicted_survival)
    assert np.allclose(p.calibrated_risk,q.calibrated_risk)
    assert not np.allclose(p.actual,q.actual)


def test_incomplete_imu_excluded_equally(tmp_path):
    a=source(tmp_path,'Train',12,missing=True)
    b=source(tmp_path,'Valid',8,missing=True)
    c=source(tmp_path,'Test',17,missing=True)
    out=tmp_path/'out'
    evaluate(params(a,b,c,out,horizon=10))
    score=pd.read_csv(out/'heldout_ablation.csv')
    assert score.n_test.nunique()==1
    assert score.n_test.iloc[0]==125
