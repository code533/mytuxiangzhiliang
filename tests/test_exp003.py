import json
from argparse import Namespace
import numpy as np
import pandas as pd
import pytest
from vio_iqa.exp003 import IMAGE_FEATURES, IMU_FEATURES, augment, evaluate, imu_features_at_t, read_euroc_imu


def make_imu(tmp_path, name, future_bomb=False, gap=False):
    folder = tmp_path / name / "mav0" / "imu0"
    folder.mkdir(parents=True)
    rows = ["#timestamp [ns],gx,gy,gz,ax,ay,az"]
    for i in range(900):
        if gap and 175 <= i <= 250:
            continue
        stamp = 1_000_000_000 + i*5_000_000
        gyro = .15 + .15*np.sin(i/20)
        if future_bomb and i >= 800:
            gyro = 9999.0
        rows.append(f"{stamp},{gyro},0,0,0,0,9.81")
    path = folder / "data.csv"
    path.write_text("\n".join(rows)+"\n")
    return path


def make_labels(tmp_path, name, count=52):
    path = tmp_path / name / "survival_labels.csv"
    path.parent.mkdir(parents=True,exist_ok=True)
    records=[]
    for i in range(count):
        event = i%6 in (0,1)
        row=dict(sequence=name,frame_idx=i,t_ns=1_800_000_000+i*50_000_000,
                 s_5=.55 if event else .92,s_10=.4 if event else .83,no_features=0)
        for j,col in enumerate(IMAGE_FEATURES):
            row[col] = .5+j*.13+.1*np.sin(i/3)+(.15 if event else -.08)
        records.append(row)
    pd.DataFrame(records).to_csv(path,index=False)
    return path


def test_past_only_imu_unchanged_by_future_spike(tmp_path):
    a=read_euroc_imu(make_imu(tmp_path,"base"))
    b=read_euroc_imu(make_imu(tmp_path,"spike",future_bomb=True))
    times=np.array([1_600_000_000,2_000_000_000,3_500_000_000],dtype=np.int64)
    x=imu_features_at_t(times,a);y=imu_features_at_t(times,b)
    pd.testing.assert_frame_equal(x,y)
    assert x.imu_500ms_valid.eq(1).all()


def test_imu_gaps_and_missing_history_are_not_silently_filled(tmp_path):
    imu=read_euroc_imu(make_imu(tmp_path,"gap",gap=True))
    times=np.array([1_100_000_000,2_200_000_000,3_000_000_000],dtype=np.int64)
    feats=imu_features_at_t(times,imu)
    assert feats.imu_500ms_valid.tolist()==[0,0,1]
    assert np.isnan(feats.iloc[0].imu_500ms_gyro_mean)


def test_augment_preserves_labels(tmp_path):
    labels=make_labels(tmp_path,"MH_01_easy")
    imu=make_imu(tmp_path,"MH_01_easy")
    dest=tmp_path/"aug"
    augment(Namespace(labels=labels,imu_csv=imu,output_dir=dest,max_gap_s=.03))
    frame=pd.read_csv(dest/"survival_imu_features.csv")
    assert len(frame)==52 and set(IMAGE_FEATURES+IMU_FEATURES).issubset(frame.columns)
    assert frame.imu_500ms_valid.eq(1).all()
    manifest=json.loads((dest/"manifest.json").read_text())
    assert manifest["n_imu_complete"]==52
    assert manifest["time_policy"]=="past_only_imu_t_le_camera_t"


def build(tmp_path, gap=False):
    outputs=[]
    for name in ("MH_01_easy","MH_05_difficult"):
        labels=make_labels(tmp_path,name)
        imu=make_imu(tmp_path,name,gap=gap)
        dest=tmp_path/f"aug_{name}"
        augment(Namespace(labels=labels,imu_csv=imu,output_dir=dest,max_gap_s=.03))
        outputs.append(dest/"survival_imu_features.csv")
    return outputs


def test_heldout_ablation_and_split_guard(tmp_path):
    a,b=build(tmp_path)
    dest=tmp_path/"evaluation"
    params=dict(train=[a],test=[b],horizon=5,risk_threshold=.7,flag_fraction=.1,output_dir=dest)
    evaluate(Namespace(**params))
    score=pd.read_csv(dest/"ablation_metrics.csv")
    pred=pd.read_csv(dest/"ablation_predictions.csv")
    assert set(score.model)=={"constant","image","imu","image_plus_imu"}
    assert len(pred)==4*52 and score.risk_auroc.between(0,1).all()
    assert score.risk_brier.between(0,1).all()
    assert pred.predicted_low_risk.between(0,1).all()
    with pytest.raises(ValueError,match="share a sequence"):
        evaluate(Namespace(**{**params,"test":[a]}))


def test_incomplete_imu_filtered_equally_for_every_model(tmp_path):
    a,b=build(tmp_path,gap=True)
    out=tmp_path/"eval"
    evaluate(Namespace(train=[a],test=[b],horizon=10,risk_threshold=.7,flag_fraction=.1,output_dir=out))
    summary=json.loads((out/"evaluation_manifest.json").read_text())
    assert summary["train_eligible"]<52 and summary["test_eligible"]<52
    scored=pd.read_csv(out/"ablation_metrics.csv")
    assert scored.n_test.nunique()==1
    assert scored.n_test.iloc[0]==summary["test_eligible"]


def test_imu_timestamp_strictly_increasing(tmp_path):
    file=make_imu(tmp_path,"broken")
    contents=file.read_text().splitlines()
    contents[2],contents[3]=contents[3],contents[2]
    file.write_text("\n".join(contents)+"\n")
    with pytest.raises(ValueError,match="strictly increasing"):
        read_euroc_imu(file)
