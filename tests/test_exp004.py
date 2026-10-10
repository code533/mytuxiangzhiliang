import json
from argparse import Namespace
import cv2
import numpy as np
import pandas as pd
import pytest
import yaml
from vio_iqa.exp004 import read_calibration, geometry_of_matches, extract, report, grid_coverage

def calibration(path):
    path.write_text(yaml.safe_dump(dict(camera_model="pinhole",resolution=[160,120],
        intrinsics=[120.,120.,80.,60.],distortion_model="radtan",
        distortion_coefficients=[0.,0.,0.,0.])))
    return path

def sequence(tmp_path,name,n=17,blank=False):
    path=tmp_path/name/"mav0"/"cam0"
    (path/"data").mkdir(parents=True)
    im=np.random.default_rng(43).integers(20,235,(120,160),dtype=np.uint8)
    lines=["#timestamp [ns],filename"]
    for k in range(n):
        img=np.zeros_like(im) if blank else np.roll(im,k,axis=1)
        cv2.imwrite(str(path/"data"/f"{k}.png"),img)
        lines.append(f"{1_000_000_000+k*50_000_000},{k}.png")
    (path/"data.csv").write_text("\n".join(lines)+"\n")
    calibration(path/"sensor.yaml")
    return path.parent.parent

def args(seq,out,**changes):
    kw=dict(sequence=seq,output_dir=out,calibration=None,max_frames=0,stride=3,
        budget=75,candidates=400,fb_threshold=1.,uint16_white=None,mask=None,
        circle_radius=None,min_matches=12,epi_threshold_px=1.5,min_parallax_deg=.25)
    kw.update(changes)
    return Namespace(**kw)

def test_calibration_strict(tmp_path):
    path=calibration(tmp_path/"sensor.yaml")
    K,dist,model=read_calibration(path,(120,160))
    assert K[0,0]==120 and model=="radtan"
    with pytest.raises(ValueError,match="resolution"):
        read_calibration(path,(80,160))

def test_synthetic_3d_correspondence_geometry():
    rng=np.random.default_rng(50)
    X=rng.uniform([-2.,-1.5,4.],[2.,1.5,10.],(120,3))
    R=cv2.Rodrigues(np.array([.025,-.016,.012]))[0]
    t=np.array([.45,.07,.12])
    K=np.array([[350.,0,320],[0,350.,240],[0,0,1.]])
    def project(Y):
        q=(K@Y.T).T
        return q[:,:2]/q[:,2,None]
    a=project(X)+rng.normal(0,.15,(120,2))
    b=project((R@X.T).T+t)+rng.normal(0,.15,(120,2))
    b[-24:]=rng.uniform([0,0],[639,479],(24,2))
    o=geometry_of_matches(a,b,K,np.zeros(4),"radtan",np.ones((480,640),bool))
    assert o["geometry_status"]=="estimated"
    assert o["n_geom_inlier"]>=85
    assert o["median_sampson_px"]<1
    assert o["median_parallax_deg"]>.25

def test_insufficient_is_nan():
    pts=np.zeros((5,2))
    K=np.eye(3)
    o=geometry_of_matches(pts,pts,K,np.zeros(4),"radtan",np.ones((12,12),bool))
    assert o["geometry_status"]=="insufficient_matches"
    assert np.isnan(o["n_geom_inlier"]) and np.isnan(o["inlier_ratio_survivors"])

def test_grid():
    x,n=grid_coverage(np.array([[2,2],[15,2],[140,110]],float),np.ones((120,160),bool))
    assert n==48 and x==pytest.approx(2/48)

def test_end_to_end_and_tail(tmp_path):
    seq=sequence(tmp_path,"A")
    out=tmp_path/"out"
    extract(args(seq,out))
    g=pd.read_csv(out/"frame_geometry.csv")
    assert len(g)==12
    assert g[g.censored==1].survival_rate.isna().all()
    assert g[g.censored==0].survival_rate.between(0,1).all()
    assert g[(g.horizon==5)&(g.frame_idx>=12)].censored.eq(1).all()
    trace=pd.read_csv(out/"feature_tracks.csv")
    assert not trace.duplicated(["start_frame_idx","feature_id","step"]).any()
    assert trace[trace.step==0].groupby("start_frame_idx").feature_id.nunique().eq(
        g[g.horizon==5].set_index("frame_idx").n_initial).all()
    assert json.loads((out/"manifest.json").read_text())["distortion_model"]=="radtan"
    report(Namespace(input_dir=out,output_dir=None,exp002_labels=None,high_survival=.8,low_inlier=.5))
    assert (out/"geometry_correlations.csv").exists()
    original=(out/"frame_geometry.csv").read_bytes()
    extract(args(seq,out))
    assert (out/"frame_geometry.csv").read_bytes()==original

def test_blank_and_stride(tmp_path):
    seq=sequence(tmp_path,"Empty",n=12,blank=True)
    out=tmp_path/"empty"
    extract(args(seq,out))
    d=pd.read_csv(out/"frame_geometry.csv")
    assert d.n_initial.eq(0).all() and d.n_geom_inlier.isna().all()
    assert set(d.geometry_status)=={"no_initial_features","tail_censored"}
    seq2=sequence(tmp_path,"Stride",n=15)
    other=tmp_path/"stride"
    extract(args(seq2,other,stride=4))
    tracks=pd.read_csv(other/"feature_tracks.csv")
    assert tracks[tracks.start_frame_idx==0].step.max()==10

def test_exp002_comparison(tmp_path):
    seq=sequence(tmp_path,"Consistency",n=14)
    out=tmp_path/"result"
    extract(args(seq,out))
    g=pd.read_csv(out/"frame_geometry.csv")
    base=g[g.horizon==5][["sequence","frame_idx","n_initial"]].copy()
    for k in (5,10):
        z=g[g.horizon==k].set_index("frame_idx")
        base[f"censored_{k}"]=base.frame_idx.map(z.censored)
        base[f"n_survive_{k}"]=base.frame_idx.map(z.n_survive)
        base[f"s_{k}"]=base.frame_idx.map(z.survival_rate)
    csv=tmp_path/"exp002.csv"
    base.to_csv(csv,index=False)
    report(Namespace(input_dir=out,output_dir=None,exp002_labels=csv,high_survival=.8,low_inlier=.5))
    res=pd.read_csv(out/"exp002_consistency.csv")
    assert res.matched_exp002.all() and len(res)==len(g)
