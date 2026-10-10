import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.spatial.transform import Rotation
from vio_iqa.exp004b import GTPose, camera_extrinsic, relative_pose, check_matches

def gt_csv(tmp_path):
    p=tmp_path/"gt.csv"
    base=1403636579763555584
    rows=["# timestamp,position,quaternion,..."]
    for i in range(40):
        t=base+i*5000000
        q=Rotation.from_rotvec([0,0,i*.001]).as_quat()
        rows.append(",".join(map(str,[t,i*.005,0,0,q[3],q[0],q[1],q[2]]+[0]*9)))
    p.write_text("\n".join(rows)+"\n")
    return p,base

def test_precise_interpolation(tmp_path):
    p,base=gt_csv(tmp_path)
    gt=GTPose(p)
    pose,status=gt.at(base+12500000)
    assert status=="ok"
    assert pose[1][0]==pytest.approx(.0125)
    assert gt.at(base-1)[1]=="gt_outside_time_range"

def test_extrinsic_and_pose_convention(tmp_path):
    p=tmp_path/"sensor.yaml"
    T=np.eye(4)
    T[:3,3]=[0,.5,0]
    p.write_text(yaml.safe_dump(dict(T_BS=T.tolist())))
    T0=camera_extrinsic(p)
    R,t=relative_pose((np.eye(3),np.zeros(3)),
        (np.eye(3),np.array([.1,0,0])),T0)
    np.testing.assert_allclose(R,np.eye(3))
    np.testing.assert_allclose(t,[-.1,0,0])
    R,t=relative_pose((np.eye(3),np.zeros(3)),
        (Rotation.from_euler("z",90,degrees=True).as_matrix(),np.zeros(3)),T0)
    assert np.linalg.norm(t)>.6

def test_gt_epipolar_and_baseline():
    rng=np.random.default_rng(8)
    X=rng.uniform([-2,-2,5],[2,2,10],(100,3))
    R=Rotation.from_rotvec([.03,.02,.01]).as_matrix()
    t=np.array([-.5,.02,.01])
    K=np.array([[350,0,320],[0,350,240],[0,0,1]],float)
    def project(P):
        q=(K@P.T).T
        return q[:,:2]/q[:,2,None]
    a=project(X)+rng.normal(0,.12,(100,2))
    b=project((R@X.T).T+t)+rng.normal(0,.12,(100,2))
    b[-18:]=rng.uniform([0,0],[639,479],(18,2))
    out=check_matches(a,b,K,np.zeros(4),"radtan",R,t,.01,1.5)
    assert out["gt_status"]=="gt_evaluated"
    assert out["good"][:82].mean()>.95
    assert out["good"][-18:].mean()<.2
    assert out["gt_positive_depth_fraction"]>.9
    low=check_matches(a,b,K,np.zeros(4),"radtan",R,np.zeros(3),.01,1.5)
    assert low["gt_status"]=="low_gt_baseline"
    assert np.isnan(low["n_gt_epi_inlier"])

def test_gap_and_malformed_extrinsic(tmp_path):
    path,base=gt_csv(tmp_path)
    contents=path.read_text().splitlines()
    path.write_text("\n".join(contents[:10]+contents[20:])+"\n")
    gt=GTPose(path,max_gap_s=.03)
    assert gt.at(base+70_000_000)[1]=="gt_interpolation_gap"
    sensor=tmp_path/"sensor.yaml"
    T=np.eye(4);T[2,2]=-1
    sensor.write_text(yaml.safe_dump(dict(T_BS=T.tolist())))
    with pytest.raises(ValueError,match="orthonormal"):
        camera_extrinsic(sensor)
