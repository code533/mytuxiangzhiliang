"""EXP004-B: verify EXP004-A KLT matches against independent EuRoC poses.

EuRoC conventions: GT T_WB, camera sensor.yaml T_BS = T_BC (camera to
body). T_WC = T_WB @ T_BC and T_C1_C0 = inv(T_WC1) @ T_WC0.
Use exact cam0/data.csv integer nanosecond times; never use EXP004-A's
floating point end_t_ns. Epipolar consistency is NOT VIO update success.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import spearmanr
from vio_iqa.cli import load_index
from vio_iqa.exp004 import normalized_rays, read_calibration, sampson_px

def sha(path):
    hh=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""): hh.update(b)
    return hh.hexdigest()

def read_groundtruth(path):
    d=pd.read_csv(path,comment="#",header=None)
    if len(d)<2 or d.shape[1]!=17:
        raise ValueError("Expected EuRoC state_groundtruth_estimate0/data.csv (17 columns)")
    t=pd.to_numeric(d.iloc[:,0],errors="raise").to_numpy(np.int64)
    if np.any(np.diff(t)<=0):raise ValueError("GT timestamps must strictly increase")
    position=d.iloc[:,1:4].to_numpy(float)
    q=d.iloc[:,4:8].to_numpy(float) # EuRoC w,x,y,z
    if not (np.isfinite(position).all() and np.isfinite(q).all()):
        raise ValueError("Nonfinite GT poses")
    norms=np.linalg.norm(q,axis=1)
    if np.any(np.abs(norms-1)>.02):raise ValueError("GT quaternion norm invalid")
    q=q/norms[:,None]
    return t,position,Rotation.from_quat(q[:,[1,2,3,0]])

class GTPose:
    def __init__(self,path,max_gap_s=.03):
        if max_gap_s<=0:raise ValueError("Invalid GT gap bound")
        self.t,self.xyz,self.rot=read_groundtruth(path)
        self.t0=int(self.t[0])
        self.max_gap=max_gap_s
        self.slerp=Slerp((self.t-self.t0).astype(float)*1e-9,self.rot)
    def at(self,ns):
        ns=int(ns)
        if ns<int(self.t[0]) or ns>int(self.t[-1]):
            return None,"gt_outside_time_range"
        i=int(np.searchsorted(self.t,ns))
        if i<len(self.t) and self.t[i]==ns:
            return (self.rot[i].as_matrix(),self.xyz[i]),"ok"
        if i==0 or i>=len(self.t):return None,"gt_outside_time_range"
        dt=(int(self.t[i])-int(self.t[i-1]))*1e-9
        if dt>self.max_gap:return None,"gt_interpolation_gap"
        alpha=(ns-int(self.t[i-1]))/(int(self.t[i])-int(self.t[i-1]))
        p=(1-alpha)*self.xyz[i-1]+alpha*self.xyz[i]
        R=self.slerp([(ns-self.t0)*1e-9]).as_matrix()[0]
        return (R,p),"ok"

def camera_extrinsic(sensor):
    val=yaml.safe_load(Path(sensor).read_text(encoding="utf8")).get("T_BS")
    if val is None:raise ValueError("Missing T_BS in EuRoC camera sensor.yaml")
    # EuRoC/kalibr YAML encodes matrices as {rows: 4, cols: 4, data: [...]}
    # in row-major order. Also retain support for an explicit 4x4 list.
    if isinstance(val, dict):
        try:
            rows, cols = int(val["rows"]), int(val["cols"])
            raw = np.asarray(val["data"], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid EuRoC T_BS matrix mapping; expected rows/cols/data") from exc
        if rows != 4 or cols != 4 or raw.size != 16 or raw.ndim != 1:
            raise ValueError("EuRoC T_BS must describe a 4x4 row-major matrix")
        T = raw.reshape(4, 4)
    else:
        try:
            T = np.asarray(val, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid camera extrinsic T_BS") from exc
    if T.shape!=(4,4) or not np.isfinite(T).all():
        raise ValueError("Invalid camera extrinsic T_BS")
    if not np.allclose(T[3,:],[0,0,0,1],atol=1e-6):
        raise ValueError("T_BS last row invalid")
    R=T[:3,:3]
    if not np.allclose(R.T@R,np.eye(3),atol=.001) or not np.isclose(np.linalg.det(R),1,atol=.001):
        raise ValueError("T_BS must be a proper orthonormal rotation")
    return T

def relative_pose(p0,p1,T_BC):
    R_WB0,t_WB0=p0;R_WB1,t_WB1=p1
    R_BC=T_BC[:3,:3];t_BC=T_BC[:3,3]
    R_WC0=R_WB0@R_BC;R_WC1=R_WB1@R_BC
    t_WC0=t_WB0+R_WB0@t_BC
    t_WC1=t_WB1+R_WB1@t_BC
    return R_WC1.T@R_WC0,R_WC1.T@(t_WC0-t_WC1)

def check_matches(p0,p1,K,D,model,R,t,baseline_min,epi_px):
    x0=normalized_rays(p0,K,D,model)
    x1=normalized_rays(p1,K,D,model)
    n=len(p0)
    b0=np.column_stack([x0,np.ones(n)])
    b1=np.column_stack([x1,np.ones(n)])
    b0/=np.linalg.norm(b0,axis=1)[:,None]
    b1/=np.linalg.norm(b1,axis=1)[:,None]
    theta=np.rad2deg(np.arccos(np.clip(np.einsum("ij,ij->i",b0,(R.T@b1.T).T),-1,1)))
    flow=np.linalg.norm(p1-p0,axis=1)
    baseline=float(np.linalg.norm(t))
    out=dict(gt_status="low_gt_baseline",baseline_m=baseline,
        rotation_deg=float(np.rad2deg(Rotation.from_matrix(R).magnitude())),
        median_flow_px=float(np.median(flow)),gt_parallax_deg=float(np.median(theta)),
        n_gt_epi_inlier=np.nan,gt_inlier_ratio_survivors=np.nan,
        gt_median_sampson_px=np.nan,gt_positive_depth_fraction=np.nan,
        gt_median_reproj_px=np.nan,
        errors=np.full(n,np.nan),good=np.zeros(n,bool),theta=theta)
    if baseline<baseline_min:return out
    q=t/baseline
    cross=np.array([[0,-q[2],q[1]],[q[2],0,-q[0]],[-q[1],q[0],0]])
    E=cross@R
    f=float(np.sqrt(K[0,0]*K[1,1]))
    error=sampson_px(x0,x1,E,f)
    good=error<=epi_px
    out.update(gt_status="gt_evaluated",errors=error,good=good,
        n_gt_epi_inlier=int(good.sum()),gt_inlier_ratio_survivors=float(good.mean()),
        gt_median_sampson_px=float(np.median(error)))
    try:
        X4=cv2.triangulatePoints(np.column_stack([np.eye(3),np.zeros(3)]),
                                np.column_stack([R,t]),x0.T,x1.T)
        valid=np.isfinite(X4).all(axis=0)&(np.abs(X4[3])>1e-12)
        X=np.full((3,n),np.nan)
        X[:,valid]=X4[:3,valid]/X4[3,valid]
        X1=R@X+t[:,None]
        positive=valid&(X[2]>0)&(X1[2]>0)
        if good.any():out["gt_positive_depth_fraction"]=float((positive&good).sum()/good.sum())
        selected=positive&good
        if selected.any():
            uv0=(X[:2,selected]/X[2,selected]).T
            uv1=(X1[:2,selected]/X1[2,selected]).T
            reproj=np.sqrt(.5*(np.sum((uv0-x0[selected])**2,axis=1)+
                                   np.sum((uv1-x1[selected])**2,axis=1)))*f
            out["gt_median_reproj_px"]=float(np.median(reproj))
    except cv2.error:
        pass
    return out

def evaluate(args):
    sequence=Path(args.sequence)
    source=Path(args.exp004_dir)
    # EXP004-A already records the exact sequence root used to read images.
    # If a mistyped --sequence is supplied, print that known-good path rather
    # than misleadingly suggesting EuRoC lacks the "mav0" directory.
    manifest=json.loads((source/"manifest.json").read_text())
    try:
        cam,index=load_index(sequence)
    except FileNotFoundError as exc:
        candidates=[sequence/"mav0"/"cam0"/"data.csv",sequence/"cam0"/"data.csv"]
        expected=manifest.get("sequence", "<unknown>")
        raise FileNotFoundError(
            "EXP004-B could not locate the EuRoC camera index at either: "
            + ", ".join(map(str,candidates))
            + f". EXP004-A successfully used --sequence {expected!s}; "
            + "reuse that exact path, or find the dataset's cam0/data.csv."
        ) from exc
    sensor=Path(args.calibration) if args.calibration else cam/"sensor.yaml"
    # Support both full EuRoC root (.../sequence/mav0/cam0) and the mav0
    # directory itself (.../mav0/cam0), just like load_index().
    gt_file=Path(args.gt_csv) if args.gt_csv else cam.parent/"state_groundtruth_estimate0"/"data.csv"
    original=pd.read_csv(source/"frame_geometry.csv",dtype={"t_ns":"int64"})
    matches=pd.read_csv(source/"geometry_matches.csv")
    if Path(manifest["sequence"]).name!=sequence.name or set(original.sequence)!={sequence.name}:
        raise ValueError("EXP004-A outputs must match the requested sequence")
    if not set(matches.sequence).issubset({sequence.name}):
        raise ValueError("EXP004-A matches have a different sequence")
    K,D,model=read_calibration(sensor,tuple(manifest["image_size_hw"]))
    if manifest.get("calibration_sha256")!=sha(sensor):
        raise ValueError("Calibration SHA256 differs from EXP004-A")
    T_BC=camera_extrinsic(sensor)
    gt=GTPose(gt_file,args.max_gt_gap_s)
    index_ns=index.timestamp_ns.to_numpy(np.int64)
    if (original.frame_idx<0).any() or (original.frame_idx>=len(index_ns)).any():
        raise ValueError("Frame index outside EuRoC camera data.csv")
    if not np.array_equal(original.t_ns.to_numpy(np.int64),
        index_ns[original.frame_idx.to_numpy(int)]):
        raise ValueError("Start timestamps disagree with exact integer camera index")
    if original.duplicated(["frame_idx","horizon"]).any():
        raise ValueError("Duplicate frame/horizon")
    grouped={(int(k),int(h)):d for (k,h),d in matches.groupby(["start_frame_idx","horizon"])}
    frames=[];points=[]
    for row in original.itertuples(index=False):
        i=int(row.frame_idx);h=int(row.horizon)
        record=dict(sequence=sequence.name,frame_idx=i,horizon=h,exp004a_status=row.geometry_status,
                    n_initial=row.n_initial,n_survive=row.n_survive,survival_rate=row.survival_rate,
                    start_t_ns=str(int(index_ns[i])),end_t_ns="",gt_status="not_evaluated")
        if int(row.censored):
            record["gt_status"]="tail_censored";frames.append(record);continue
        end=i+h
        if end>=len(index_ns) or not np.isclose(float(row.end_frame_idx),end):
            raise ValueError("Horizon/end frame mismatch")
        record["end_t_ns"]=str(int(index_ns[end]))
        p0,st0=gt.at(int(index_ns[i])+args.gt_time_offset_ns)
        p1,st1=gt.at(int(index_ns[end])+args.gt_time_offset_ns)
        if p0 is None or p1 is None:
            record["gt_status"]=st0 if p0 is None else st1
            frames.append(record);continue
        R,t=relative_pose(p0,p1,T_BC)
        record["baseline_m"]=float(np.linalg.norm(t))
        record["rotation_deg"]=float(np.rad2deg(Rotation.from_matrix(R).magnitude()))
        corresp=grouped.get((i,h))
        if corresp is None:
            if int(row.n_survive)!=0:raise ValueError("Missing saved matches")
            record["gt_status"]="insufficient_matches"
            frames.append(record);continue
        if len(corresp)!=int(row.n_survive):
            raise ValueError("Survival count and matches disagree")
        x0=corresp[["x0_px","y0_px"]].to_numpy(float)
        x1=corresp[["xk_px","yk_px"]].to_numpy(float)
        if not (np.isfinite(x0).all() and np.isfinite(x1).all()):
            raise ValueError("Invalid correspondences")
        if len(x0)<args.min_matches:
            record["gt_status"]="insufficient_matches"
            record["median_flow_px"]=float(np.median(np.linalg.norm(x1-x0,axis=1)))
            frames.append(record);continue
        q=check_matches(x0,x1,K,D,model,R,t,args.min_baseline_m,args.epipolar_threshold_px)
        for key in ("gt_status","baseline_m","rotation_deg","median_flow_px","gt_parallax_deg",
                    "n_gt_epi_inlier","gt_inlier_ratio_survivors","gt_median_sampson_px",
                    "gt_positive_depth_fraction","gt_median_reproj_px"):
            record[key]=q[key]
        if q["gt_status"]=="gt_evaluated" and row.n_initial>0:
            record["gt_inlier_ratio_initial"]=q["n_gt_epi_inlier"]/row.n_initial
        mask=corresp.ransac_inlier.to_numpy(float)
        if np.isfinite(mask).all():
            record["self_ransac_inlier_ratio"]=float(mask.mean())
            if q["gt_status"]=="gt_evaluated":
                record["self_ransac_gt_agreement_fraction"]=float(np.mean((mask!=0)==q["good"]))
        for j,fid in enumerate(corresp.feature_id):
            points.append(dict(sequence=sequence.name,frame_idx=i,horizon=h,feature_id=int(fid),
                 gt_status=q["gt_status"],gt_sampson_px=float(q["errors"][j]),
                 gt_epipolar_inlier=int(q["good"][j]) if q["gt_status"]=="gt_evaluated" else np.nan,
                 gt_rot_comp_parallax_deg=float(q["theta"][j]),
                 self_ransac_inlier=mask[j]))
        frames.append(record)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    score=pd.DataFrame(frames)
    score.to_csv(out/"gt_frame_geometry.csv",index=False)
    pd.DataFrame(points).to_csv(out/"gt_match_geometry.csv",index=False)
    score.groupby(["horizon","exp004a_status","gt_status"],dropna=False).size().rename("n").reset_index().to_csv(
        out/"status_crosstab.csv",index=False)
    lines=[]
    for k,part in score.groupby("horizon"):
        for metric in ("gt_inlier_ratio_survivors","gt_parallax_deg","gt_positive_depth_fraction"):
            zz=part[["survival_rate",metric]].dropna()
            rho=float(spearmanr(zz.survival_rate,zz[metric]).statistic) if len(zz)>9 and zz.survival_rate.nunique()>1 and zz[metric].nunique()>1 else np.nan
            lines.append(dict(horizon=int(k),metric=metric,spearman=rho,n=len(zz),
                n_complete=int(part.exp004a_status.ne("tail_censored").sum()),
                n_gt_evaluated=int(part.gt_status.eq("gt_evaluated").sum()),
                n_low_gt_baseline=int(part.gt_status.eq("low_gt_baseline").sum())))
    pd.DataFrame(lines).to_csv(out/"gt_correlations.csv",index=False)
    score[(score.survival_rate>=args.high_survival)&(
       score.gt_status.eq("low_gt_baseline") |
       (score.gt_status.eq("gt_evaluated")&(score.gt_inlier_ratio_survivors<args.low_gt_inlier)) |
       score.exp004a_status.eq("insufficient_positive_depth")
       )].to_csv(out/"high_survival_geometry_warnings.csv",index=False)
    (out/"manifest.json").write_text(json.dumps(dict(experiment="EXP004-B",
        sequence=str(sequence.resolve()),gt_sha256=sha(gt_file),
        exp004a_sha256=sha(source/"manifest.json"),calibration_sha256=sha(sensor),
        gt_time_offset_ns=args.gt_time_offset_ns,max_gt_gap_s=args.max_gt_gap_s,
        min_baseline_m=args.min_baseline_m,epipolar_threshold_px=args.epipolar_threshold_px,
        pose_conventions="T_WC=T_WB@T_BC; T_C1_C0=inv(T_WC1)@T_WC0",
        note="GT-referenced epipolar consistency is not VIO update success",
        gt_status_counts={str(k):int(v) for k,v in score.gt_status.value_counts().items()}),
        ensure_ascii=False,indent=2),encoding="utf8")
    print("EXP004-B wrote",len(score),"frame/horizon rows to",out)
    print(score.groupby(["horizon","gt_status"]).size().to_string())

def main():
    p=argparse.ArgumentParser(description="EXP004-B independent EuRoC geometry reference")
    p.add_argument("--sequence",type=Path,required=True)
    p.add_argument("--exp004-dir",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--gt-csv",type=Path)
    p.add_argument("--calibration",type=Path)
    p.add_argument("--gt-time-offset-ns",type=int,default=0)
    p.add_argument("--max-gt-gap-s",type=float,default=.03)
    p.add_argument("--min-baseline-m",type=float,default=.01)
    p.add_argument("--min-matches",type=int,default=12)
    p.add_argument("--epipolar-threshold-px",type=float,default=1.5)
    p.add_argument("--high-survival",type=float,default=.8)
    p.add_argument("--low-gt-inlier",type=float,default=.5)
    args=p.parse_args()
    if args.max_gt_gap_s<=0 or args.min_baseline_m<=0 or args.epipolar_threshold_px<=0 or args.min_matches<1:
        raise ValueError("Invalid geometric and timestamp controls")
    evaluate(args)

if __name__=="__main__":main()
