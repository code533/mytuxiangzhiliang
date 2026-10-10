"""EXP004-A: calibrated multiframe KLT tracking and geometric consistency.

All geometry outcomes are retrospective. Essential RANSAC is fitted to the
same correspondences and is not independent geometric truth or VIO success.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from vio_iqa.cli import load_index
from vio_iqa.exp001 import image_to_u8, make_roi
from vio_iqa.exp002 import HORIZONS, LK, _inside, select_spread

GEOM_COLUMNS = ["sequence", "frame_idx", "t_ns", "horizon", "end_frame_idx",
    "end_t_ns", "n_initial", "n_survive", "survival_rate", "censored",
    "geometry_status", "n_geom_inlier", "inlier_ratio_survivors",
    "inlier_ratio_initial", "median_sampson_px", "median_parallax_deg",
    "positive_depth_fraction", "inlier_grid_coverage", "n_grid_valid"]
TRACK_COLUMNS = ["sequence", "start_frame_idx", "feature_id", "step",
                 "frame_idx", "t_ns", "x_px", "y_px", "status"]
MATCH_COLUMNS = ["sequence", "start_frame_idx", "horizon", "feature_id",
                 "x0_px", "y0_px", "xk_px", "yk_px", "ransac_inlier",
                 "cheirality_inlier", "sampson_px", "parallax_deg"]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for blob in iter(lambda: f.read(1024*1024), b""):
            h.update(blob)
    return h.hexdigest()


def read_calibration(sensor_yaml: Path, image_shape: tuple[int, int]):
    """Require EuRoC pinhole intrinsics, correct image size, known distortion."""
    if not sensor_yaml.is_file():
        raise FileNotFoundError(f"Missing calibrated cam0/sensor.yaml: {sensor_yaml}")
    p = yaml.safe_load(sensor_yaml.read_text(encoding="utf8"))
    if not isinstance(p, dict) or p.get("camera_model") != "pinhole":
        raise ValueError("EXP004-A supports calibrated camera_model: pinhole only")
    wh = p.get("resolution")
    if not isinstance(wh,list) or len(wh)!=2 or [int(wh[1]),int(wh[0])] != list(image_shape):
        raise ValueError(f"Calibration resolution {wh} differs from image shape {image_shape}")
    intrinsic = np.asarray(p.get("intrinsics",[]), dtype=float)
    if intrinsic.shape!=(4,) or not np.isfinite(intrinsic).all() or min(intrinsic[:2])<=0:
        raise ValueError("Expected finite positive [fx,fy,cx,cy] intrinsics")
    fx,fy,cx,cy=intrinsic
    K=np.asarray([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype=np.float64)
    model=p.get("distortion_model")
    dist=np.asarray(p.get("distortion_coefficients",[]),dtype=float)
    if model in ("radtan","equidistant") and dist.shape==(4,):
        pass
    elif model in ("none",None) and dist.size==0:
        model="none"
    else:
        raise ValueError(f"Unsupported distortion model {model!r} or coefficients")
    if not np.isfinite(dist).all():
        raise ValueError("Nonfinite distortion coefficients")
    return K,dist,model


def normalized_rays(px:np.ndarray,K:np.ndarray,dist:np.ndarray,model:str)->np.ndarray:
    p=np.asarray(px,dtype=np.float64).reshape(-1,1,2)
    if not len(p):
        return np.empty((0,2),np.float64)
    if model=="equidistant":
        rays=cv2.fisheye.undistortPoints(p,K,dist.reshape(4,1))
    else:
        rays=cv2.undistortPoints(p,K,dist.reshape(-1,1) if dist.size else None)
    return rays.reshape(-1,2)


def sampson_px(x0,x1,E,f_ref):
    a=np.column_stack([x0,np.ones(len(x0))])
    b=np.column_stack([x1,np.ones(len(x1))])
    Ea=(E@a.T).T
    Etb=(E.T@b.T).T
    numerator=np.einsum("ij,ij->i",b,Ea)**2
    denom=Ea[:,0]**2+Ea[:,1]**2+Etb[:,0]**2+Etb[:,1]**2
    return np.sqrt(numerator/np.maximum(denom,1e-20))*f_ref


def grid_coverage(points,valid_mask,shape=(8,6)):
    h,w=valid_mask.shape
    ys,xs=np.where(valid_mask)
    valid_cells=set(zip(np.minimum(shape[0]-1,xs*shape[0]//w),
                        np.minimum(shape[1]-1,ys*shape[1]//h)))
    if not valid_cells:
        return float("nan"),0
    occupied=set((min(shape[0]-1,int(x*shape[0]/w)),
                  min(shape[1]-1,int(y*shape[1]/h))) for x,y in points)
    return float(len(occupied & valid_cells)/len(valid_cells)),len(valid_cells)


def geometry_of_matches(p0,pk,K,dist,model,valid_mask,threshold_px=1.5,
                        min_correspondences=12,min_parallax_deg=.25):
    """Fit E with KLT survivors, report status if geometry cannot be resolved."""
    p0=np.asarray(p0,dtype=np.float64).reshape(-1,2)
    pk=np.asarray(pk,dtype=np.float64).reshape(-1,2)
    if len(p0)!=len(pk):
        raise ValueError("Mismatched correspondence arrays")
    n=len(p0)
    res=dict(geometry_status="insufficient_matches",n_geom_inlier=np.nan,
             inlier_ratio_survivors=np.nan,median_sampson_px=np.nan,
             median_parallax_deg=np.nan,positive_depth_fraction=np.nan,
             inlier_grid_coverage=np.nan,
             n_grid_valid=grid_coverage(np.empty((0,2)),valid_mask)[1],
             ransac_inlier=np.zeros(n,dtype=bool),
             cheirality_inlier=np.zeros(n,dtype=bool),
             sampson_px=np.full(n,np.nan),parallax_deg=np.full(n,np.nan))
    if n<min_correspondences:
        return res
    if not np.isfinite(p0).all() or not np.isfinite(pk).all():
        raise ValueError("Nonfinite matching coordinate")
    if threshold_px<=0 or min_parallax_deg<=0 or min_correspondences<8:
        raise ValueError("Invalid geometric thresholds")
    x0=normalized_rays(p0,K,dist,model)
    x1=normalized_rays(pk,K,dist,model)
    f_ref=float(np.sqrt(K[0,0]*K[1,1]))
    cv2.setRNGSeed(7)
    try:
        E,mask=cv2.findEssentialMat(x0,x1,cameraMatrix=np.eye(3),method=cv2.RANSAC,
                                   prob=.999,threshold=threshold_px/f_ref,maxIters=1000)
    except cv2.error:
        E,mask=None,None
    if E is None or mask is None or E.shape!=(3,3):
        res["geometry_status"]="essential_failed_or_ambiguous"
        return res
    mask=mask.ravel().astype(bool)
    if mask.sum()<5:
        res["geometry_status"]="insufficient_ransac_inliers"
        return res
    error=sampson_px(x0,x1,E,f_ref)
    res["ransac_inlier"]=mask
    res["sampson_px"]=error
    res["n_geom_inlier"]=int(mask.sum())
    res["inlier_ratio_survivors"]=float(mask.mean())
    res["median_sampson_px"]=float(np.median(error[mask]))
    res["inlier_grid_coverage"]=grid_coverage(p0[mask],valid_mask)[0]
    try:
        _,R,_t,pose_mask=cv2.recoverPose(E,x0,x1,cameraMatrix=np.eye(3),
                                          mask=(mask.astype(np.uint8)*255).reshape(-1,1))
    except cv2.error:
        res["geometry_status"]="pose_recovery_failed"
        return res
    positive=(pose_mask.ravel()!=0)&mask
    res["cheirality_inlier"]=positive
    res["positive_depth_fraction"]=float(positive.sum()/mask.sum())
    if positive.sum()<5:
        res["geometry_status"]="insufficient_positive_depth"
        return res
    b0=np.column_stack([x0,np.ones(n)])
    b1=np.column_stack([x1,np.ones(n)])
    b0/=np.linalg.norm(b0,axis=1)[:,None]
    b1/=np.linalg.norm(b1,axis=1)[:,None]
    compensated=(R.T@b1.T).T
    angles=np.rad2deg(np.arccos(np.clip(np.einsum("ij,ij->i",b0,compensated),-1,1)))
    res["parallax_deg"]=angles
    median=float(np.median(angles[positive]))
    res["median_parallax_deg"]=median
    res["geometry_status"]="low_parallax_heuristic" if median<min_parallax_deg else "estimated"
    return res


def advance_cohorts(cohorts,prev,cur,inside,fb_threshold,frame_idx,t_ns,tracks_writer):
    """Same batched forward/backward KLT as EXP002, with feature IDs retained."""
    sizes=[len(c["points"]) for c in cohorts]
    n=sum(sizes)
    passed=np.zeros(n,dtype=bool)
    next_pts=np.empty((n,2),np.float32)
    if n:
        original=np.concatenate([c["points"] for c in cohorts],axis=0)
        pts=original.reshape(-1,1,2)
        fwd,s1,_=cv2.calcOpticalFlowPyrLK(prev,cur,pts,None,**LK)
        if fwd is not None and s1 is not None:
            fwd=fwd.reshape(-1,2)
            finite=np.isfinite(fwd).all(axis=1)
            safe=fwd.copy()
            safe[~finite]=original[~finite]
            back,s2,_=cv2.calcOpticalFlowPyrLK(cur,prev,safe.reshape(-1,1,2),None,**LK)
            if back is not None and s2 is not None:
                back=back.reshape(-1,2)
                h,w=prev.shape
                valid=(s1.ravel()==1)&(s2.ravel()==1)&finite&np.isfinite(back).all(axis=1)
                valid&=(fwd[:,0]>=0)&(fwd[:,0]<w)&(fwd[:,1]>=0)&(fwd[:,1]<h)
                ids=np.flatnonzero(valid)
                valid[ids]&=inside[fwd[ids,1].astype(int),fwd[ids,0].astype(int)]
                error=np.linalg.norm(original-back,axis=1)
                passed=valid&(error<=fb_threshold)
                next_pts[:]=safe
    offset=0
    for c,size in zip(cohorts,sizes):
        good=passed[offset:offset+size]
        for fid,p,ok in zip(c["ids"],next_pts[offset:offset+size],good):
            tracks_writer.writerow(dict(sequence=c["sequence"],start_frame_idx=c["start_idx"],
                feature_id=int(fid),step=frame_idx-c["start_idx"],frame_idx=frame_idx,t_ns=t_ns,
                x_px=float(p[0]) if ok else "",y_px=float(p[1]) if ok else "",
                status="alive" if ok else "lost"))
        c["points"]=next_pts[offset:offset+size][good]
        c["ids"]=c["ids"][good]
        c["steps"]=frame_idx-c["start_idx"]
        offset+=size


def finish_horizon(c,k,t_ns,end_idx,K,dist,model,inside,
                   min_matches,epi_threshold,min_parallax,match_writer):
    n0=len(c["first"])
    ns=len(c["points"])
    row=dict(sequence=c["sequence"],frame_idx=c["start_idx"],t_ns=c["start_ns"],
             horizon=k,end_frame_idx=end_idx,end_t_ns=t_ns,n_initial=n0,
             n_survive=ns,survival_rate=ns/n0 if n0 else np.nan,censored=0)
    if n0==0:
        row.update(geometry_status="no_initial_features")
        return row
    if ns==0:
        row.update(geometry_status="insufficient_matches")
        return row
    start=c["first"][c["ids"]]
    result=geometry_of_matches(start,c["points"],K,dist,model,inside,epi_threshold,
                               min_matches,min_parallax)
    row.update({key:result[key] for key in (
        "geometry_status","n_geom_inlier","inlier_ratio_survivors","median_sampson_px",
        "median_parallax_deg","positive_depth_fraction","inlier_grid_coverage","n_grid_valid")})
    row["inlier_ratio_initial"]=(float(result["n_geom_inlier"]/n0)
                                 if np.isfinite(result["n_geom_inlier"]) else np.nan)
    for j,fid in enumerate(c["ids"]):
        match_writer.writerow(dict(sequence=c["sequence"],start_frame_idx=c["start_idx"],
            horizon=k,feature_id=int(fid),x0_px=float(start[j,0]),y0_px=float(start[j,1]),
            xk_px=float(c["points"][j,0]),yk_px=float(c["points"][j,1]),
            ransac_inlier=int(result["ransac_inlier"][j]) if np.isfinite(result["n_geom_inlier"]) else "",
            cheirality_inlier=int(result["cheirality_inlier"][j]) if np.isfinite(result["positive_depth_fraction"]) else "",
            sampson_px=float(result["sampson_px"][j]) if np.isfinite(result["sampson_px"][j]) else "",
            parallax_deg=float(result["parallax_deg"][j]) if np.isfinite(result["parallax_deg"][j]) else ""))
    return row


def extract(args):
    if args.max_frames<0 or args.stride<1 or args.budget<1 or args.candidates<args.budget or args.fb_threshold<0:
        raise ValueError("Invalid KLT controls")
    if args.min_matches<8 or args.epi_threshold_px<=0 or args.min_parallax_deg<=0:
        raise ValueError("Invalid geometric thresholds")
    cam,index=load_index(Path(args.sequence))
    if args.max_frames:
        index=index.iloc[:args.max_frames]
    if len(index)<2:
        raise ValueError("At least two frames required")
    def read(i):
        return image_to_u8(cam/"data"/str(index.iloc[i].filename),args.uint16_white)
    first,dtype=read(0)
    sensor=Path(args.calibration) if args.calibration else cam/"sensor.yaml"
    K,dist,model=read_calibration(sensor,first.shape)
    roi=make_roi(first.shape,args.mask,args.circle_radius)
    inside=_inside(roi)
    if not inside.any():
        raise ValueError("ROI has no detectable region")
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    active=[]
    prev=first
    with (out/"feature_tracks.csv").open("w",newline="") as f_tracks, \
         (out/"geometry_matches.csv").open("w",newline="") as f_matches:
        tw=csv.DictWriter(f_tracks,TRACK_COLUMNS)
        tw.writeheader()
        mw=csv.DictWriter(f_matches,MATCH_COLUMNS)
        mw.writeheader()
        for frame_idx in range(len(index)):
            current,typ=(first,dtype) if frame_idx==0 else read(frame_idx)
            if typ!=dtype or current.shape!=first.shape:
                raise ValueError("All frames must retain image size and dtype")
            t_ns=int(index.iloc[frame_idx].timestamp_ns)
            if frame_idx>0:
                advance_cohorts(active,prev,current,inside,args.fb_threshold,frame_idx,t_ns,tw)
                for c in active:
                    if c["steps"] in HORIZONS:
                        rows.append(finish_horizon(c,c["steps"],t_ns,frame_idx,K,dist,model,
                                     inside,args.min_matches,args.epi_threshold_px,
                                     args.min_parallax_deg,mw))
                active=[c for c in active if c["steps"]<max(HORIZONS)]
            if frame_idx%args.stride==0:
                pts=select_spread(current,inside,args.budget,args.candidates)
                c=dict(sequence=Path(args.sequence).name,start_idx=frame_idx,start_ns=t_ns,
                       first=pts,points=pts.copy(),ids=np.arange(len(pts)),steps=0)
                for fid,p in enumerate(pts):
                    tw.writerow(dict(sequence=c["sequence"],start_frame_idx=frame_idx,
                        feature_id=fid,step=0,frame_idx=frame_idx,t_ns=t_ns,
                        x_px=float(p[0]),y_px=float(p[1]),status="alive"))
                active.append(c)
            prev=current
            if (frame_idx+1)%500==0:
                print(f"EXP004-A processed {frame_idx+1}/{len(index)} frames",flush=True)
        for c in active:
            for k in HORIZONS:
                if c["steps"]<k:
                    rows.append(dict(sequence=c["sequence"],frame_idx=c["start_idx"],
                        t_ns=c["start_ns"],horizon=k,end_frame_idx=np.nan,end_t_ns=np.nan,
                        n_initial=len(c["first"]),n_survive=np.nan,survival_rate=np.nan,
                        censored=1,geometry_status="tail_censored"))
    rows=sorted(rows,key=lambda r:(r["frame_idx"],r["horizon"]))
    df=pd.DataFrame(rows).reindex(columns=GEOM_COLUMNS)
    if len(df)!=len(range(0,len(index),args.stride))*len(HORIZONS):
        raise AssertionError("Each starting frame must have all horizon records")
    df.to_csv(out/"frame_geometry.csv",index=False)
    manifest=dict(experiment="EXP004-A",sequence=str(Path(args.sequence).resolve()),
        n_images=len(index),n_start_frames=len(range(0,len(index),args.stride)),
        n_horizon_rows=len(df),source_dtype=dtype,uint16_white=args.uint16_white,
        calibration=str(sensor.resolve()),calibration_sha256=sha256(sensor),
        intrinsics=K.tolist(),distortion_model=model,distortion_coefficients=dist.tolist(),
        image_size_hw=list(first.shape),roi_fraction=float(roi.mean()),
        mask=args.mask,circle_radius=args.circle_radius,budget=args.budget,
        candidates=args.candidates,stride=args.stride,max_frames=args.max_frames,
        fb_threshold=args.fb_threshold,horizons=list(HORIZONS),min_matches=args.min_matches,
        ransac_normalized_threshold=args.epi_threshold_px/np.sqrt(K[0,0]*K[1,1]),
        ransac_equivalent_threshold_px=args.epi_threshold_px,
        min_parallax_deg=args.min_parallax_deg,
        status_counts={str(k):int(v) for k,v in df.geometry_status.value_counts().items()},
        note="Self-fitted E/RANSAC is not GT; low-parallax is heuristic; future frames used for offline labels.")
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf8")
    print("EXP004-A saved",len(df),"horizon rows to",out)


def report(args):
    df=pd.read_csv(Path(args.input_dir)/"frame_geometry.csv")
    out=Path(args.output_dir or args.input_dir)
    out.mkdir(parents=True,exist_ok=True)
    if getattr(args,"exp002_labels",None):
        base=pd.read_csv(args.exp002_labels)
        comparisons=[]
        for k in HORIZONS:
            required=["sequence","frame_idx","n_initial",f"censored_{k}",f"n_survive_{k}",f"s_{k}"]
            if any(c not in base for c in required):
                raise ValueError(f"EXP002 baseline must include {required}")
            old=base[required].rename(columns={f"censored_{k}":"exp002_censored",
                f"n_survive_{k}":"exp002_n_survive",f"s_{k}":"exp002_survival",
                "n_initial":"exp002_n_initial"})
            joined=df[df.horizon==k].merge(old,on=["sequence","frame_idx"],
                                             how="left",validate="one_to_one")
            for r in joined.itertuples():
                same=(r.exp002_n_initial==r.n_initial and r.exp002_censored==r.censored
                      and ((pd.isna(r.n_survive) and pd.isna(r.exp002_n_survive))
                           or r.n_survive==r.exp002_n_survive)
                      and ((pd.isna(r.survival_rate) and pd.isna(r.exp002_survival))
                           or np.isclose(r.survival_rate,r.exp002_survival,atol=1e-8)))
                comparisons.append(dict(sequence=r.sequence,frame_idx=int(r.frame_idx),
                                        horizon=k,matched_exp002=bool(same)))
        pd.DataFrame(comparisons).to_csv(out/"exp002_consistency.csv",index=False)
        print("EXP002 consistent:",sum(x["matched_exp002"] for x in comparisons),
              "/",len(comparisons))
    records=[]
    failures=[]
    for (seq,k),g in df.groupby(["sequence","horizon"]):
        eligible=g[g.geometry_status.isin(["estimated","low_parallax_heuristic"])]
        for metric in ["inlier_ratio_survivors","inlier_ratio_initial",
                       "median_parallax_deg","inlier_grid_coverage","n_geom_inlier"]:
            x=eligible[["survival_rate",metric]].replace([np.inf,-np.inf],np.nan).dropna()
            rho=(float(spearmanr(x.survival_rate,x[metric]).statistic)
                 if len(x)>=10 and x.survival_rate.nunique()>1 and x[metric].nunique()>1 else np.nan)
            records.append(dict(sequence=seq,horizon=k,metric=metric,n=len(x),
                spearman=rho,usable_estimated=int(g.geometry_status.eq("estimated").sum()),
                eligible_for_geometry=len(eligible),tail_censored=int(g.censored.fillna(0).sum()),
                n_total=len(g)))
        for row in eligible[(eligible.survival_rate>=args.high_survival)&
                            (eligible.inlier_ratio_survivors<args.low_inlier)].itertuples():
            failures.append(dict(sequence=seq,frame_idx=int(row.frame_idx),horizon=k,
                survival_rate=row.survival_rate,n_survive=int(row.n_survive),
                geometry_status=row.geometry_status,n_geom_inlier=int(row.n_geom_inlier),
                inlier_ratio_survivors=row.inlier_ratio_survivors,
                median_parallax_deg=row.median_parallax_deg))
    pd.DataFrame(records).to_csv(out/"geometry_correlations.csv",index=False)
    pd.DataFrame(failures,columns=["sequence","frame_idx","horizon","survival_rate",
        "n_survive","geometry_status","n_geom_inlier","inlier_ratio_survivors",
        "median_parallax_deg"]).to_csv(out/"high_survival_low_inlier_examples.csv",index=False)
    df.groupby(["sequence","horizon","geometry_status"],dropna=False).size().rename("n").reset_index().to_csv(
        out/"status_counts.csv",index=False)
    print("EXP004-A reports saved to",out)


def main():
    parser=argparse.ArgumentParser(description="EXP004-A: calibrated KLT geometric consistency")
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("extract")
    p.add_argument("--sequence",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--calibration",type=Path)
    p.add_argument("--max-frames",type=int,default=300)
    p.add_argument("--stride",type=int,default=5)
    p.add_argument("--budget",type=int,default=240)
    p.add_argument("--candidates",type=int,default=3000)
    p.add_argument("--fb-threshold",type=float,default=1.)
    p.add_argument("--uint16-white",type=int)
    p.add_argument("--mask")
    p.add_argument("--circle-radius",type=float)
    p.add_argument("--min-matches",type=int,default=12)
    p.add_argument("--epi-threshold-px",type=float,default=1.5)
    p.add_argument("--min-parallax-deg",type=float,default=.25)
    p.set_defaults(func=extract)
    q=sub.add_parser("report")
    q.add_argument("--input-dir",type=Path,required=True)
    q.add_argument("--output-dir",type=Path)
    q.add_argument("--high-survival",type=float,default=.8)
    q.add_argument("--low-inlier",type=float,default=.5)
    q.add_argument("--exp002-labels",type=Path)
    q.set_defaults(func=report)
    args=parser.parse_args()
    args.func(args)


if __name__=="__main__":
    main()
