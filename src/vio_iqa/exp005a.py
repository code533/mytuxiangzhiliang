"""EXP005-A: align past-only image/IMU predictors to retrospective GT labels.

This module builds and audits datasets; it NEVER fits a predictive model.
Event labels are horizon-specific, not observations available at frame t.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from vio_iqa.exp003 import IMAGE_FEATURES, IMU_FEATURES

HORIZONS=(5,10)
BASELINE_M=.01  # Frozen from EXP004-B
GT_EPI_LIMIT=.60  # Frozen from EXP004-B.1 descriptive threshold
EPI_PX=1.5  # Source EXP004-B GT Sampson threshold; never re-fit
MIN_MATCHES=12
PREDICTORS=IMAGE_FEATURES+IMU_FEATURES


def hash_file(path):
    dig=hashlib.sha256()
    with Path(path).open('rb') as fh:
        for data in iter(lambda:fh.read(1024*1024),b''):
            dig.update(data)
    return dig.hexdigest()


def _integer_timestamp(s,name):
    """Reject scientific notation and float timestamps; ensure exact ns."""
    text=s.astype('string')
    if text.isna().any() or not text.str.fullmatch(r'[0-9]{10,19}').all():
        raise ValueError(f'{name} must be exact integer nanoseconds, not float/scientific notation')
    return text.astype('int64')


def read_features(path):
    df=pd.read_csv(path,dtype={'t_ns':'string'})
    must=set(PREDICTORS+['sequence','frame_idx','t_ns','no_features',
              'imu_200ms_valid','imu_500ms_valid','n_initial','s_5','s_10'])
    absent=must-set(df)
    if absent:raise ValueError(f'{path}: missing EXP003 columns: {sorted(absent)}')
    if df.empty or df.sequence.nunique()!=1 or df[['sequence','frame_idx']].duplicated().any():
        raise ValueError(f'{path}: expected exactly one nonempty sequence with unique frames')
    df['t_ns']=_integer_timestamp(df.t_ns,'feature t_ns')
    if not df.frame_idx.is_monotonic_increasing or not df.t_ns.is_monotonic_increasing:
        raise ValueError(f'{path}: expected monotonic frames and nanosecond timestamps')
    return df


def read_geometry(path):
    d=pd.read_csv(path,dtype={'start_t_ns':'string','end_t_ns':'string'})
    must={'sequence','frame_idx','horizon','gt_status','baseline_m',
          'gt_inlier_ratio_survivors','n_initial','n_survive','survival_rate','start_t_ns'}
    absent=must-set(d)
    if absent:raise ValueError(f'{path}: missing EXP004-B columns: {sorted(absent)}')
    if d.empty or d.sequence.nunique()!=1 or d[['sequence','frame_idx','horizon']].duplicated().any():
        raise ValueError(f'{path}: expected unique (sequence,frame_idx,horizon) rows')
    if not set(d.horizon).issubset(HORIZONS) or set(d.horizon)!=set(HORIZONS):
        raise ValueError('Geometry must include both S5 and S10 horizons')
    d['start_t_ns']=_integer_timestamp(d.start_t_ns,'geometry start_t_ns')
    return d


def merge_sequence(feature_path,geometry_path):
    img=read_features(feature_path)
    geom=read_geometry(geometry_path)
    name=str(img.sequence.iloc[0])
    if set(geom.sequence)!={name}:
        raise ValueError(f'Sequence mismatch: {name} vs {geom.sequence.unique()}')
    # EXP004 cohorts are sampled (stride 5) from EXP003 per-frame features.
    # Never interpolate, shift, or choose the nearest available frame.
    merged=geom.merge(img,on=['sequence','frame_idx'],how='left',validate='many_to_one',
                      indicator=True,suffixes=('','_exp003'))
    if not merged['_merge'].eq('both').all():
        missing=merged.loc[merged['_merge']!='both',['frame_idx','horizon']].head(8)
        raise ValueError(f'{name}: geometry frames absent from EXP003 features: {missing.to_dict("records")}')
    if not np.array_equal(merged.start_t_ns.to_numpy(np.int64),merged.t_ns.to_numpy(np.int64)):
        raise ValueError(f'{name}: exact camera timestamp mismatch; do not round nanoseconds')
    for horizon in HORIZONS:
        sel=merged.horizon.eq(horizon)
        a=merged.loc[sel,'survival_rate'].to_numpy(float)
        b=merged.loc[sel,f's_{horizon}'].to_numpy(float)
        if not np.allclose(a,b,atol=1e-8,rtol=0,equal_nan=True):
            raise ValueError(f'{name} K={horizon}: EXP002 vs EXP004 survival mismatch')
        x=merged.loc[sel,'n_initial'].to_numpy(float)
        y=merged.loc[sel,'n_initial_exp003'].to_numpy(float)
        if not np.allclose(x,y,atol=0,rtol=0,equal_nan=False):
            raise ValueError(f'{name} K={horizon}: initial cohort count mismatch')
    if (~merged.gt_inlier_ratio_survivors.dropna().between(0,1)).any():
        raise ValueError('Out of range GT inlier fraction')
    gt_present=merged.gt_status.isin(['gt_evaluated','low_gt_baseline'])
    if not np.isfinite(merged.loc[gt_present,'baseline_m'].to_numpy(float)).all():
        raise ValueError('Finite GT baseline required for evaluable baseline labels')
    if (merged.loc[gt_present,'baseline_m']<0).any():
        raise ValueError('Negative GT baseline')
    if merged.loc[~merged.gt_status.eq('gt_evaluated'),'gt_inlier_ratio_survivors'].notna().any():
        raise ValueError('GT residual ratio must be missing when GT epipolar is unevaluable')
    if merged.loc[merged.gt_status.eq('gt_evaluated'),'gt_inlier_ratio_survivors'].isna().any():
        raise ValueError('GT-evaluated sample missing GT epipolar fraction')
    baseline_label=pd.Series(pd.NA,index=merged.index,dtype='Int64')
    baseline_label.loc[gt_present]=(merged.loc[gt_present,'baseline_m']<BASELINE_M).astype('int64').to_numpy()
    epi_ok=merged.gt_status.eq('gt_evaluated')&merged.n_survive.ge(MIN_MATCHES)
    epi_label=pd.Series(pd.NA,index=merged.index,dtype='Int64')
    epi_label.loc[epi_ok]=(merged.loc[epi_ok,'gt_inlier_ratio_survivors']<GT_EPI_LIMIT).astype('int64').to_numpy()
    merged['label_low_baseline']=baseline_label
    merged['label_low_epipolar']=epi_label
    merged['baseline_label_status']=np.select([
        merged.gt_status.eq('tail_censored'),gt_present],
        ['tail_censored','labeled'],default='gt_unavailable')
    merged['epipolar_label_status']=np.select([
        merged.gt_status.eq('tail_censored'),epi_ok,
        merged.gt_status.eq('low_gt_baseline'),
        merged.gt_status.eq('gt_evaluated')],
        ['tail_censored','labeled','low_baseline_excluded','insufficient_correspondences'],
        default='gt_unavailable')
    feature_ok=(merged.no_features.eq(0)&merged.imu_200ms_valid.eq(1)&
                merged.imu_500ms_valid.eq(1)&
                np.isfinite(merged[PREDICTORS].to_numpy(float)).all(axis=1))
    merged['predictors_available']=feature_ok.astype('int8')
    # Retrospective GT, future survival and labels are NOT model predictors.
    cols=['sequence','frame_idx','horizon','t_ns','start_t_ns',
          'predictors_available','no_features','imu_200ms_valid','imu_500ms_valid',
          *PREDICTORS,'gt_status','n_initial','n_survive','survival_rate',
          'baseline_m','gt_inlier_ratio_survivors','gt_parallax_deg','gt_positive_depth_fraction',
          'label_low_baseline','baseline_label_status','label_low_epipolar',
          'epipolar_label_status']
    return merged.reindex(columns=cols).sort_values(['frame_idx','horizon']).reset_index(drop=True)


def summary_table(df):
    rows=[]
    for (seq,k),g in df.groupby(['sequence','horizon'],sort=True):
        item={'sequence':seq,'horizon':int(k),'n_geometry_rows':len(g),
              'n_predictors_available':int(g.predictors_available.sum())}
        for target in ('label_low_baseline','label_low_epipolar'):
            eligible=g[target].notna()
            ready=eligible&g.predictors_available.eq(1)
            item[target+'_n_evaluable']=int(eligible.sum())
            item[target+'_n_events']=int(g.loc[eligible,target].sum())
            item[target+'_event_rate']=float(g.loc[eligible,target].mean()) if eligible.any() else np.nan
            item[target+'_n_ready']=int(ready.sum())
            item[target+'_n_ready_events']=int(g.loc[ready,target].sum())
            item[target+'_ready_event_rate']=float(g.loc[ready,target].mean()) if ready.any() else np.nan
            item[target+'_n_missing']=int((~eligible).sum())
            item[target+'_ready_two_classes']=bool(ready.any() and g.loc[ready,target].nunique()==2)
        for status,count in g.gt_status.value_counts().items():
            item['gt_status_'+status]=int(count)
        rows.append(item)
    return pd.DataFrame(rows)


def runs_table(df):
    runs=[]
    for (sequence,k),d in df.groupby(['sequence','horizon'],sort=True):
        d=d.sort_values('frame_idx').reset_index(drop=True)
        gaps=d.frame_idx.diff().dropna()
        stride=int(gaps.mode().iloc[0]) if len(gaps) else 1
        for label in ('label_low_baseline','label_low_epipolar'):
            active=d[label].eq(1).fillna(False).to_numpy(bool)
            starts=d.frame_idx.to_numpy(int)
            i=0
            while i<len(d):
                if not active[i]:i+=1;continue
                j=i+1
                while j<len(d) and active[j] and starts[j]-starts[j-1]==stride:j+=1
                runs.append(dict(sequence=sequence,horizon=int(k),target=label,
                    start_frame=int(starts[i]),end_frame=int(starts[j-1]),
                    sampled_groups=j-i,stride=stride,
                    start_t_ns=str(int(d.iloc[i].t_ns))))
                i=j
    return pd.DataFrame(runs,columns=['sequence','horizon','target','start_frame',
        'end_frame','sampled_groups','stride','start_t_ns'])


def associations(df):
    records=[]
    for (seq,k),g in df.groupby(['sequence','horizon'],sort=True):
        for label in ('label_low_baseline','label_low_epipolar'):
            eligible=g.predictors_available.eq(1)&g[label].notna()
            subset=g.loc[eligible]
            for feat in PREDICTORS:
                x=subset[feat].to_numpy(float)
                y=subset[label].to_numpy(int)
                rho=float(spearmanr(x,y).statistic) if len(x)>=10 and len(np.unique(x))>1 and len(np.unique(y))>1 else np.nan
                records.append(dict(sequence=seq,horizon=int(k),target=label,
                    feature=feat,n=len(x),n_event=int(y.sum()),spearman=rho))
    return pd.DataFrame(records)


def build(args):
    if len(args.features)!=len(args.geometry) or not args.features:
        raise ValueError('Supply one EXP003 and one EXP004-B CSV per sequence')
    fs={}; gs={}; hashes={}
    for p in map(Path,args.features):
        d=read_features(p);name=str(d.sequence.iloc[0])
        if name in fs:raise ValueError('Duplicate features for same sequence')
        fs[name]=p;hashes[str(p)]=hash_file(p)
    for p in map(Path,args.geometry):
        d=read_geometry(p);name=str(d.sequence.iloc[0])
        if name in gs:raise ValueError('Duplicate geometry for same sequence')
        gs[name]=p;hashes[str(p)]=hash_file(p)
    if set(fs)!=set(gs):raise ValueError(f'Sequence pairing mismatch: EXP003={sorted(fs)}, GT={sorted(gs)}')
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    all_rows=[]
    for seq in sorted(fs):
        d=merge_sequence(fs[seq],gs[seq]);all_rows.append(d)
        path=out/seq
        path.mkdir(parents=True,exist_ok=True)
        d.to_csv(path/'geometry_prediction_labels.csv',index=False)
    combined=pd.concat(all_rows,ignore_index=True)
    result=summary_table(combined)
    result.to_csv(out/'dataset_summary.csv',index=False)
    runs_table(combined).to_csv(out/'event_runs.csv',index=False)
    associations(combined).to_csv(out/'feature_event_associations.csv',index=False)
    (combined.groupby(['sequence','horizon','gt_status'],dropna=False)
        .size().rename('n').reset_index()).to_csv(out/'gt_status_counts.csv',index=False)
    (out/'dataset_manifest.json').write_text(json.dumps(dict(
        experiment='EXP005-A',sequences=sorted(fs),source_hashes_sha256=hashes,
        feature_columns=PREDICTORS,
        excluded_online_fields=['baseline_m','gt_inlier_ratio_survivors','gt_parallax_deg',
             'gt_positive_depth_fraction','survival_rate','s_5','s_10'],
        baseline_threshold_m=BASELINE_M,
        epipolar_inlier_ratio_threshold=GT_EPI_LIMIT,
        gt_source_epipolar_error_threshold_px=EPI_PX,
        min_geometry_correspondences=MIN_MATCHES,
        join_keys=['sequence','frame_idx'],horizons=list(HORIZONS),
        timestamp_policy='exact_int64_equality_start_t_ns_vs_t_ns',
        geometry_sampling='sampled cohort start frames; no interpolation',
        task_definitions={
            'low_baseline':'label where a reference GT baseline is available (including low_gt_baseline rows)',
            'low_epipolar':'label only when status=gt_evaluated and n_survive >= 12; low baseline, missing GT and censored remain NA'
        },
        note='Exploratory label construction only; no model training or independent held-out generalization. GT/retrospective survival never enter feature_columns.'
    ),indent=2,ensure_ascii=False),encoding='utf8')
    print(result[['sequence','horizon','n_geometry_rows','label_low_baseline_n_events',
      'label_low_baseline_n_evaluable','label_low_epipolar_n_events',
      'label_low_epipolar_n_evaluable']].to_string(index=False))
    print('EXP005-A labels and audits written to',out)


def main():
    pa=argparse.ArgumentParser(description='EXP005-A align EXP003 past-only features and EXP004-B retrospective GT labels')
    pa.add_argument('--features',nargs='+',type=Path,required=True)
    pa.add_argument('--geometry',nargs='+',type=Path,required=True)
    pa.add_argument('--output-dir',type=Path,required=True)
    build(pa.parse_args())


if __name__=='__main__':main()
