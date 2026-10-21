"""Step 32B: exploratory Sentinel-2 next-year width, water-edge, vegetation forecasts.

Targets are SATELLITE-DERIVED PROXIES, not survey-validated bank erosion or
independent vegetation removal. The three morphology outputs are algebraically
related: width=end-side edge - start-side edge (within rounding).

Temporal protocol:
 train: target years 2019--2023
 validate/select: target year 2024 ONLY
 test: target year 2025, once AFTER validation model selection, model refit through 2024
 forecast: retrain selected model on eligible through target 2025; source-only 2025
No target-year features/QA in predictors. Evaluation labels can only be scored
when target observations are available/QA-screened (selection bias warning).

Input files produced in Steps 27B/28B. Only creates NEW files with prefix 32b_*.
Example: python 16_main_forecasting_targets_v1.py
Optional synthetic integration test: python 16_main_forecasting_targets_v1.py --root PATH
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
MODEL_ORDER = ['Persistence', 'TrainingMean', 'Ridge', 'RandomForest', 'ExtraTrees']
MIN_PAIR_COVERAGE_PCT = 70.0
MIN_PAIR_PIXELS = 30
WIDTH_TARGETS = {
    'width_delta_m': ('width_delta_m', 'metres'),
    'start_edge_delta_m': ('start_edge_delta_m', 'metres'),
    'end_edge_delta_m': ('end_edge_delta_m', 'metres'),
}
VEG_TARGETS = {
    'vegetation_loss_pct': ('vegetation_loss_pct', '% common-valid land'),
    'median_ndvi_delta': ('median_ndvi_delta', 'NDVI units'),
}
WIDTH_FEATURES = [
    'source_year', 'chainage_m', 'source_width_m', 'source_start_edge_m',
    'source_end_edge_m', 'source_valid_transect_pct',
    'source_index_disagreement_m', 'source_water_run_count',
    'source_day_of_year', 'previous_width_change_m',
]
VEG_FEATURES = [
    'source_year', 'reach_number', 'source_day_of_year', 'source_median_ndvi',
    'source_median_evi', 'source_veg_pct_valid_land',
    'source_observed_veg_area_ha', 'source_land_coverage_pct',
    'previous_median_ndvi_change', 'previous_veg_pct_change',
]


def require(df, columns, label):
    absent = sorted(set(columns) - set(df.columns))
    if absent:
        raise ValueError(f'{label}: missing columns {absent}')


def models():
    return {
        'Ridge': Pipeline([
            ('impute', SimpleImputer(strategy='median', add_indicator=False)),
            ('scale', StandardScaler()), ('model', Ridge(alpha=10.0))]),
        'RandomForest': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('model', RandomForestRegressor(n_estimators=160, min_samples_leaf=3,
                                          max_features=0.8, n_jobs=-1, random_state=SEED))]),
        'ExtraTrees': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('model', ExtraTreesRegressor(n_estimators=160, min_samples_leaf=3,
                                        max_features=0.8, n_jobs=-1, random_state=SEED))]),
    }


def prep_dates(vegetation):
    date = vegetation[['year', 'observation_date']].drop_duplicates()
    if date['year'].duplicated().any():
        raise ValueError('More than one vegetation observation date per year')
    date['day_of_year'] = pd.to_datetime(date['observation_date'], errors='raise').dt.dayofyear
    return date[['year', 'day_of_year']]


def load_width(root, dates):
    path = root / 'outputs/reports/27_s2_transect_widths_2018_2025.csv'
    df = pd.read_csv(path)
    needed = ['year','transect_id','reach_id','chainage_m','qa_pass',
              'apparent_width_screened_m','start_side_water_edge_offset_m',
              'end_side_water_edge_offset_m', 'observed_valid_transect_pct',
              'index_disagreement_intersection_m','water_run_count']
    require(df, needed, 'Step27 annual width')
    if len(df) != 271 * 8 or df['year'].nunique() != 8:
        raise ValueError(f'Unexpected width observations ({len(df)}): expected 271 x 8; audit annual input')
    if df.duplicated(['transect_id','year']).any():
        raise ValueError('Duplicate transect-year records')
    df = df.sort_values(['transect_id','year']).copy()
    for c in needed:
        if c not in {'transect_id','reach_id'}:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    if not df['qa_pass'].isin([0,1]).all():
        raise ValueError('Unexpected QA flags')
    df['prev_year'] = df.groupby('transect_id')['year'].shift(1)
    df['prev_width'] = df.groupby('transect_id')['apparent_width_screened_m'].shift(1)
    df['previous_width_change_m'] = np.where(
        df['prev_year'] == df['year']-1,
        df['apparent_width_screened_m']-df['prev_width'], np.nan)
    df = df.merge(dates, on='year', validate='many_to_one')
    name = {
        'year':'source_year', 'apparent_width_screened_m':'source_width_m',
        'start_side_water_edge_offset_m':'source_start_edge_m',
        'end_side_water_edge_offset_m':'source_end_edge_m',
        'observed_valid_transect_pct':'source_valid_transect_pct',
        'index_disagreement_intersection_m':'source_index_disagreement_m',
        'water_run_count':'source_water_run_count', 'day_of_year':'source_day_of_year'
    }
    sources = df.rename(columns=name).copy()
    sources = sources.loc[sources['qa_pass'] == 1].copy()
    sources['target_year'] = sources['source_year']+1
    tgt = df[['transect_id','year','qa_pass','apparent_width_screened_m',
              'start_side_water_edge_offset_m','end_side_water_edge_offset_m']].rename(columns={
                  'year':'target_year','qa_pass':'target_qa_pass',
                  'apparent_width_screened_m':'target_width_m',
                  'start_side_water_edge_offset_m':'target_start_edge_m',
                  'end_side_water_edge_offset_m':'target_end_edge_m'})
    matched = sources.merge(tgt, on=['transect_id','target_year'], how='inner',
                            validate='one_to_one')
    matched = matched.loc[matched['target_qa_pass']==1].copy()
    matched['width_delta_m'] = matched['target_width_m']-matched['source_width_m']
    matched['start_edge_delta_m'] = matched['target_start_edge_m']-matched['source_start_edge_m']
    matched['end_edge_delta_m'] = matched['target_end_edge_m']-matched['source_end_edge_m']
    identity = np.abs(matched['end_edge_delta_m']-matched['start_edge_delta_m']-
                      matched['width_delta_m'])
    if len(matched) == 0 or identity.max() > .02:
        raise ValueError(f'Width-edge geometry consistency failed: {identity.max()}')
    if not matched['target_year'].between(2019,2025).all():
        raise ValueError('Unexpected width target year')
    future = sources.loc[sources['source_year']==2025].copy()
    if not future['target_year'].eq(2026).all():
        raise ValueError('Forecast year bug')
    return matched, future, float(identity.max())


def load_veg(root):
    annual = pd.read_csv(root/'outputs/reports/28_s2_annual_riparian_vegetation.csv')
    pairs = pd.read_csv(root/'outputs/reports/28_s2_riparian_change_by_reach.csv')
    require(annual,['year','reach_id','zone','observation_date','median_ndvi','median_evi',
                    'veg_pct_of_valid_land','observed_veg_area_ha',
                    'observed_land_coverage_pct'], 'Step28 annual')
    require(pairs, ['year_a','year_b','reach_id','zone','median_ndvi_delta',
                    'vegetation_loss_ha','matched_valid_land_pixels',
                    'matched_land_coverage_pct'], 'Step28 pairs')
    a = annual.loc[annual['zone']=='0-100m'].copy()
    p = pairs.loc[(pairs['zone']=='0-100m') &
                  (pairs['year_b'] == pairs['year_a'] + 1)].copy()
    if len(a) != 8*28 or len(p) != 7*28:
        raise ValueError(f'Unexpected vegetation rows annual={len(a)}, pairs={len(p)}')
    if a.duplicated(['year','reach_id']).any() or p.duplicated(['year_a','reach_id']).any():
        raise ValueError('Duplicate vegetation rows')
    for c in ['year','median_ndvi','median_evi','veg_pct_of_valid_land',
              'observed_veg_area_ha','observed_land_coverage_pct']:
        a[c] = pd.to_numeric(a[c], errors='coerce')
    for c in ['year_a','year_b','median_ndvi_delta','vegetation_loss_ha',
              'matched_valid_land_pixels','matched_land_coverage_pct']:
        p[c] = pd.to_numeric(p[c], errors='coerce')
    a['source_day_of_year'] = pd.to_datetime(a['observation_date'],errors='raise').dt.dayofyear
    a['reach_number'] = a['reach_id'].str.extract(r'R(\d+)',expand=False).astype(int)
    a = a.sort_values(['reach_id','year'])
    a['previous_year'] = a.groupby('reach_id')['year'].shift(1)
    a['previous_median_ndvi'] = a.groupby('reach_id')['median_ndvi'].shift(1)
    a['previous_veg_pct'] = a.groupby('reach_id')['veg_pct_of_valid_land'].shift(1)
    a['previous_median_ndvi_change'] = np.where(
        a['previous_year']==a['year']-1,
        a['median_ndvi']-a['previous_median_ndvi'],np.nan)
    a['previous_veg_pct_change'] = np.where(
        a['previous_year']==a['year']-1,
        a['veg_pct_of_valid_land']-a['previous_veg_pct'],np.nan)
    a = a.rename(columns={
        'year':'source_year', 'median_ndvi':'source_median_ndvi',
        'median_evi':'source_median_evi', 'veg_pct_of_valid_land':'source_veg_pct_valid_land',
        'observed_veg_area_ha':'source_observed_veg_area_ha',
        'observed_land_coverage_pct':'source_land_coverage_pct'})
    a['target_year'] = a['source_year']+1
    p = p.rename(columns={'year_a':'source_year','year_b':'target_year'})
    p['vegetation_loss_pct'] = (
        100.0 * p['vegetation_loss_ha'] /
        (p['matched_valid_land_pixels'] * 0.01).replace(0, np.nan))
    # Supplied loss_ha is already rounded to 0.001ha: allow a small rounding tolerance.
    if (p['vegetation_loss_pct'].dropna() > 100.1).any() or (p['vegetation_loss_pct'].dropna() < -.01).any():
        raise ValueError('Invalid vegetation loss percentage')
    matched = a.merge(p[['reach_id','source_year','target_year',
                         'median_ndvi_delta','vegetation_loss_pct',
                         'matched_valid_land_pixels','matched_land_coverage_pct']],
                      on=['reach_id','source_year','target_year'],
                      how='inner',validate='one_to_one')
    ok = ((matched['source_land_coverage_pct']>=MIN_PAIR_COVERAGE_PCT) &
          (matched['matched_land_coverage_pct']>=MIN_PAIR_COVERAGE_PCT) &
          (matched['matched_valid_land_pixels']>=MIN_PAIR_PIXELS))
    matched = matched.loc[ok].copy()
    future = a.loc[(a['source_year']==2025) &
                   (a['source_land_coverage_pct']>=MIN_PAIR_COVERAGE_PCT)].copy()
    if future.empty:
        raise ValueError('No 2025 usable vegetation source observations')
    return matched,future,a


def numeric_features(df, feature_columns, target_label):
    arr = df[feature_columns].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    if arr.isna().all().any():
        bad=arr.columns[arr.isna().all()].tolist()
        raise ValueError(f'{target_label}: features all missing: {bad}')
    return arr.to_numpy(dtype=float)


def preds_for(train,evaluation,features,target):
    Xtr = numeric_features(train,features,target)
    Xev = numeric_features(evaluation,features,target)
    y = pd.to_numeric(train[target],errors='coerce').to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError(f'Nonfinite training target: {target}')
    output = {'Persistence':np.zeros(len(evaluation)),
              'TrainingMean':np.full(len(evaluation),float(y.mean()))}
    fitted = {}
    for name,model in models().items():
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',category=UserWarning, message='Skipping features without any observed values')
            model.fit(Xtr,y)
        output[name] = model.predict(Xev)
        fitted[name] = model
    return output, fitted


def fit_selected(train, evaluate, features,target,name):
    if name=='Persistence':
        return np.zeros(len(evaluate),dtype=float), None
    if name=='TrainingMean':
        return np.full(len(evaluate),float(train[target].mean())), None
    m = models()[name]
    m.fit(numeric_features(train,features,target),
          train[target].to_numpy(dtype=float))
    return m.predict(numeric_features(evaluate,features,target)),m


def metrics_row(domain,target,unit,split,truth,pred,model):
    y=np.asarray(truth,float); yh=np.asarray(pred,float)
    if not np.isfinite(y).all() or not np.isfinite(yh).all():
        raise ValueError(f'Nonfinite {domain}/{target}/{split}/{model}')
    return dict(domain=domain,target=target,unit=unit,split=split,model=model,
                n=len(y),mae=float(mean_absolute_error(y,yh)),
                rmse=float(np.sqrt(mean_squared_error(y,yh))),
                bias=float(np.mean(yh-y)),
                observed_mean_abs_change=float(np.mean(np.abs(y))))


def task(domain,target,unit,matched,future,features,outputs,models_dir):
    train=matched.loc[matched['target_year'].between(2019,2023)].copy()
    val=matched.loc[matched['target_year']==2024].copy()
    test=matched.loc[matched['target_year']==2025].copy()
    for label,df in [('train',train),('validation',val),('test',test)]:
        if len(df)<8:
            raise ValueError(f'{domain}/{target}: too few {label} observations: {len(df)}')
    for label,df in [('training',train),('validation',val),('test',test)]:
        if not np.isfinite(pd.to_numeric(df[target],errors='coerce')).all():
            raise ValueError(f'{domain}/{target}: invalid {label} target')
    va_predictions,_ = preds_for(train,val,features,target)
    for model,yp in va_predictions.items():
        outputs['metrics'].append(metrics_row(domain,target,unit,'validation_2024',
                                              val[target],yp,model))
    # Selection is strictly based on the 2024 validation MAE, never 2025 test.
    val_r = [r for r in outputs['metrics'] if r['domain']==domain and
             r['target']==target and r['split']=='validation_2024']
    winner=min(val_r,key=lambda r:(r['mae'], MODEL_ORDER.index(r['model'])))['model']
    through24=matched.loc[matched['target_year'].between(2019,2024)].copy()
    test_preds,_ = preds_for(through24,test,features,target)
    for model,yp in test_preds.items():
        outputs['metrics'].append(metrics_row(domain,target,unit,'test_2025',
                                              test[target],yp,model))
    for k,row in enumerate(test.to_dict('records')):
        record={"domain":domain,"target":target,"target_year":2025,
                "source_year":2024,"selected_model_2024":winner,
                "observed":float(row[target]), "unit":unit}
        record['location_id']=row['transect_id'] if domain=='morphology' else row['reach_id']
        for mod,pred in test_preds.items():
            record['pred_'+mod] = float(pred[k])
        outputs['holdout'].append(record)
    all_train=matched.loc[matched['target_year'].between(2019,2025)].copy()
    forward,m=fit_selected(all_train,future,features,target,winner)
    if domain=='vegetation' and target=='vegetation_loss_pct':
        forward=np.clip(forward,0,100)
    if domain=='vegetation' and target=='median_ndvi_delta':
        forward=np.clip(forward,-2,2)
    if domain=='morphology':
        # Offsets/width should stay within 300m transect, but unconstrained
        # independent models can conflict. Report inconsistent outputs later.
        forward=np.clip(forward,-300,300)
    if m is not None:
        models_dir.mkdir(parents=True,exist_ok=True)
        joblib.dump({'model':m,'features':features,'target':target,
                     'trained_through_target_year':2025,'unit':unit,
                     'validation_selected_model':winner},
                    models_dir/f'32b_{domain}_{target}_{winner}.joblib')
    for k,row in enumerate(future.to_dict('records')):
        outputs['future'].append({
            'domain':domain,'target':target,'source_year':2025,'target_year':2026,
            'location_id':row['transect_id'] if domain=='morphology' else row['reach_id'],
            'reach_id':row['reach_id'], 'selected_model':winner,
            'forecast_delta':float(forward[k]),'unit':unit,
            'source_qa_pass':1 if domain=='morphology' else int(row['source_land_coverage_pct']>=70),
            'future_qa_known':False,
            'interpretation':'EXPLORATORY_SATELLITE_DERIVED_PROXY_NOT_VALIDATED_BANK_OR_DEFORESTATION'
        })
    outputs['selection'].append({'domain':domain,'target':target,'unit':unit,
                                 'winner_selected_only_on_2024':winner,
                                 'n_train_through_2023':len(train),
                                 'n_validation_2024':len(val),
                                 'n_test_2025':len(test),
                                 'n_future_from_2025_source':len(future)})
    val_mae=next(r['mae'] for r in val_r if r['model']==winner)
    te_mae=next(r['mae'] for r in outputs['metrics'] if r['target']==target and
                r['domain']==domain and r['split']=='test_2025' and r['model']==winner)
    print(f'  {target:23s} | train={len(train):4d} val={len(val):3d} test={len(test):3d} '
          f'| chosen={winner:13s} | val MAE={val_mae:.4f} | test MAE={te_mae:.4f} '
          f'| forecasts={len(future)}')


def save_outputs(root,outputs,identity,max_forecast_width_diff,source_width):
    report = root / 'outputs/reports'
    report.mkdir(parents=True,exist_ok=True)
    for filename,rows in [('32b_main_forecast_model_metrics.csv',outputs['metrics']),
                          ('32b_main_forecast_model_selection.csv',outputs['selection']),
                          ('32b_main_forecast_holdout_predictions_2025.csv',outputs['holdout']),
                          ('32b_main_forecast_2026_predictions.csv',outputs['future'])]:
        pd.DataFrame(rows).to_csv(report/filename,index=False)
        print('Output:',report/filename)
    # Direct width and separately predicted edges need not agree.
    # Preserve both with explicit consistency warning; DO NOT silently convert
    # their independent outputs into one surveyed/physically consistent bankline.
    future = pd.DataFrame(outputs['future'])
    geom = future.loc[future['domain']=='morphology'].pivot(
        index='location_id', columns='target', values='forecast_delta').reset_index()
    geom = geom.merge(source_width[['transect_id','reach_id','source_width_m',
                                    'source_start_edge_m','source_end_edge_m']],
                      left_on='location_id',right_on='transect_id',
                      validate='one_to_one')
    geom['edge_implied_width_delta_m'] = geom['end_edge_delta_m']-geom['start_edge_delta_m']
    geom['independent_width_vs_edge_discrepancy_m'] = (
        geom['width_delta_m'] - geom['edge_implied_width_delta_m']).abs()
    geom['direct_predicted_width_m'] = geom['source_width_m'] + geom['width_delta_m']
    geom['edge_implied_predicted_width_m'] = (
        geom['source_width_m']+geom['edge_implied_width_delta_m'])
    geom['predicted_start_edge_offset_m'] = geom['source_start_edge_m'] + geom['start_edge_delta_m']
    geom['predicted_end_edge_offset_m'] = geom['source_end_edge_m'] + geom['end_edge_delta_m']
    geom['geometry_review_required'] = (
        (geom['independent_width_vs_edge_discrepancy_m'] > 5.0) |
        (geom['direct_predicted_width_m'] <= 0.0) |
        (geom['edge_implied_predicted_width_m'] <= 0.0) |
        (geom['direct_predicted_width_m'] > 300.0) |
        (geom['edge_implied_predicted_width_m'] > 300.0) |
        (geom['predicted_start_edge_offset_m'].abs() > 150.0) |
        (geom['predicted_end_edge_offset_m'].abs() > 150.0))
    gp=report/'32b_morphology_2026_geometry_consistency.csv'
    geom.to_csv(gp,index=False)
    print('Output:',gp)
    print(f'2026 geometric review flags: {int(geom.geometry_review_required.sum())} / {len(geom)}')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for target_name in ['width_delta_m','vegetation_loss_pct','median_ndvi_delta']:
            part=pd.DataFrame(outputs['metrics'])
            part=part[part['target']==target_name]
            if part.empty:
                continue
            v=part[part['split']=='validation_2024'].set_index('model')['mae']
            t=part[part['split']=='test_2025'].set_index('model')['mae']
            x=np.arange(len(MODEL_ORDER))
            fig,ax=plt.subplots(figsize=(10.8,4.7))
            ax.bar(x-.19,[v[k] for k in MODEL_ORDER],.38,label='2024 validation')
            ax.bar(x+.19,[t[k] for k in MODEL_ORDER],.38,label='2025 test')
            ax.set_xticks(x,MODEL_ORDER,rotation=22,ha='right')
            ax.set_ylabel('MAE ('+('metres' if target_name.endswith('_m') else
                                    '% common-valid land' if target_name=='vegetation_loss_pct' else 'NDVI units')+')')
            ax.set_title('Exploratory 1-year Sentinel-2 forecast: '+target_name)
            ax.legend(loc='best')
            ax.grid(axis='y',alpha=.2)
            fig.text(.5,.01,'Validation-only selection | Holdout 2025 | Satellite proxy, not surveyed bank/deforestation',
                     ha='center',fontsize=8)
            fig.subplots_adjust(bottom=.26)
            figfile=root/'outputs/figures'/f'32b_model_comparison_{target_name}.png'
            figfile.parent.mkdir(parents=True,exist_ok=True)
            fig.savefig(figfile,dpi=170,bbox_inches='tight')
            plt.close(fig)
            print('Figure:',figfile)
    except ImportError:
        print('WARNING: matplotlib unavailable; CSV metrics generated but not figures')

    manifest = {
        'generated_utc':datetime.now(timezone.utc).isoformat(),
        'research_status':'EXPLORATORY_NOT_SURVEY_VALIDATED',
        'training_target_years':[2019,2020,2021,2022,2023],
        'validation_target_year':2024,'test_target_year':2025,
        'final_forecast_target_year':2026,
        '2025_test_not_used_for_model_selection':True,
        '2026_forecast_uses_source_2025_observations_only':True,
        'width_edge_identity_max_difference_on_observations_m':identity,
        'independently_selected_edge_width_forecast_max_inconsistency_m':max_forecast_width_diff,
        'warning_geometry':'Width should equal end_edge minus start_edge. Separate models can violate identity; DO NOT use inconsistent triple as actual bank geometry.',
        'warning_quality':'Target-year QA known only in hindcast evaluation. 2026 target-year observation/QA unknown. Missingness may bias evaluation.',
        'warning_vegetation':'Vegetation loss % refers to common-valid future image land pixels, which are unknown in 2026. No physical deforestation confirmation.',
        'warning_scene':'Single-date annual Sentinel scenes and seasonal/water-stage variation confound true change; repeated spatial units are correlated.',
        'warning_model':'Model selected by lowest 2024 MAE, persistence is an untrained baseline. High quality holdout does not guarantee operational future accuracy.',
        'selection':outputs['selection'],
    }
    target=report/'32b_main_forecast_metadata.json'
    target.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Output:',target)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    args=ap.parse_args()
    root=args.root.resolve()
    veg_file=root/'outputs/reports/28_s2_annual_riparian_vegetation.csv'
    date_table=prep_dates(pd.read_csv(veg_file).loc[lambda d:d['zone']=='0-100m'])
    wm,wf,identity=load_width(root,date_table)
    vm,vf,_=load_veg(root)
    print('========== STEP 32B | MAIN FORECAST TARGETS ==========')
    print(f'Observed QA width/edge transitions={len(wm)} | vegetation high-Q transitions={len(vm)}')
    print(f'Width-edge observed identity max discrepancy={identity:.4f} m (rounding, NOT bank accuracy)')
    if len(wm)!=1006:
        print(f'WARNING: width transition count differs from Step32A expected 1006 ({len(wm)})')
    if len(vm)>196:
        raise ValueError('More than 196 vegetation transitions')
    outputs={'metrics':[],'selection':[],'holdout':[],'future':[]}
    models_dir=root/'models/main_forecasting_v1'
    for target,(_,unit) in WIDTH_TARGETS.items():
        task('morphology',target,unit,wm,wf,WIDTH_FEATURES,outputs,models_dir)
    for target,(_,unit) in VEG_TARGETS.items():
        task('vegetation',target,unit,vm,vf,VEG_FEATURES,outputs,models_dir)
    future=pd.DataFrame(outputs['future'])
    piv=future[future.domain=='morphology'].pivot(index='location_id',columns='target',values='forecast_delta')
    if {'width_delta_m','start_edge_delta_m','end_edge_delta_m'}.issubset(piv.columns):
        diff=(piv['end_edge_delta_m']-piv['start_edge_delta_m']-piv['width_delta_m']).abs()
        mx=float(diff.max())
        print(f'2026 independently predicted edge-width geometry discrepancy: MAX={mx:.3f} m')
        if mx>0.02:
            print('WARNING: independent width and edge models NOT internally coherent. Treat as separate target benchmarks, not bank geometry.')
    else:
        mx=None
    print('NOTE: 2026 predictions are exploratory next-observation scenarios, not measured bank erosion.')
    save_outputs(root,outputs,identity,mx,wf)
    print('STEP 32B COMPLETE | VALIDATION 2024 / TEST 2025 / EXPLORATORY 2026')

if __name__=='__main__':
    main()
