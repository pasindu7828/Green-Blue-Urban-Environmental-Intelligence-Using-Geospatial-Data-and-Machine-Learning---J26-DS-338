"""Step 26: leakage-aware, exploratory Sentinel-2 reach-change model benchmark.

Run from any working directory after Step 25:
 python River_Corridor_Research/src/part4_ml_forecasting/12_sentinel2_ml_benchmark_v2.py

Input (read-only): outputs/reports/25_s2_reach_transition_ml_all.csv
Output (new): 26_s2_v2_* files in outputs/{reports,figures} and models.

Outcome = next-year matched-valid-pixel *classification* water area change
(ha/reach), NOT validated bank erosion, real bankline migration, or flood risk.
Never include future QA metrics or target-date features in ML predictors.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'outputs' / 'reports' / '25_s2_reach_transition_ml_all.csv'
REPORTS = ROOT / 'outputs' / 'reports'
FIGURES = ROOT / 'outputs' / 'figures'
MODELS = ROOT / 'models'

FEATURES = [
    'x_reach_chainage_km', 'x_source_year', 'x_source_day_of_year',
    'x_source_water_ha', 'x_source_water_pct_valid', 'x_source_valid_pct',
    'x_source_nonwater_ndvi_median', 'x_source_nonwater_ndbi_median',
    'x_source_vegetated_pct_nonwater',
]
TARGET = 'y_matched_water_delta_ha'
MODEL_ORDER = ['Persistence', 'Training mean', 'Ridge', 'Random Forest', 'Extra Trees']
MODEL_FILE_STEMS = {'Ridge':'ridge', 'Random Forest':'random_forest', 'Extra Trees':'extra_trees'}
SEED = 42


def load_and_verify():
    if not SOURCE.is_file():
        raise FileNotFoundError(f'Step25 source missing: {SOURCE}')
    df = pd.read_csv(SOURCE)
    needed = FEATURES + [TARGET, 'reach_id', 'source_year', 'target_year', 'split',
                         'high_coverage_for_evaluation', 'common_valid_pct',
                         'water_domain_coverage_pct']
    absent = sorted(set(needed)-set(df.columns))
    if absent:
        raise ValueError(f'Missing required Step25 columns: {absent}')
    if len(df) != 196:
        raise ValueError(f'Expected 196 transitions from 28 reaches x 7 years, got {len(df)}')
    if df.duplicated(['reach_id','target_year']).any():
        raise ValueError('Duplicate reach/year labels; investigation required')
    if df['reach_id'].nunique() != 28:
        raise ValueError('Expected 28 reaches')
    expect = {'train':(140,88),'validation':(28,13),'test':(28,17)}
    for split, (all_n, quality_n) in expect.items():
        subset = df.loc[df['split']==split]
        if len(subset)!=all_n or int((subset['high_coverage_for_evaluation']==1).sum())!=quality_n:
            raise ValueError(f'Unexpected Step25 split {split}: {len(subset)}, quality={int((subset["high_coverage_for_evaluation"]==1).sum())}; expected {all_n},{quality_n}')
    if set(df['split'].unique()) != set(expect):
        raise ValueError('Unknown dataset split')
    if ((df['split']=='train') & (df['target_year']>2023)).any() or \
        ((df['split']=='validation') & (df['target_year']!=2024)).any() or \
        ((df['split']=='test') & (df['target_year']!=2025)).any():
        raise ValueError('Temporal split mismatch')
    if (df['source_year'].astype(int)+1 != df['target_year'].astype(int)).any():
        raise ValueError('Nonconsecutive source/target years')
    for f in FEATURES + [TARGET]:
        df[f] = pd.to_numeric(df[f], errors='coerce')
        df[f] = df[f].replace([np.inf, -np.inf], np.nan)
    if df[TARGET].isna().any():
        raise ValueError('Target has missing/nonfinite values, cannot evaluate')
    if df[FEATURES].isna().any().any():
        # Step25 user report says 0 missing; do not silently invent numeric data.
        miss = df[FEATURES].isna().sum()
        raise ValueError(f'Missing/nonfinite predictors found; inspect first: {miss[miss>0].to_dict()}')
    # Explicitly reject non-boolean QA flags.
    if not df['high_coverage_for_evaluation'].isin([0,1]).all():
        raise ValueError('Quality flags must be 0/1')
    return df


def make_models():
    # Fixed in advance, NOT optimized against validation or test.
    return {
        'Ridge': Pipeline([('imputer',SimpleImputer(strategy='median')),
                           ('scale',StandardScaler()), ('model',Ridge(alpha=10.0))]),
        'Random Forest': Pipeline([('imputer',SimpleImputer(strategy='median')),
                                   ('model',RandomForestRegressor(
                                       n_estimators=250, min_samples_leaf=3,
                                       max_features=0.85, random_state=SEED, n_jobs=-1))]),
        'Extra Trees': Pipeline([('imputer',SimpleImputer(strategy='median')),
                                 ('model',ExtraTreesRegressor(
                                     n_estimators=250, min_samples_leaf=3,
                                     max_features=0.85, random_state=SEED, n_jobs=-1))]),
    }


def predictions(models, training, evaluation):
    x = training[FEATURES].to_numpy(dtype=float)
    y = training[TARGET].to_numpy(dtype=float)
    x_e = evaluation[FEATURES].to_numpy(dtype=float)
    out = {
        'Persistence':np.zeros(len(evaluation),dtype=float),
        'Training mean':np.full(len(evaluation),y.mean(),dtype=float),
    }
    for name, model in models.items():
        model.fit(x,y)
        out[name] = model.predict(x_e)
    return out


def metrics_table(split, eval_df, preds):
    y = eval_df[TARGET].to_numpy(dtype=float)
    quality = eval_df['high_coverage_for_evaluation'].to_numpy(dtype=int)==1
    result=[]
    for scope, ix in [('high_coverage',quality), ('all',np.ones(len(eval_df),bool))]:
        if not ix.any():
            raise ValueError(f'No scoring observations {split}/{scope}')
        for name in MODEL_ORDER:
            yhat=np.asarray(preds[name],dtype=float)
            if not np.isfinite(yhat).all():
                raise ValueError(f'Nonfinite predictions: {name}')
            result.append({
                'split':split,'evaluation_scope':scope,'model':name,
                'n':int(ix.sum()),'mae_ha':round(float(mean_absolute_error(y[ix],yhat[ix])),6),
                'rmse_ha':round(float(np.sqrt(mean_squared_error(y[ix],yhat[ix]))),6),
                'bias_ha':round(float(np.mean(yhat[ix]-y[ix])),6),
                'observed_abs_change_mean_ha':round(float(np.mean(np.abs(y[ix]))),6),
            })
    return result


def pred_rows(split, observations, preds, trained_through):
    rows=[]
    for n, (_,row) in enumerate(observations.iterrows()):
        r={
            'split':split,'reach_id':row['reach_id'],
            'source_year':int(row['source_year']), 'target_year':int(row['target_year']),
            'trained_through_target_year':trained_through,
            'observed_water_delta_ha':round(float(row[TARGET]),6),
            'high_coverage_for_evaluation':int(row['high_coverage_for_evaluation']),
            'common_valid_pct':float(row['common_valid_pct']),
            'water_domain_coverage_pct':float(row['water_domain_coverage_pct']),
        }
        for name in MODEL_ORDER:
            slug=name.lower().replace(' ','_')
            r['prediction_'+slug+'_ha']=round(float(preds[name][n]),6)
            r['abs_error_'+slug+'_ha']=round(float(abs(preds[name][n]-row[TARGET])),6)
        rows.append(r)
    return rows


def chart(metrics, winner):
    names=MODEL_ORDER
    fig, axes=plt.subplots(1,2,figsize=(14,5.2),sharey=True)
    for ax,split in zip(axes,['validation','test']):
        rows={r['model']:r for r in metrics if r['split']==split and r['evaluation_scope']=='high_coverage'}
        values=[rows[m]['mae_ha'] for m in names]
        bars=ax.bar(range(len(names)),values)
        bars[names.index(winner)].set_edgecolor('black')
        bars[names.index(winner)].set_linewidth(2.5)
        ax.set_xticks(range(len(names)),names,rotation=27,ha='right',fontsize=9)
        ax.set_title(f'{2024 if split=="validation" else 2025} {split.title()} | high-coverage n={rows[names[0]]["n"]}')
        ax.set_ylabel('Mean absolute error (ha per reach)')
        ax.grid(axis='y',alpha=0.25)
        ax.set_axisbelow(True)
        for b,v in zip(bars,values):
            ax.annotate(f'{v:.3f}', (b.get_x()+b.get_width()/2,b.get_height()),
                        ha='center',va='bottom',fontsize=8,xytext=(0,3),textcoords='offset points')
    fig.suptitle('Exploratory Sentinel-2 classified-water change | Fixed models and temporal holdout', fontsize=13)
    fig.text(0.5,.01,'Outlined model chosen using 2024 validation only; 2025 test never used for selection. NOT erosion accuracy.',ha='center',fontsize=9)
    fig.subplots_adjust(bottom=.28,top=.82,wspace=.22)
    path=FIGURES/'26_s2_v2_model_comparison.png'
    fig.savefig(path,dpi=170,bbox_inches='tight')
    plt.close(fig)
    return path


def save_feature_summary(models, training):
    # Effect/importance diagnostics of refitted ML on observations through 2024.
    rows=[]
    for name, model in models.items():
        estimator=model.named_steps['model']
        if hasattr(estimator,'feature_importances_'):
            scores=estimator.feature_importances_
            kind='impurity_importance_exploratory'
        elif hasattr(estimator,'coef_'):
            scores=estimator.coef_
            kind='scaled_ridge_coefficient_exploratory'
        else:
            continue
        for feature, score in zip(FEATURES,scores):
            rows.append({'model':name,'feature':feature,'score':round(float(score),8),
                         'score_type':kind,'trained_through_target_year':2024,
                         'caution':'Correlated features/small samples; not a causal explanation'})
    pd.DataFrame(rows).to_csv(REPORTS/'26_s2_v2_feature_diagnostics.csv',index=False)


def rolling_origin_backtests(df):
    """Supplementary 2022/2023 one-year-ahead tests; no test-year fitting."""
    rows = []
    for target_year in (2022, 2023):
        earlier = df.loc[(df['target_year'] < target_year)
                         & (df['high_coverage_for_evaluation'] == 1)].copy()
        held = df.loc[df['target_year'] == target_year].copy()
        if len(earlier) < 20 or held.empty:
            raise ValueError(f'Insufficient rolling-origin data for {target_year}')
        folds = make_models()
        outputs = predictions(folds, earlier, held)
        rows.extend(metrics_table(f'backtest_{target_year}', held, outputs))
    return rows


def main():
    df=load_and_verify()
    train=df.loc[(df['split']=='train') & (df['high_coverage_for_evaluation']==1)].copy()
    val=df.loc[df['split']=='validation'].copy()
    test=df.loc[df['split']=='test'].copy()
    print('STEP 26: Temporal supervised regression, leakage controls active')
    print(f'Train high-quality={len(train)} | Validation all={len(val)}, high-quality=13 | Test all={len(test)}, high-quality=17')
    print('Input features:', ', '.join(FEATURES))
    # Validation stage: train on 2019-2023 high-Q observations only.
    val_models=make_models()
    val_preds=predictions(val_models,train,val)
    val_metrics=metrics_table('validation',val,val_preds)
    ranking=sorted((r for r in val_metrics if r['evaluation_scope']=='high_coverage'),
                   key=lambda r:(r['mae_ha'],r['rmse_ha'],MODEL_ORDER.index(r['model'])))
    winner=ranking[0]['model']
    # Test stage: models refitted using OBSERVED 2024 high-Q transitions.
    # 2025 observed labels and QA are excluded from ALL fitting/selection.
    refit=pd.concat([train,val.loc[val['high_coverage_for_evaluation']==1]],ignore_index=True)
    final_models=make_models()
    test_preds=predictions(final_models,refit,test)
    test_metrics=metrics_table('test',test,test_preds)
    all_metrics=val_metrics+test_metrics
    backtest_metrics=rolling_origin_backtests(df)
    val_preds_rows=pred_rows('validation',val,val_preds,2023)
    test_preds_rows=pred_rows('test',test,test_preds,2024)

    for folder in (REPORTS,FIGURES,MODELS):
        folder.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(all_metrics).to_csv(REPORTS/'26_s2_v2_model_metrics.csv',index=False)
    pd.DataFrame(backtest_metrics).to_csv(REPORTS/'26_s2_v2_rolling_origin_backtests.csv',index=False)
    pd.DataFrame(val_preds_rows+test_preds_rows).to_csv(REPORTS/'26_s2_v2_holdout_predictions.csv',index=False)
    for name,model in final_models.items():
        joblib.dump(model,MODELS/f'26_s2_v2_{MODEL_FILE_STEMS[name]}.joblib')
    save_feature_summary(final_models,refit)
    figure=chart(all_metrics,winner)
    manifest={
        'status':'EXPLORATORY_SATELLITE_WATER_CLASSIFICATION_FORECAST_NOT_EROSION',
        'dataset':str(SOURCE.relative_to(ROOT)).replace('\\','/'),
        'target':TARGET,'target_units':'hectares per ~500m reach in 150m centreline proximity window',
        'feature_columns':FEATURES, 'strict_predictor_policy':'source-year x_ fields only; no observed_gain/loss, target, future QA, or target-date values',
        'training_for_2024_validation':{'target_years':[2019,2020,2021,2022,2023],'high_coverage_rows':len(train)},
        'training_for_2025_test':{'target_years':[2019,2020,2021,2022,2023,2024],'high_coverage_rows':len(refit)},
        'validation':{'target_year':2024,'all_rows':28,'high_coverage_rows':13},
        'test':{'target_year':2025,'all_rows':28,'high_coverage_rows':17},
        'selected_on_2024_validation_only':winner,
        'selection_metric':'MAE_ha on 13 high-coverage 2024 validation samples, then RMSE, then fixed model order',
        'model_hyperparameters':'fixed before looking at validation/test; see Step26 code',
        'refit_for_2025_test':True,
        'supplementary_backtests':'Walk-forward predictions for target years 2022 and 2023; trained on strictly earlier observed transitions',
        'selected_baseline_rule':('zero_change' if winner=='Persistence' else 'train_mean_delta' if winner=='Training mean' else None),
        'baseline_train_mean_delta_ha_2025':round(float(refit[TARGET].mean()),8),
        'selected_model_file':(f'models/26_s2_v2_{MODEL_FILE_STEMS[winner]}.joblib' if winner in MODEL_FILE_STEMS else None),
        'trained_ml_artifact_files':[f'models/26_s2_v2_{MODEL_FILE_STEMS[name]}.joblib' for name in MODEL_FILE_STEMS],
        'limits':[
          'Only 8 annual observations; validation 13 and test 17 high-coverage reaches',
          'Adjacent reaches and repeated years are not independent',
          'Quality screening uses target-date QA only to filter training/scoring; not available for a real future forecast',
          'Acquisition timing and water level vary; spectral classification differs from true bankline morphology',
          'Model performance in hectares is error against satellite-derived proxy, not measurement accuracy',
          'No synthetic target labels, no surveyed erosion labels, no test-informed model selection',
          'Imputations not needed; original Step25 predictors complete',
          'Tree impurity feature importances are descriptive, not causal',
        ],
    }
    (REPORTS/'26_s2_v2_model_selection.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('\n2024 HIGH-COVERAGE VALIDATION (selection):')
    for item in sorted([r for r in val_metrics if r['evaluation_scope']=='high_coverage'],key=lambda r:r['mae_ha']):
        print(f'  {item["model"]:<17} n={item["n"]:2} MAE={item["mae_ha"]:.4f} ha RMSE={item["rmse_ha"]:.4f} ha')
    print('SELECTED:',winner,'(2024 validation MAE only)')
    print('RETROSPECTIVE HIGH-QUALITY BACKTEST for selected model:')
    for r in backtest_metrics:
        if r['model']==winner and r['evaluation_scope']=='high_coverage':
            print(f'  {r["split"]}: n={r["n"]} MAE={r["mae_ha"]:.4f} ha')
    print('\n2025 HIGH-COVERAGE TEST (unseen during selection):')
    for item in sorted([r for r in test_metrics if r['evaluation_scope']=='high_coverage'],key=lambda r:r['mae_ha']):
        print(f'  {item["model"]:<17} n={item["n"]:2} MAE={item["mae_ha"]:.4f} ha RMSE={item["rmse_ha"]:.4f} ha')
    print('\nArtifacts:',REPORTS/'26_s2_v2_model_metrics.csv')
    print('Manifest:',REPORTS/'26_s2_v2_model_selection.json')
    print('Figure:',figure)
    print('IMPORTANT: Evaluation only vs water-classification proxy; not validated erosion.')
    print('STEP 26 COMPLETE')

if __name__=='__main__':
    main()
