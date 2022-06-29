import pathlib
import json
import pandas as pd
import seaborn as sns
import os
sns.set_context("poster")

majority_model_dirs = [
    'runs/DET_TRAIN_20220605-133744_train010_test050_ep70_bs032_lr1.00E-02_majority/output_best_rocauc_weighted',
    'runs/DET_TRAIN_20220605-155349_train020_test050_ep70_bs032_lr1.00E-02_majority/output_best_rocauc_weighted',
    'runs/DET_TRAIN_20220605-133941_train030_test050_ep70_bs032_lr1.00E-02_majority/output_best_rocauc_weighted',
    'runs/DET_TRAIN_20220605-134044_train040_test050_ep70_bs032_lr1.00E-02_majority/output_best_rocauc_weighted',
    'runs/DET_TRAIN_20220605-134059_train050_test050_ep70_bs032_lr1.00E-02_majority/output_best_rocauc_weighted',
]

ensemble_model_dir = 'runs/DET_TRAIN_20220607-160655_train050_test050_ep30_bs032_lr1.00E-02_majority_ENSEMBLE/output_best_rocauc_weighted'


def plot_metric(model_dirs, metric_filename, metric_name, ens_model_dir=None):
    step = 10
    res = []

    file_path = os.path.join(os.path.dirname(model_dirs[-1]), 'val_test_distribution.csv')
    # print(file_path)

    df_samples = pd.read_csv(file_path).set_index('biomarker')
    # print(df.head())
    # print(df.loc['Cannot Grade'])

    for model_dir in model_dirs:
        
        # kappa_path = pathlib.Path(model_dir).joinpath('output_kappa_score.json')
        kappa_path = pathlib.Path(model_dir).joinpath(metric_filename)
        with open(kappa_path) as f:
            kappa = json.load(f)
            kappa_test = kappa['test']
            kappa_val = kappa['validation']

        kappa_test['step'] = step
        kappa_val['step'] = step

        kappa_test['dataset'] = 'test'
        kappa_val['dataset'] = 'valid'

        res.append(kappa_test)
        res.append(kappa_val)
        step += 10

    df = pd.DataFrame(res).replace('nan', pd.NA)
    df = df.melt(('step', 'dataset'), var_name='biomarker', value_name=metric_name)
    df['biomarker'] = df['biomarker'].fillna(0.0) # in case of zero occurrences
    df[metric_name] = df[metric_name].astype(float)

    g = sns.relplot(
        data=df, kind="line",
        x="step", y=metric_name, col="biomarker",
        hue="biomarker", style="dataset",
        col_wrap=4, markers=True
        # facet_kws=dict(sharex=False),
    )

    if ens_model_dir is not None:
        kappa_ens_path = pathlib.Path(ens_model_dir).joinpath(metric_filename)

        with open(kappa_ens_path) as f:
            kappa_ens = json.load(f)

    for axes in g.axes:
        axes.tick_params(labelbottom=True)
        axes.set_xticks(range(10, len(model_dirs)*10 + 1, 10))
        bm = axes.get_title().split('=')[1].strip()
        axes.set_title(bm)
        axes.text(
            0.5, 0.02, 
            f'Test = {df_samples.loc[bm].test}\nValid = {df_samples.loc[bm].val}', 
            horizontalalignment='center', verticalalignment='bottom', transform=axes.transAxes
        )
        
        if ens_model_dir is not None:
            axes.scatter([50], [float(kappa_ens['test'][bm])], marker='o', color='black', label='ensemble_test')
            axes.scatter([50], [float(kappa_ens['validation'][bm])], marker='x', color='black', label='ensemble_val')

    g.savefig(f'{metric_name.lower()}_plot.png')

plot_metric(majority_model_dirs, 'output_auc_score.json', 'AUC', ensemble_model_dir)
plot_metric(majority_model_dirs, 'output_kappa_score.json', 'Kappa', ensemble_model_dir)