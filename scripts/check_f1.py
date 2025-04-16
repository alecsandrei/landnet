from __future__ import annotations

import pandas as pd

from landnet.enums import GeomorphometricalVariable
from landnet.config import MODELS_DIR, TRIAL_NAME


def enum_to_displayable(var: GeomorphometricalVariable):
    split = var.name.split('_')
    split[0] = split[0].title()
    for i in range(1, len(split[1:])+1):
        split[i] = split[i].lower()
    return ' '.join(split)

def compute_f_beta_legacy(series: pd.Series, beta=2):
    try:
        sensitivity = series.sensitivity
    except:
        sensitivity = series.val_sensitivity
    try:
        f1_score = series.f1_score
    except:
        f1_score = series.val_f1_score
    return compute_f_beta(sensitivity, f1_score, beta=beta)


def compute_f_beta(sensitivity, f1_score, beta):
    # Compute Precision (PPV) from Sensitivity and F1-score
    try:
        ppv = (f1_score * sensitivity) / (2 * sensitivity - f1_score)
    except ZeroDivisionError:
        return 0

    # Compute F-beta using the correct formula
    beta_sq = beta**2
    f_beta = (1 + beta_sq) * (sensitivity * ppv) / (beta_sq * sensitivity + ppv)

    return f_beta


# def f_beta(sensitivity, ppv, beta):
#     beta_sq = beta**2
#     return (1 + beta_sq) * (sensitivity * ppv) / (beta_sq * sensitivity + ppv)

METRICS = ['f1_score', 'sensitivity', 'specificity', 'accuracy', 'roc_auc']

if __name__ == '__main__':
    parent = MODELS_DIR / TRIAL_NAME
    metric = 'val_sensitivity'
    vals = []
    frames = []
    for csv in parent.glob('*.csv'):
        # print(csv)
        if csv.stem == 'results_summary':
            continue
        df = pd.read_csv(csv)
        # df['f_beta'] = df.apply(compute_f_beta_legacy, axis='columns')
        try:
            best = df.sort_values('f1_score', ascending=False).iloc[0]
        except:
            best = df.sort_values('val_f1_score', ascending=False).iloc[0]
        # max_f_beta = best.f_beta
        try:
            max_f1_score = best.f1_score
        except:
            max_f1_score = best.val_f1_score
        try:
            max_sensitivity = best.val_sensitivity
        except:
            max_sensitivity = best.sensitivity
        try:
            max_specificity = best.specificity
        except:
            max_specificity = best.val_specificity
        vals.append(
            (
                enum_to_displayable(GeomorphometricalVariable(csv.stem.split('_')[0])),
                # max_f_beta,
                max_f1_score,
                max_sensitivity,
                max_specificity,
            )
        )

        # max_f1_score = best.f1_score
    vals.sort(key=lambda x: x[2])
    for val in vals:
        print(val)
    df=pd.DataFrame(vals, columns=['variable', 'f1_score', 'sensitivity', 'specificity'])
    df.sort_values(by='sensitivity', ascending=False, inplace=True)

    df.to_csv(parent / 'results_summary.csv')
