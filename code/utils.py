import numpy as np
import math


def rmsle(labels, predictions):
    try:
        n_preds, n_targets = predictions.shape
        p = predictions.sum(axis=1)
        a = labels.sum(axis=1)
    except ValueError:
        n_preds = len(predictions)
        n_targets = 1
        p = predictions
        a = labels

    sum_squared_error = np.sum((np.log(p+1) - np.log(a+1))**2)
    return np.sqrt(1/(n_preds*n_targets) * sum_squared_error)


def print_score(answers, predictions, model, train=False):

    print('{} score is', rmsle(answers, predictions))


def rare_category(x, category_distribution, cutoff=1, value='Rare'):
    # if x == '9999' or x == 'NONE':
    #     return np.nan
    try:
        if category_distribution[x] < cutoff:
            return value
    except (ValueError, KeyError):
        return np.nan
    else:
        return x


def make_inv(x):
    if x == 0:
        return 0
    return 1 / math.log(1 + x)


def rename_comp_columns(df, suffix, skip='component_id'):

    df.columns = ['_'.join([c, suffix])
                  if c != skip and not c.endswith(suffix) else c
                  for c in df.columns]
    return df
