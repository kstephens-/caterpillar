import pandas as pd
import numpy as np
from sklearn import linear_model

import utils
from constants import yes_no_null


def threaded_components(bill_components, threaded):

    thread_comps = pd.merge(
        bill_components[['tube_assembly_id', 'component']], threaded,
        left_on='component', right_on='component_id'
    )
    grouped_threaded_comps = thread_comps \
        .groupby(thread_comps.tube_assembly_id).sum()
    return grouped_threaded_comps


def overall_length(threaded):

    d = threaded[['overall_length', 'weight']]
    msk = pd.notnull(d['overall_length'])
    train = d[msk]
    test = d[~msk]

    train.loc[:, 'weight'] = train['weight'] \
        .apply(lambda x: x**(1/3))
    test.loc[:, 'weight'] = test['weight'] \
        .apply(lambda x: x**(1/3))

    labels = np.log(train.overall_length.values)
    idx = test.index

    train = train.drop(['overall_length'], axis=1)
    test = test.drop(['overall_length'], axis=1)

    ols = linear_model.Ridge(alpha=0.1, normalize=True)
    ols.fit(train, labels)

    print('ols r2', ols.score(train, labels))
    predictions = ols.predict(test)

    threaded.loc[idx, 'overall_length'] = np.exp(predictions)
    return threaded


def threaded_impute(threaded):

    threaded.loc[:, 'weight'] = threaded['weight'] \
        .apply(lambda x: 0.2566 if pd.isnull(x) else x)

    #threaded = overall_length(threaded)

    return threaded


def threaded(df, bill_components, threaded):

    # threaded.loc[:, 'unique_feature_threaded'] = threaded['unique_feature'] \
    #     .map(yes_no_null)
    threaded.loc[:, 'orientation_threaded'] = threaded['orientation'] \
        .map(yes_no_null)

    threaded = threaded_impute(threaded)

    threaded = threaded.drop(['unique_feature', 'orientation',
                              'connection_type_id_1',
                              'connection_type_id_2',
                              'connection_type_id_3',
                              'connection_type_id_4',
                              'end_form_id_1', 'end_form_id_2',
                              'end_form_id_3', 'end_form_id_4',
                              'nominal_size_4'], axis=1)
    threaded = utils.rename_comp_columns(threaded, 'threaded')

    thread_comps = threaded_components(bill_components, threaded)
    df = pd.merge(df, thread_comps, left_on='tube_assembly_id',
                  right_index=True, how='left')

    return df
