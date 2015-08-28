import pandas as pd
import numpy as np
from sklearn import linear_model

import utils
from constants import yes_no_null


def end_forms(adaptor):

    adaptor_end_forms = pd.concat(
        adaptor[['component_id', 'end_form_id_{}'.format(i)]]
        .rename(columns={'end_form_id_{}'.format(i): 'end_form_id'})
        for i in range(1, 3)
    )

    end_form_distribution = adaptor_end_forms['end_form_id'].value_counts()
    adaptor_end_forms.loc[:, 'end_form_id'] = adaptor_end_forms['end_form_id'] \
        .apply(utils.rare_category, args=(end_form_distribution, ),
               cutoff=20, value='RareEndForm')

    end_form_bin = pd.get_dummies(adaptor_end_forms['end_form_id'])
    end_form_dummy = pd.concat([adaptor_end_forms, end_form_bin], axis=1)
    #end_form_dummy = end_form_dummy.drop(['end_form_id_1', 'end_form_id_2'], axis=1)

    grouped_end_form = end_form_dummy \
        .groupby(end_form_dummy.component_id).sum()

    adaptor = pd.merge(adaptor, grouped_end_form,
                       left_on='component_id', right_index=True)
    adaptor = adaptor.drop(['RareEndForm'], axis=1)
    return adaptor


def adaptor_components(bill_components, adaptor):

    adaptor_comps = pd.merge(
        bill_components[['tube_assembly_id', 'component']], adaptor,
        left_on='component', right_on='component_id'
    )
    grouped_adaptor_comps = adaptor_comps \
        .groupby(adaptor_comps.tube_assembly_id).sum()
    return grouped_adaptor_comps


def overall_length(adaptor):

    d = adaptor[['overall_length', 'weight']]
    msk = pd.notnull(d['overall_length'])
    train = d[msk]
    test = d[~msk]

    train.dropna(axis=0, how='any', inplace=True)

    labels = train.overall_length.values
    idx = test.index

    train = train.drop(['overall_length'], axis=1)
    test = test.drop(['overall_length'], axis=1)

    train = train.astype(float)
    test = test.astype(float)

    print('train shape', train.shape)
    print('test shape', test.shape)

    ols = linear_model.LinearRegression(normalize=True)
    ols.fit(train, labels)

    print('ols r2', ols.score(train, labels))
    predictions = ols.predict(test)

    adaptor.loc[idx, 'overall_length'] = predictions
    return adaptor


def hex_size(adaptor):

    d = adaptor[['hex_size', 'weight']]
    msk = pd.notnull(d['hex_size'])
    train = d[msk]
    test = d[~msk]

    train.dropna(axis=0, how='any', inplace=True)

    labels = train.hex_size.values
    idx = test.index

    train = train.drop(['hex_size'], axis=1).astype(float)
    test = test.drop(['hex_size'], axis=1).astype(float)

    print('train shape', train.shape)
    print('test shape', test.shape)

    ols = linear_model.Ridge(alpha=0.7, normalize=True, tol=0.00001)
    ols.fit(train, labels)

    print('ols r2', ols.score(train, labels))
    predictions = ols.predict(test)

    adaptor.loc[idx, 'hex_size'] = predictions
    return adaptor


def adaptor_impute(adaptor):

    #adaptor.loc[pd.isnull(adaptor['overall_length']), 'overall_length'] = 30
    # adaptor.loc[:, 'overall_length'] = adaptor['overall_length'] \
    #     .apply(lambda x: 31 if pd.isnull(x) else x)
    #adaptor = hex_size(adaptor)
    #adaptor = overall_length(adaptor)

    adaptor.loc[:, 'thread_size_1'] = adaptor['thread_size_1'] \
        .apply(lambda x: 0.68 if pd.isnull(x) else x)

    adaptor.loc[:, 'thread_pitch_1'] = adaptor['thread_pitch_1'] \
        .apply(lambda x: 16.0 if pd.isnull(x) else x)

    adaptor.loc[:, 'nominal_size_1'] = adaptor['nominal_size_1'] \
        .apply(lambda x: 8.8 if pd.isnull(x) else x)

    # adaptor.loc[:, 'thread_size_2'] = adaptor['thread_size_2'] \
    #    .apply(lambda x: 0.78 if pd.isnull(x) else x)

    # adaptor.loc[:, 'thread_pitch_2'] = adaptor['thread_pitch_2'] \
    #     .apply(lambda x: 19.3 if pd.isnull(x) else x)

    # adaptor.loc[:, 'nominal_size_2'] = adaptor['nominal_size_2'] \
    #     .apply(lambda x: 9.52 if pd.isnull(x) else x)

    # adaptor.loc[:, 'hex_size'] = adaptor['hex_size'] \
    #     .apply(lambda x: 8 if pd.isnull(x) else x)

    # adaptor.loc[:, 'weight'] = adaptor['weight'] \
    #     .apply(lambda x: 0.2 if pd.isnull(x) else x)
    # adaptor = overall_length(adaptor)

    return adaptor


def adaptor(df, bill_components, adaptor):

    adaptor.loc[:, 'unique_feature_adaptor'] = \
        adaptor['unique_feature'].map(yes_no_null)
    adaptor.loc[:, 'orientation_adaptor'] = \
        adaptor['orientation'].map(yes_no_null)

    adaptor = adaptor_impute(adaptor)
    adaptor = end_forms(adaptor)

    adaptor = adaptor.drop(['component_type_id',
                            'connection_type_id_1',
                            'connection_type_id_2',
                            'unique_feature', 'orientation',
                            'end_form_id_1', 'end_form_id_2',
                            'adaptor_angle'], axis=1)

    adaptor_comps = adaptor_components(bill_components, adaptor)
    df = pd.merge(df, adaptor_comps, left_on='tube_assembly_id',
                  right_index=True, how='left')

    return df
