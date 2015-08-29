import pandas as pd
from sklearn import linear_model

from constants import yes_no_null


def straight_components(bill_components, straight):

    straight_comps = pd.merge(
        bill_components[['tube_assembly_id', 'component']], straight,
        left_on='component', right_on='component_id'
    )
    grouped_straight_comps = straight_comps \
        .groupby(straight_comps.tube_assembly_id).sum()
    return grouped_straight_comps


def bolt_pattern_wide(straight):

    d = straight[['bolt_pattern_wide', 'weight', 'thickness']]
    msk = pd.notnull(d['bolt_pattern_wide'])
    train = d[msk]
    test = d[~msk]
    if test.empty:
        return straight

    labels = train.bolt_pattern_wide.values
    idx = test.index

    train = train.drop(['bolt_pattern_wide'], axis=1)
    test = test.drop(['bolt_pattern_wide'], axis=1)

    train = train.astype(float)
    test = test.astype(float)

    ols = linear_model.Ridge(alpha=0.2, normalize=True)
    ols.fit(train, labels)

    #print('ols r2', ols.score(train, labels))
    predictions = ols.predict(test)

    straight.loc[idx, 'bolt_pattern_wide'] = predictions
    return straight


def bolt_pattern_long(straight):

    d = straight[['bolt_pattern_long', 'weight']]
    msk = pd.notnull(d['bolt_pattern_long'])
    train = d[msk]
    test = d[~msk]
    if test.empty:
        return straight

    labels = train.bolt_pattern_long.values
    idx = test.index

    train = train.drop(['bolt_pattern_long'], axis=1)
    test = test.drop(['bolt_pattern_long'], axis=1)

    ols = linear_model.Ridge(alpha=0.2, normalize=True)
    ols.fit(train, labels)

    #print('ols r2', ols.score(train, labels))
    predictions = ols.predict(test)

    straight.loc[idx, 'bolt_pattern_long'] = predictions
    return straight


def head_diameter(straight):

    d = straight[['head_diameter', 'weight', 'bolt_pattern_wide']]
    msk = pd.notnull(d['head_diameter'])
    train = d[msk]
    test = d[~msk]
    if test.empty:
        return straight

    labels = train.head_diameter.values
    idx = test.index

    train = train.drop(['head_diameter'], axis=1)
    test = test.drop(['head_diameter'], axis=1)

    ols = linear_model.Ridge(alpha=0.2, normalize=True)
    ols.fit(train, labels)

    #print('ols r2', ols.score(train, labels))
    predictions = ols.predict(test)

    straight.loc[idx, 'head_diameter'] = predictions
    return straight


def overall_length(straight):

    d = straight[['overall_length', 'weight', 'bolt_pattern_wide', 'thickness']]
    msk = pd.notnull(d['overall_length'])
    train = d[msk]
    test = d[~msk]

    if test.empty:
        return straight

    labels = train.overall_length.values
    idx = test.index

    train = train.drop(['overall_length'], axis=1)
    test = test.drop(['overall_length'], axis=1)

    ols = linear_model.LinearRegression(normalize=False)
    ols.fit(train, labels)

    #print('ols r2', ols.score(train, labels))
    predictions = ols.predict(test)

    straight.loc[idx, 'overall_length'] = predictions
    return straight


def straight_impute(straight):

    #straight = straight[straight['weight'] < 5.0]

    straight.loc[:, 'weight'] = straight['weight'] \
        .apply(lambda x: 0.7884 if pd.isnull(x) else x)

    straight = bolt_pattern_wide(straight)
    straight = bolt_pattern_long(straight)
    straight = head_diameter(straight)
    straight = overall_length(straight)

    return straight


def straight(df, bill_components, straight):

    # straight.loc[:, 'groove_straight'] = straight['groove'] \
    #     .map(yes_no_null)
    # straight.loc[:, 'unique_feature_straight'] = straight['unique_feature'] \
    #     .map(yes_no_null)
    straight.loc[:, 'orientation_straight'] = straight['orientation'] \
        .map(yes_no_null)

    straight = straight_impute(straight)
    straight = straight.drop(['groove', 'unique_feature',
                             'orientation', 'component_type_id',
                             'mj_class_code'], axis=1)

    straight_comps = straight_components(bill_components, straight)
    df = pd.merge(df, straight_comps, left_on='tube_assembly_id',
                  right_index=True, how='left')
    return df
