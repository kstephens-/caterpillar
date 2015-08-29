import pandas as pd
from sklearn import linear_model

from constants import yes_no_null


def nut_components(bill_components, nut):

    nut_comps = pd.merge(
        bill_components[['tube_assembly_id', 'component']], nut,
        left_on='component', right_on='component_id'
    )
    grouped_nut_components = nut_comps \
        .groupby(nut_comps.tube_assembly_id).sum()
    return grouped_nut_components


def metric_convert(nut):

    # thread pitch
    nut.loc[nut['thread_size'] == 'M6', 'thread_pitch'] = 25.4
    nut.loc[nut['thread_size'] == 'M8', 'thread_pitch'] = 20.32
    nut.loc[nut['thread_size'] == 'M10', 'thread_pitch'] = 16.93
    nut.loc[nut['thread_size'] == 'M12', 'thread_pitch'] = 14.51

    # thread size
    nut.loc[nut['thread_size'] == 'M6', 'thread_size'] = 1/4
    nut.loc[nut['thread_size'] == 'M8', 'thread_size'] = 5/16
    nut.loc[nut['thread_size'] == 'M10', 'thread_size'] = 3/8
    nut.loc[nut['thread_size'] == 'M12', 'thread_size'] = 1/2
    return nut


def hex_nut_size(nut):

    d = nut[['hex_nut_size', 'length', 'thread_pitch', 'thread_size', 'weight']]
    msk = pd.notnull(d['hex_nut_size'])
    train = d[msk]
    test = d[~msk]
    if test.empty:
        return nut

    labels = train.hex_nut_size.values
    idx = test.index

    train = train.drop(['hex_nut_size'], axis=1)
    test = test.drop(['hex_nut_size'], axis=1)

    train = train.astype(float)
    test = test.astype(float)

    ols = linear_model.Ridge(alpha=0.2, normalize=False)
    ols.fit(train, labels)

    # print('hex nut size r2', ols.score(train, labels))
    predictions = ols.predict(test)

    nut.loc[idx, 'hex_nut_size'] = predictions
    return nut


def diameter(nut):

    d = nut[['hex_nut_size', 'length', 'thread_pitch', 'thread_size', 'weight', 'diameter']]
    msk = pd.notnull(d['diameter'])
    train = d[msk]
    test = d[~msk]
    if test.empty:
        return nut

    labels = train.diameter.values
    idx = test.index

    train = train.drop(['diameter'], axis=1)
    test = test.drop(['diameter'], axis=1)

    train = train.astype(float)
    test = test.astype(float)

    ols = linear_model.Ridge(alpha=0.2, normalize=False)
    #ols = linear_model.LinearRegression(normalize=False)
    ols.fit(train, labels)

    # print('ols r2', ols.score(train, labels))
    predictions = ols.predict(test)

    nut.loc[idx, 'diameter'] = predictions
    return nut


def seat_angle(nut):

    d = nut[['seat_angle', 'length', 'hex_nut_size', 'thread_pitch']]
    msk = pd.notnull(d['seat_angle'])
    train = d[msk]
    test = d[~msk]
    if test.empty:
        return nut

    labels =  train.seat_angle.values
    idx = test.index

    train = train.drop(['seat_angle'], axis=1)
    test = test.drop(['seat_angle'], axis=1)

    train = train.astype(float)
    test = test.astype(float)

    ols = linear_model.Ridge(alpha=0.3, normalize=False)
    ols.fit(train, labels)

    print('ols r2', ols.score(train, labels))
    predictions = ols.predict(test)

    nut.loc[idx, 'seat_angle'] = predictions
    return nut


def nut_impute(nut):

    nut = metric_convert(nut)

    # nut.loc[:, 'hex_nut_size'] = nut['hex_nut_size'] \
    #     .apply(lambda x: 29.81 if pd.isnull(x) else x)

    # nut.loc[:, 'length'] = nut['length'] \
    #     .apply(lambda x: 26.45 if x == 9999 else x)

    # nut.loc[:, 'diameter'] = nut['diameter'] \
    #     .apply(lambda x: 19.8 if pd.isnull(x) else x)

    nut.loc[:, 'weight'] = nut['weight'] \
        .apply(lambda x: 0.07709 if pd.isnull(x) else x)

    # nut.loc[:, 'seat_angle'] = nut['seat_angle'] \
    #     .apply(lambda x: 38.6 if pd.isnull(x) else x)

    nut = hex_nut_size(nut)
    nut = diameter(nut)
    #nut = seat_angle(nut)

    return nut


def nut(df, bill_components, nut):

    # nut.loc[:, 'orientation_nut'] = nut['orientation'] \
    #     .map(yes_no_null)
    # nut.loc[:, 'blind_hole_nut'] = nut['blind_hole'] \
    #     .map(yes_no_null)
    nut = nut_impute(nut)

    nut = nut.drop(['component_type_id', 'orientation',
                    'blind_hole', 'seat_angle'], axis=1)

    nut_comps = nut_components(bill_components, nut)
    df = pd.merge(df, nut_comps, left_on='tube_assembly_id',
                  right_index=True, how='left')
    return df
