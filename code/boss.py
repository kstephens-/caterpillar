import pandas as pd
from sklearn import linear_model

import utils
from constants import yes_no_null


def boss_type(boss):

    bt_dist = boss['type'].value_counts()
    boss.loc[:, 'type'] = boss['type'] \
        .apply(utils.rare_category, args=(bt_dist, ),
               cutoff=20, value='RareBossType')

    boss_type_bin = pd.get_dummies(boss['type'])
    boss_type_dummy = pd.concat([boss, boss_type_bin], axis=1)
    boss_type_dummy = boss_type_dummy.drop(['type'], axis=1)

    return boss_type_dummy


def outside_shape(boss):

    shape_dist = boss['outside_shape'].value_counts()
    boss.loc[:, 'outside_shape'] = boss['outside_shape'] \
        .apply(utils.rare_category, args=(shape_dist, ),
               cutoff=1, value='RareOutsideShape')

    boss_shape_bin = pd.get_dummies(boss['outside_shape'])
    boss_shape_dummy = pd.concat([boss, boss_shape_bin], axis=1)
    boss_shape_dummy = boss_shape_dummy.drop(['outside_shape'], axis=1)

    return boss_shape_dummy


def base_type(boss):

    base_type_dist = boss['base_type'].value_counts()
    boss.loc[:, 'base_type'] = boss['base_type'] \
        .apply(utils.rare_category, args=(base_type_dist, ),
               cutoff=1, value='RareBaseType')

    base_type_bin = pd.get_dummies(boss['base_type'])
    base_type_dummy = pd.concat([boss, base_type_bin], axis=1)
    base_type_dummy = base_type_dummy.drop(['base_type'], axis=1)

    return base_type_dummy


def boss_components(bill_components, boss):

    boss_comps = pd.merge(
        bill_components[['tube_assembly_id', 'component']], boss,
        left_on='component', right_on='component_id'
    )
    grouped_boss_comps = boss_comps \
        .groupby(boss_comps.tube_assembly_id).sum()
    return grouped_boss_comps


def bolt_pattern_long(boss):

    d = boss[['height_over_tube', 'weight', 'bolt_pattern_long']]
    msk = pd.notnull(d['bolt_pattern_long'])
    train = d[msk]
    test = d[~msk]
    if test.empty:
        return boss

    labels = train.bolt_pattern_long.values
    idx = test.index

    train = train.drop(['bolt_pattern_long'], axis=1)
    test = test.drop(['bolt_pattern_long'], axis=1)

    ols = linear_model.Ridge(alpha=0.4, normalize=True)
    # ols = linear_model.Lasso(alpha=1.0, normalize=True,
    #                          random_state=42, selection='random')
    ols.fit(train, labels)
    print('ols r2 in bolt pattern long', ols.score(train, labels))
    predictions = ols.predict(test)

    boss.loc[idx, 'bolt_pattern_long'] = predictions
    return boss


def shoulder_diameter(boss):

    d = boss[['height_over_tube', 'weight', 'shoulder_diameter']]
    msk = pd.notnull(d['shoulder_diameter'])
    train = d[msk]
    test = d[~msk]

    labels = train.shoulder_diameter.values
    idx = test.index

    train = train.drop(['shoulder_diameter'], axis=1)
    test = test.drop(['shoulder_diameter'], axis=1)

    ols = linear_model.Lasso(alpha=1.0, normalize=False,
                             random_state=42, selection='random')
    ols.fit(train, labels)
    print('lasso r2', ols.score(train, labels))
    predictions = ols.predict(test)

    boss.loc[idx, 'shoulder_diameter'] = predictions
    return boss


def boss_impute(boss):

    #boss = boss[boss['weight'] < 4]

    boss.loc[:, 'height_over_tube'] = boss['height_over_tube'] \
       .apply(lambda x: 18.8 if x == 9999 else x)
    # boss.loc[:, 'bolt_pattern_long'] = boss['bolt_pattern_long'] \
    #     .apply(lambda x: 83.77 if pd.isnull(x) else x)
    # boss.loc[:, 'bolt_pattern_wide'] = boss['bolt_pattern_wide'] \
    #     .apply(lambda x: 42.9 if pd.isnull(x) else x)
    # boss.loc[:, 'base_diameter'] = boss['base_diameter'] \
    #     .apply(lambda x: 29.88 if pd.isnull(x) else x)
    # boss.loc[:, 'shoulder_diameter'] = boss['shoulder_diameter'] \
    #     .apply(lambda x: 23.5 if pd.isnull(x) else x)
    boss.loc[:, 'weight'] = boss['weight'] \
        .apply(lambda x: 0.082 if pd.isnull(x) else x)

    boss = bolt_pattern_long(boss)
    #boss = shoulder_diameter(boss)

    return boss


def boss(df, bill_components, boss):

    # boss.loc[:, 'groove_boss'] = \
    #     boss['groove'].map(yes_no_null)
    # boss.loc[:, 'unique_feature_boss'] = \
    #     boss['unique_feature'].map(yes_no_null)
    boss.loc[:, 'orientation_boss'] = \
        boss['orientation'].map(yes_no_null)

    boss = boss_impute(boss)
    #boss = boss_type(boss)
    #boss = outside_shape(boss)
    #boss = base_type(boss)
    #boss = boss[boss['weight'] <= 2.2]
    #boss = boss[boss['height_over_tube'] <= 50.9]

    boss = boss.drop(['groove', 'unique_feature',
                      'orientation', 'component_type_id',
                      'connection_type_id',  'base_diameter',
                      'outside_shape', 'type','shoulder_diameter',
                      'base_type', 'bolt_pattern_wide'], axis=1)

    boss_comps = boss_components(bill_components, boss)
    df = pd.merge(df, boss_comps, left_on='tube_assembly_id',
                  right_index=True, how='left')
    return df
