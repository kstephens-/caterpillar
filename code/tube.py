import pandas as pd
import math
from sklearn import linear_model

import utils
import end_form
from constants import yes_no_null

def material(df, material_distribution):

    df.loc[:, 'material_id'] = df['material_id'] \
        .apply(utils.rare_category, args=(material_distribution, ),
               cutoff=50, value='RareMaterial')

    material_bin = pd.get_dummies(df['material_id'])
    df = pd.concat([df, material_bin], axis=1)
    df = df.drop(['material_id'], axis=1)

    return df


def has_bends(df):

    df.loc[:, 'has_bends'] = (df['num_bends'] > 0).astype(int)
    return df


def bend_radius(df):

    #df = df[df['bend_radius'] < 9000]
    df.loc[:, 'bend_radius'] = df['bend_radius'].apply(
       lambda x: 31.75 if x == 9999 else x
    )
    return df


# def bend_radius(df):

#     d = df[['bend_radius', 'diameter', 'wall', 'num_bends']]
#     d = d[d['bend_radius'] != 0]
#     msk = d['bend_radius'] != 9999
#     train = d[msk]
#     test = d[~msk]

#     if test.empty:
#         return df

#     labels = train.bend_radius.values
#     idx = test.index

#     train.drop(['bend_radius'], axis=1, inplace=True)
#     test.drop(['bend_radius'], axis=1, inplace=True)

#     ols = linear_model.Ridge(alpha=0.2, normalize=False)
#     ols.fit(train, labels)

#     print()
#     print('ols r2', ols.score(train, labels))
#     print()
#     predictions = ols.predict(test)

#     df.loc[idx, 'bend_radius'] = predictions
#     return df


def end_bool(df):

    df.loc[:, 'end_a_1x'] = df['end_a_1x'].map(yes_no_null)
    df.loc[:, 'end_a_2x'] = df['end_a_2x'].map(yes_no_null)

    df.loc[:, 'end_x_1x'] = df['end_x_1x'].map(yes_no_null)
    df.loc[:, 'end_x_2x'] = df['end_x_2x'].map(yes_no_null)
    return df


def volume(df):

    df.loc[:, 'volume'] = \
        ((math.pi * (df['diameter'] / 2)**2) -
            math.pi * ((df['diameter'] - df['wall']) / 2)**2) * df['length']
    return df


def ends(df):

    end_a = df[['tube_assembly_id', 'end_a']] \
        .rename(columns={'end_a': 'end'})
    end_x = df[['tube_assembly_id', 'end_x']] \
        .rename(columns={'end_x': 'end'})
    ends = pd.concat([end_a, end_x], axis=0)
    ends = ends.reset_index()
    ends = ends.drop(['index'], axis=1)

    end_distribution = ends['end'].value_counts()
    ends.loc[:, 'end'] = ends['end'] \
        .apply(utils.rare_category, args=(end_distribution, ),
               cutoff=20, value='RareEnd')

    end_bin = pd.get_dummies(ends['end'])
    ends = pd.concat([ends, end_bin], axis=1)
    ends = ends.groupby(ends.tube_assembly_id).sum()

    df = pd.merge(df, ends, left_on='tube_assembly_id', right_index=True)
    df = df.drop(['end_a', 'end_x'], axis=1)
    return df


def tube(df):

    df = df[df['length'] < 9999]

    material_distribution = df['material_id'].value_counts()
    df = material(df, material_distribution)

    #df = has_bends(df)
    df = bend_radius(df)
    df = end_bool(df)
    df = volume(df)
    df = end_form.end_forms(df)
    df = ends(df)

    #df.loc[:, 'total_bend_radius'] = df['num_bends'] * df['bend_radius']

    # possibly drop these
    df = df.drop(['num_bracket'], axis=1)
    return df

