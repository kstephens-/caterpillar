import pandas as pd

import utils
from constants import yes_no_null


def class_code(elbow):

    class_code_distribution = elbow['mj_class_code'].value_counts()
    elbow.loc[:, 'mj_class_code'] = elbow['mj_class_code'] \
        .apply(utils.rare_category, args=(class_code_distribution, ),
               cutoff=1, value='RareClassCode')

    class_code_bin = pd.get_dummies(elbow['mj_class_code'])
    class_code_dummy = pd.concat([elbow, class_code_bin], axis=1)
    class_code_dummy = class_code_dummy.drop(['mj_class_code'], axis=1)

    return class_code_dummy


def plug_class(elbow):

    plug_class_distribution = elbow['mj_plug_class_code'].value_counts()
    elbow.loc[:, 'mj_plug_class_code'] = elbow['mj_plug_class_code'] \
        .apply(utils.rare_category, args=(plug_class_distribution, ),
               cutoff=1, value='RarePlugClass')

    plug_class_bin = pd.get_dummies(elbow['mj_plug_class_code'])
    plug_class_dummy = pd.concat([elbow, plug_class_bin], axis=1)
    plug_class_dummy = plug_class_dummy.drop(['mj_plug_class_code'], axis=1)

    return plug_class_dummy


def elbow_impute(elbow):

    #elbow = elbow[(elbow['extension_length'] < 121.8) & (elbow['extension_length'] > 19.488)]
    #elbow = elbow[elbow['overall_length'] < 170]

    elbow.loc[:, 'bolt_pattern_long'] = elbow['bolt_pattern_long'] \
        .apply(lambda x: 63.62 if pd.isnull(x) else x)
    elbow.loc[:, 'bolt_pattern_wide'] = elbow['bolt_pattern_wide'] \
        .apply(lambda x: 33.24 if pd.isnull(x) else x)
    elbow.loc[:, 'extension_length'] = elbow['extension_length'] \
        .apply(lambda x: 48.72 if pd.isnull(x) else x)
    elbow.loc[:, 'overall_length'] = elbow['overall_length'] \
        .apply(lambda x: 83.25 if pd.isnull(x) else x)
    elbow.loc[:, 'thickness'] = elbow['thickness'] \
        .apply(lambda x: 46.55 if pd.isnull(x) else x)
    elbow.loc[:, 'drop_length'] = elbow['drop_length'] \
        .apply(lambda x: 26.92 if pd.isnull(x) else x)

    return elbow


def elbow_components(bill_components, elbow):

    elbow_comps = pd.merge(
        bill_components[['tube_assembly_id', 'component']], elbow,
        left_on='component', right_on='component_id')
    elbow_comps = elbow_comps.drop(['component', 'component_id'], axis=1)
    grouped_elbow_comps = elbow_comps \
        .groupby(elbow_comps.tube_assembly_id).sum()
    return grouped_elbow_comps


def elbow(df, bill_components, elbow):

    elbow.loc[:, 'groove_elbow'] = elbow['groove'].map(yes_no_null)
    #elbow.loc[:, 'unique_feature_elbow'] = elbow['unique_feature'].map(yes_no_null)
    #elbow.loc[:, 'orientation_elbow'] = elbow['orientation'].map(yes_no_null)

    elbow = elbow_impute(elbow)
    #elbow = class_code(elbow)
    #elbow = plug_class(elbow)

    elbow = elbow.drop(['groove', 'unique_feature',
                        'orientation', 'component_type_id',
                        'elbow_angle', 'overall_length',
                        'plug_diameter', 'mj_class_code',
                        'extension_length', 'mj_plug_class_code'], axis=1)

    elbow_comps = elbow_components(bill_components, elbow)
    df = pd.merge(df, elbow_comps, left_on='tube_assembly_id',
                  right_index=True, how='left')
    return df
