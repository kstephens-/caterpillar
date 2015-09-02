import pandas as pd

import utils
from constants import yes_no_null


def class_code(tee):

    class_code_bin = pd.get_dummies(tee['mj_class_code'])
    class_code_dummy = pd.concat([tee, class_code_bin], axis=1)
    class_code_dummy = class_code_dummy.drop(['mj_class_code'], axis=1)

    return class_code_dummy


def plug_class(tee):

    plug_code_bin = pd.get_dummies(tee['mj_plug_class_code'])
    plug_code_dummy = pd.concat([tee, plug_code_bin], axis=1)
    plug_code_dummy = plug_code_dummy.drop(['mj_plug_class_code'], axis=1)

    return plug_code_dummy


def tee_components(bill_components, tee):

    tee_comps = pd.merge(
        bill_components[['tube_assembly_id', 'component']], tee,
        left_on='component', right_on='component_id'
    )
    grouped_tee_comps = tee_comps \
        .groupby(tee_comps.tube_assembly_id).sum()
    return grouped_tee_comps


def tee(df, bill_components, tee):

    tee.loc[:, 'groove_tee'] = \
        tee['groove'].map(yes_no_null)
    tee.loc[:, 'unique_feature_tee'] = \
        tee['unique_feature'].map(yes_no_null)
    # tee.loc[:, 'orientation_tee'] = \
    #     tee['orientation'].map(yes_no_null)

    tee = class_code(tee)
    #tee = plug_class(tee)

    tee = tee.drop(['groove', 'unique_feature',
                    'orientation', 'component_type_id',
                    'mj_plug_class_code'], axis=1)
    tee = utils.rename_comp_columns(tee, 'tee')

    tee_comps = tee_components(bill_components, tee)
    df = pd.merge(df, tee_comps, left_on='tube_assembly_id',
                  right_index=True, how='left')

    return df
