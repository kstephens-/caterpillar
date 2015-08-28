import pandas as pd

from constants import yes_no_null


def float_components(bill_components, float_):

    float_comps = pd.merge(
        bill_components[['tube_assembly_id', 'component']], float_,
        left_on='component', right_on='component_id'
    )
    grouped_float_comps = float_comps \
        .groupby(float_comps.tube_assembly_id).sum()
    return grouped_float_comps


def float_(df, bill_components, float_):

    float_.loc[:, 'orientation_float'] = float_['orientation'] \
        .map(yes_no_null)

    float_ = float_[float_['weight'] <= 2.5]

    float_ = float_.drop(['component_type_id', 'orientation'], axis=1)

    float_comps = float_components(bill_components, float_)
    df = pd.merge(df, float_comps, left_on='tube_assembly_id',
                  right_index=True, how='left')
    return df
