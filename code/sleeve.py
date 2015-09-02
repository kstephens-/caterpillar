import pandas as pd

import utils
from constants import yes_no_null


def sleeve_components(bill_components, sleeve):

    sleeve_comps = pd.merge(
        bill_components[['tube_assembly_id', 'component']], sleeve,
        left_on='component', right_on='component_id'
    )
    grouped_sleeve_comps = sleeve_comps \
        .groupby(sleeve_comps.tube_assembly_id).sum()
    return grouped_sleeve_comps


# def clean_sleeve(df):

#     print()
#     print('df columns', df.columns)
#     print()
#     df = df.drop(['component', 'component_id', 'component_type_id',
#                   'connection_type_id', 'unique_feature', 'plating',
#                   'orientation'], axis=1)
#     return df


def sleeve(df, bill_components, sleeve):

    #sleeve.loc[:, 'unique_feature_sleeve'] = \
    #    sleeve['unique_feature'].map(yes_no_null)
    #sleeve.loc[:, 'plating_sleeve'] = \
    #    sleeve['plating'].map(yes_no_null)
    #sleeve.loc[:, 'orientation_sleeve'] = \
    #    sleeve['orientation'].map(yes_no_null)
    sleeve.loc[:, 'length'] = sleeve['length'] \
        .apply(lambda x: 14.4 if x == 9999 else x)

    sleeve = sleeve.drop(['orientation', 'plating', 'unique_feature',
                          'intended_nut_thread', 'weight'], axis=1)
    sleeve = utils.rename_comp_columns(sleeve, 'sleeve')

    sleeve_comps = sleeve_components(bill_components, sleeve)
    df = pd.merge(df, sleeve_comps, left_on='tube_assembly_id',
                  right_index=True, how='left')
    #df = clean_sleeve(df)
    return df

