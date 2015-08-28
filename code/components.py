import pandas as pd
import numpy as np
import re

import utils


components = pd.read_csv('../competition_data/components.csv')


comp_name_re = re.compile(r'^([^ -]+)')
def base_component_name(x):
    try:
        return comp_name_re.match(x).group(1)
    except TypeError:
        return np.nan


def base_name(df, bill_components):

    components.loc[:, 'base_name'] = \
        components['name'].apply(base_component_name)

    base_name_distribution = components['base_name'].value_counts()
    components.loc[:, 'base_name'] = components['base_name'] \
        .apply(utils.rare_category, args=(base_name_distribution, ),
               cutoff=1000, value='RareBaseName')

    component_bin = pd.get_dummies(components['base_name'])
    components_dummy = pd.concat([components, component_bin], axis=1)

    merged_components = pd.merge(bill_components,
                                 components_dummy,
                                 left_on='component',
                                 right_on='component_id',
                                 how='left')
    grouped_components = merged_components \
        .groupby(merged_components.tube_assembly_id).sum()

    df = pd.merge(df, grouped_components,
                  left_on='tube_assembly_id',
                  right_index=True, how='left')
    return df


def component_name(df, bill_components):

    name_distribution = components['name'].value_counts()
    components.loc[:, 'name'] = components['name'] \
        .apply(utils.rare_category, args=(name_distribution, ),
               cutoff=200, value='RareComponentName')

    component_bin = pd.get_dummies(components['name'])
    components_dummy = pd.concat([components, component_bin], axis=1)

    merged_components = pd.merge(bill_components,
                                 components_dummy,
                                 left_on='component',
                                 right_on='component_id',
                                 how='left')
    grouped_components = merged_components \
        .groupby(merged_components.tube_assembly_id).sum()

    df = pd.merge(df, grouped_components,
                  left_on='tube_assembly_id',
                  right_index=True, how='left')
    return df
