import pandas as pd
import numpy as np

import utils
import components as comps


bill = pd.read_csv('../competition_data/bill_of_materials.csv')


def bill_components():

    components = pd.concat(
        bill[['tube_assembly_id',
              'component_id_{}'.format(i),
              'quantity_{}'.format(i)]]
            .rename(columns={'component_id_{}'.format(i): 'component',
                             'quantity_{}'.format(i): 'quantity'})
        for i in range(1, 9)
    )
    component_distribution = components['component'].value_counts()
    components.loc[:, 'filtered_component'] = \
        components['component'].apply(
            utils.rare_category, args=(component_distribution, ),
            cutoff=3000, value='RareComponent'
        )
    return components


def component_quantity(df, components):

    component_table = pd.pivot_table(
        components, values='quantity',
        index='tube_assembly_id', columns='filtered_component',
        aggfunc=np.sum, fill_value=0
    )
    df = pd.merge(df, component_table, left_on='tube_assembly_id',
                  right_index=True, how='left')
    return df


def bill_of_material(df):

    components = bill_components()
    df = component_quantity(df, components)
    df = comps.base_name(df, components)


    return df
