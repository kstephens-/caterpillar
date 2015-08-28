import pandas as pd

import utils
from constants import yes_no_null


def hfl_components(bill_components, hfl):

    hfl_comps = pd.merge(
        bill_components[['tube_assembly_id', 'component']], hfl,
        left_on='component', right_on='component_id'
    )
    grouped_hfl_comps = hfl_comps \
        .groupby(hfl_comps.tube_assembly_id).sum()
    return grouped_hfl_comps


def corresponding_shell(hfl):

    shell_bin = pd.get_dummies(hfl['corresponding_shell'])
    hfl_shell = pd.concat([hfl, shell_bin], axis=1)
    hfl_shell = hfl_shell.drop(['corresponding_shell'], axis=1)
    return hfl_shell


def coupling_class(hfl):

    coupling_bin = pd.get_dummies(hfl['coupling_class'])
    hfl_coupling = pd.concat([hfl, coupling_bin], axis=1)
    hfl_coupling = hfl_coupling.drop(['coupling_class'], axis=1)
    return hfl_coupling


def material(hfl):

    mat_bin = pd.get_dummies(hfl['material'])
    hfl_mat = pd.concat([hfl, mat_bin], axis=1)
    hfl_mat = hfl_mat.drop(['material'], axis=1)
    return hfl_mat


def hfl(df, bill_components, hfl):

    # hfl.loc[:, 'plating_hfl'] = hfl['plating'] \
    #     .map(yes_no_null)
    # hfl.loc[:, 'orientation_hfl'] = hfl['orientation'] \
    #     .map(yes_no_null)

    hfl = corresponding_shell(hfl)
    #hfl = coupling_class(hfl)
    #hfl = material(hfl)

    hfl = hfl.drop(['plating', 'orientation', 'material',
                    'coupling_class', 'hose_diameter'], axis=1)

    hfl_comps = hfl_components(bill_components, hfl)
    df = pd.merge(df, hfl_comps, left_on='tube_assembly_id',
                  right_index=True, how='left')
    return df

