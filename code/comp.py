import pandas as pd

import utils
import sleeve
import adaptor
import boss
import elbow
import float_
import hfl
import nut
import other
import straight


comp_sleeve = pd.read_csv('../competition_data/comp_sleeve.csv')
comp_adaptor = pd.read_csv('../competition_data/comp_adaptor.csv')
comp_boss = pd.read_csv('../competition_data/comp_boss.csv')
comp_elbow = pd.read_csv('../competition_data/comp_elbow.csv')
comp_float = pd.read_csv('../competition_data/comp_float.csv')
comp_hfl = pd.read_csv('../competition_data/comp_hfl.csv')
comp_nut = pd.read_csv('../competition_data/comp_nut.csv')
comp_other = pd.read_csv('../competition_data/comp_other.csv')
comp_straight = pd.read_csv('../competition_data/comp_straight.csv')
comp_tee = pd.read_csv('../competition_data/comp_tee.csv')
comp_threaded = pd.read_csv('../competition_data/comp_threaded.csv')


def component_type(df, bill_components):

    comp_files = []
    comp_files.append(comp_sleeve[['component_id', 'component_type_id']])
    comp_files.append(comp_adaptor[['component_id', 'component_type_id']])
    comp_files.append(comp_boss[['component_id', 'component_type_id']])
    comp_files.append(comp_elbow[['component_id', 'component_type_id']])
    comp_files.append(comp_float[['component_id', 'component_type_id']])
    comp_files.append(comp_hfl[['component_id', 'component_type_id']])
    comp_files.append(comp_nut[['component_id', 'component_type_id']])
    comp_files.append(comp_straight[['component_id', 'component_type_id']])
    comp_files.append(comp_tee[['component_id', 'component_type_id']])
    comp_files.append(comp_threaded[['component_id', 'component_type_id']])

    comps = pd.concat(comp_files)
    comps_distribution = comps['component_type_id'].value_counts()
    comps.loc[:, 'component_type_id'] = comps['component_type_id'] \
        .apply(utils.rare_category, args=(comps_distribution, ),
               cutoff=1000, value='RareComponentType')

    comps_bin = pd.get_dummies(comps['component_type_id'])
    comps_dummy = pd.concat([comps, comps_bin], axis=1)

    merged_components = pd.merge(bill_components,
                                 comps_dummy,
                                 left_on='component',
                                 right_on='component_id',
                                 how='left')
    grouped_components = merged_components \
        .groupby(merged_components.tube_assembly_id).sum()

    df = pd.merge(df, grouped_components,
                  left_on='tube_assembly_id',
                  right_index=True, how='left')
    return df


def connection_type(df, bill_components):

    comp_files = []
    comp_files.append(comp_sleeve[['component_id', 'connection_type_id']])
    comp_files.append(comp_boss[['component_id', 'connection_type_id']])

    adaptor = pd.concat(
        comp_adaptor[['component_id',
                      'connection_type_id_{}'.format(i)]]
                      .rename(columns={'connection_type_id_{}'.format(i):
                                       'connection_type_id'})
        for i in range(1, 3)
    )
    comp_files.append(adaptor)

    threaded = pd.concat(
        comp_threaded[['component_id',
                       'connection_type_id_{}'.format(i)]]
                      .rename(columns={'connection_type_id_{}'.format(i):
                                       'connection_type_id'})
        for i in range(1, 3)
    )
    comp_files.append(threaded)

    connections = pd.concat(comp_files)

    connection_distribution = connections['connection_type_id'].value_counts()
    connections.loc[:, 'connection_type_id'] = connections['connection_type_id'] \
        .apply(utils.rare_category, args=(connection_distribution, ),
               cutoff=100, value='RareConnection')

    connections_bin = pd.get_dummies(connections['connection_type_id'])
    connections_dummy = pd.concat([connections, connections_bin], axis=1)

    merged_connections = pd.merge(bill_components,
                                  connections_dummy,
                                  left_on='component',
                                  right_on='component_id',
                                  how='left')
    grouped_connections = merged_connections \
        .groupby(merged_connections.tube_assembly_id).sum()

    df = pd.merge(df, grouped_connections,
                  left_on='tube_assembly_id',
                  right_index=True, how='left')
    return df


def component(df, bill_components):

    #df = component_type(df, bill_components)
    df = connection_type(df, bill_components)
    df = sleeve.sleeve(df, bill_components, comp_sleeve)
    df = adaptor.adaptor(df, bill_components, comp_adaptor)
    df = boss.boss(df, bill_components, comp_boss)
    df = elbow.elbow(df, bill_components, comp_elbow)
    df = float_.float_(df, bill_components, comp_float)
    #df = hfl.hfl(df, bill_components, comp_hfl)
    df = nut.nut(df, bill_components, comp_nut)
    #df = other.other(df, bill_components, comp_other)
    df = straight.straight(df, bill_components, comp_straight)

    return df
