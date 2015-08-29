import pandas as pd

import utils


def part_names(other):

    part_names_distribution = other['part_name'].value_counts()
    other.loc[:, 'part_name'] = other['part_name'] \
        .apply(utils.rare_category, args=(part_names_distribution, ),
               cutoff=100, value='RarePartName')

    part_names_bin = pd.get_dummies(other['part_name'])
    part_names_dummy = pd.concat([other, part_names_bin], axis=1)

    grouped_part_names = part_names_dummy \
        .groupby(part_names_dummy.component_id).sum()

    other = pd.merge(other, grouped_part_names,
                     left_on='component_id', right_index=True)
    #other = other.drop(['RarePartName'], axis=1)
    return other


def other_impute(other):

    other = other[other['part_name'] != 'REQUIREMENT']
    other.loc[other['part_name'] == 'PLATE AS.', 'part_name'] = 'PLATE AS'
    other.loc[other['part_name'] == 'TUBE AS.', 'part_name'] = 'TUBE AS'
    other.loc[other['part_name'] == 'TUBE (BULK)', 'part_name'] = 'TUBE'
    other.loc[other['part_name'] == 'TUBE AS-FUEL IN', 'part_name'] = 'TUBE'
    other.loc[other['part_name'] == 'TUBE AS-O SUPPL', 'part_name'] = 'TUBE'
    other.loc[other['part_name'] == 'SCREEN AS.', 'part_name'] = 'SCREEN AS'
    other.loc[other['part_name'] == 'FLANGE AS.', 'part_name'] = 'FLANGE AS'
    other.loc[other['part_name'] == 'ELBOW-OIL INLET', 'part_name'] = 'ELBOW'
    other.loc[other['part_name'] == 'ELBOW-WATER', 'part_name'] = 'ELBOW'
    other.loc[other['part_name'] == 'CLAMP-INJ LINE', 'part_name'] = 'CLAMP'
    other.loc[other['part_name'] == 'ADAPTER-O DRAIN', 'part_name'] = 'ADAPTER'
    other.loc[other['part_name'] == 'NUT-MALE PIPE', 'part_name'] = 'NUT'
    other.loc[other['part_name'] == 'COVER-FRONT HSG', 'part_name'] = 'COVER'
    other.loc[other['part_name'] == 'BRACKET AS.', 'part_name'] = 'BRACKET'
    other.loc[other['part_name'] == 'FITTING-A/C', 'part_name'] = 'FITTING'
    other.loc[other['part_name'] == 'TUBE AS-RH', 'part_name'] = 'TUBE AS'
    other.loc[other['part_name'] == 'TUBE AS-LH', 'part_name'] = 'TUBE AS'

    other.loc[other['part_name'] == 'NUT-A/C', 'weight'] = \
        other.loc[other['part_name'] == 'NUT-A/C', 'weight'].apply(
            lambda x: 0.032 if pd.isnull(x) else x
        )
    other.loc[other['part_name'] == 'ADAPTER-O LINE', 'weight'] = \
        other.loc[other['part_name'] == 'ADAPTER-O LINE', 'weight'].apply(
            lambda x: 0.277 if pd.isnull(x) else x
        )
    other.loc[other['part_name'] == 'ADAPTER-A/C', 'weight'] = \
        other.loc[other['part_name'] == 'ADAPTER-A/C', 'weight'].apply(
            lambda x: 0.0585 if pd.isnull(x) else x
        )
    other.loc[other['part_name'] == 'NUT-F INJ LINE', 'weight'] = \
        other.loc[other['part_name'] == 'NUT-F INJ LINE', 'weight'].apply(
            lambda x: 0.044 if pd.isnull(x) else x
        )
    other.loc[other['part_name'] == 'ADAPTER', 'weight'] = \
        other.loc[other['part_name'] == 'ADAPTER', 'weight'].apply(
            lambda x: 0.2255 if pd.isnull(x) else x
        )
    other.loc[other['part_name'] == 'FITTING', 'weight'] = \
        other.loc[other['part_name'] == 'FITTING', 'weight'].apply(
            lambda x: 0.043 if pd.isnull(x) else x
        )
    # other.loc[other['part_name'] == 'ADAPTER-O DRAIN', 'weight'] = \
    #     other.loc[other['part_name'] == 'ADAPTER-O DRAIN', 'weight'].apply(
    #         lambda x: 0.395 if pd.isnull(x) else x
    #     )
    other.loc[other['part_name'] == 'FLANGE', 'weight'] = \
        other.loc[other['part_name'] == 'FLANGE', 'weight'].apply(
            lambda x: 0.680 if pd.isnull(x) else x
        )
    other.loc[other['part_name'] == 'PLATE AS', 'weight'] = \
        other.loc[other['part_name'] == 'PLATE AS', 'weight'].apply(
            lambda x: 0.3235 if pd.isnull(x) else x
        )
    other.loc[other['part_name'] == 'WASHER', 'weight'] = \
        other.loc[other['part_name'] == 'WASHER', 'weight'].apply(
            lambda x: 0.018 if pd.isnull(x) else x
        )
    other.loc[other['part_name'] == 'CONNECTOR-WELD', 'weight'] = \
        other.loc[other['part_name'] == 'CONNECTOR-WELD', 'weight'].apply(
            lambda x: 0.0930 if pd.isnull(x) else x
        )
    other.loc[other['part_name'] == 'NUT-SWIVEL', 'weight'] = \
        other.loc[other['part_name'] == 'NUT-SWIVEL', 'weight'].apply(
            lambda x: 0.01 if pd.isnull(x) else x
        )
    other.loc[other['part_name'] == 'TUBE AS', 'weight'] = \
        other.loc[other['part_name'] == 'TUBE AS', 'weight'].apply(
            lambda x: 0.6140 if pd.isnull(x) else x
        )
    other.loc[other['part_name'] == 'BRACKET', 'weight'] = \
        other.loc[other['part_name'] == 'BRACKET', 'weight'].apply(
            lambda x: 0.1875 if pd.isnull(x) else x
        )
    other.loc[other['part_name'] == 'ELBOW-HYDRAULIC', 'weight'] = \
        other.loc[other['part_name'] == 'ELBOW-HYDRAULIC', 'weight'].apply(
            lambda x: 2.930 if pd.isnull(x) else x
        )
    other.loc[other['part_name'] == 'FLANGE AS', 'weight'] = \
        other.loc[other['part_name'] == 'FLANGE AS', 'weight'].apply(
            lambda x: 0.01 if pd.isnull(x) else x
        )
    other.loc[other['part_name'] == 'SCREEN AS', 'weight'] = \
        other.loc[other['part_name'] == 'SCREEN AS', 'weight'].apply(
            lambda x: 0.5 if pd.isnull(x) else x
        )
    other.loc[other['part_name'] == 'CONNECTOR-SEAL', 'weight'] = \
        other.loc[other['part_name'] == 'CONNECTOR-SEAL', 'weight'].apply(
            lambda x: 0.01 if pd.isnull(x) else x
        )
    other.loc[other['part_name'] == 'TUBE', 'weight'] = \
        other.loc[other['part_name'] == 'TUBE', 'weight'].apply(
            lambda x: 2.1050 if pd.isnull(x) else x
        )
    # other.loc[other['part_name'] == 'COVER', 'weight'] = \
    #     other.loc[other['part_name'] == 'COVER', 'weight'].apply(
    #         lambda x: 0.6060 if pd.isnull(x) else x
    #     )
    other.loc[other['part_name'] == 'NUT', 'weight'] = \
        other.loc[other['part_name'] == 'NUT', 'weight'].apply(
            lambda x: 0.0325 if pd.isnull(x) else x
        )

    return other


def other_components(bill_components, other):

    other_comps = pd.merge(
        bill_components[['tube_assembly_id', 'component']], other,
        left_on='component', right_on='component_id'
    )
    grouped_other_comps = other_comps \
        .groupby(other_comps.tube_assembly_id).sum()
    return grouped_other_comps


def other(df, bill_components, other):

    other = other_impute(other)
    #other = part_names(other)
    other = other.drop(['part_name'], axis=1)

    other_comps = other_components(bill_components, other)
    df = pd.merge(df, other_comps, left_on='tube_assembly_id',
                  right_index=True, how='left')
    return df

