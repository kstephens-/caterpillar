import pandas as pd

import utils


spec_data = pd.read_csv('../competition_data/specs.csv')


def specs(df):

    spec_data.loc[:, 'total_specs'] = spec_data.filter(regex='spec\d').count(axis=1)
    spec_data.loc[:, 'has_specs'] = (spec_data['total_specs'] > 0).astype(int)

    spec_codes = pd.concat(
        spec_data[['tube_assembly_id', c]].rename(columns={c: 'spec'})
        for c in spec_data.filter(regex='spec\d')
    )

    spec_distribution = spec_codes['spec'].value_counts()
    spec_codes.loc[:, 'spec'] = spec_codes['spec'] \
        .apply(utils.rare_category, args=(spec_distribution, ),
               cutoff=2000, value='RareSpec')

    spec_bin = pd.get_dummies(spec_codes['spec'])
    spec_codes = pd.concat([spec_codes, spec_bin], axis=1)
    spec_codes = spec_codes.groupby(spec_codes.tube_assembly_id).sum()

    spec_data_merged = pd.merge(spec_data, spec_codes, left_on='tube_assembly_id',
                                right_index=True, how='left')
    spec_data_final = spec_data_merged \
        .drop(['spec{}'.format(i) for i in range(1, 11)], axis=1)
    spec_data_final = spec_data_final.drop(['RareSpec', 'total_specs'], axis=1)

    df = pd.merge(df, spec_data_final, on='tube_assembly_id', how='left')

    return df
