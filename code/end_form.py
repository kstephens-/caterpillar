import pandas as pd

from constants import yes_no_null


end_form_data = pd.read_csv('../competition_data/tube_end_form.csv')


def end_forms(df):

    df = pd.merge(df, end_form_data, left_on='end_a',
                  right_on='end_form_id', how='left')
    df = df.rename(columns={'forming': 'forming_a'})
    df.loc[:, 'forming_a'] = df['forming_a'].map(yes_no_null)

    df = pd.merge(df, end_form_data, left_on='end_x',
                  right_on='end_form_id', how='left')
    df = df.rename(columns={'forming': 'forming_x'})
    df.loc[:, 'forming_x'] = df['forming_x'].map(yes_no_null)

    df = df.drop(['end_form_id_x', 'end_form_id_y'], axis=1)
    return df
