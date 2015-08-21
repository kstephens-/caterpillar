import pandas as pd
import numpy as np
import xgboost as xgb

import utils
import tube
import specs
import bill_of_material
from split import train_test
from constants import yes_no_null


def bracket_pricing(df):

    df.loc[:, 'bracket_pricing'] = df['bracket_pricing'].map(yes_no_null)
    return df


def quote_date(df):

    df.loc[:, 'year'] = df.quote_date.dt.year
    df.loc[:, 'month'] = df.quote_date.dt.month
    df.loc[:, 'dayofyear'] = df.quote_date.dt.dayofyear
    df.loc[:, 'dayofweek'] = df.quote_date.dt.dayofweek
    #df.loc[:, 'day'] = df.quote_date.dt.day
    df = df.drop(['quote_date'], axis=1)

    return df


def supplier(df, supplier_distribution):

    df.loc[:, 'supplier'] = df['supplier'] \
        .apply(utils.rare_category, args=(supplier_distribution, ),
               cutoff=20, value='RareSupplier')

    supplier_bin = pd.get_dummies(df['supplier'])
    df = pd.concat([df, supplier_bin], axis=1)
    df = df.drop(['supplier'], axis=1)
    return df


def xgboost_model(train, labels, test):

    params = {}
    params['objective'] = 'reg:linear'
    params['eta'] = 0.15
    params['min_child_weight'] = 15
    params['subsample'] = 0.7
    params['scale_pos_weight'] = 1.0
    params['silent'] = 1
    params['max_depth'] = 10
    params['alpha'] = 6.25
    params['eval_metric'] = 'rmse'

    xgtrain = xgb.DMatrix(train, label=labels)
    xgtest = xgb.DMatrix(test)

    # 0.267133 at num_rounds == 1000
    num_rounds = 120
    m = xgb.train(list(params.items()), xgtrain, num_rounds)
    return m, np.expm1(m.predict(xgtest))


print('reading train data')
data = pd.read_csv('../competition_data/train_set.csv', parse_dates=[2, ])

data = data[data['quantity'] != 2500]

# tube data
tube_data = pd.read_csv('../competition_data/tube.csv')
tube_data = tube.tube(tube_data)
data = pd.merge(data, tube_data, on='tube_assembly_id')

data = specs.specs(data)
data = bill_of_material.bill_of_material(data)


scores = []
for train, test in train_test(data, unique_split_label='tube_assembly_id'):

    #print('train', train.head())
    #print('test', test.head())

    train = bracket_pricing(train)
    test = bracket_pricing(test)

    train = quote_date(train)
    test = quote_date(test)

    supplier_distribution = train['supplier'].value_counts()
    train = supplier(train, supplier_distribution)
    test = supplier(test, supplier_distribution)

    # additional
    train.loc[:, 'inv_annual_usage'] = train['annual_usage'].apply(utils.make_inv)
    test.loc[:, 'inv_annual_usage'] = test['annual_usage'].apply(utils.make_inv)

    train.loc[:, 'inv_min_order_quantity'] = \
        train['min_order_quantity'].apply(utils.make_inv)
    test.loc[:, 'inv_min_order_quantity'] = \
        test['min_order_quantity'].apply(utils.make_inv)

    # train.loc[:, 'inv_quantity'] = \
    #     train['quantity'].apply(utils.make_inv)
    # test.loc[:, 'inv_quantity'] = \
    #     test['quantity'].apply(utils.make_inv)

    # train.loc[:, 'inv_diameter'] = train['diameter'].apply(utils.make_inv)
    # test.loc[:, 'inv_diameter'] = test['diameter'].apply(utils.make_inv)

    labels = np.log1p(train.cost.values)
    answers = test.cost.values
    train = train.drop(['cost', 'tube_assembly_id', 'annual_usage',
                        'min_order_quantity'], axis=1)
    test = test.drop(['cost', 'tube_assembly_id', 'annual_usage',
                      'min_order_quantity'], axis=1)

    print('train shape is', train.shape)
    print('test shape is', test.shape)

    train = train.astype(float)
    test = test.astype(float)

    model, predictions = xgboost_model(train, labels, test)
    scores.append(utils.rmsle(answers, predictions))
    print('current score is', utils.rmsle(answers, predictions))

    inverse_model, inverse_predictions = xgboost_model(test, np.log1p(answers), train)
    scores.append(utils.rmsle(np.expm1(labels), inverse_predictions))
    print('inverse score is', utils.rmsle(np.expm1(labels), inverse_predictions))

print('Score is', np.mean(scores))
