import pandas as pd
import numpy as np
import xgboost as xgb
import operator
import math

from sklearn import decomposition

import utils
import tube
import specs
import bill_of_material
from split import train_test
from constants import yes_no_null


test = True
version = '5.15'


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

    # print()
    # print('supplier distribution')
    # print(supplier_distribution.index)
    # print()
    supplier_bin = pd.get_dummies(df['supplier'])
    # supplier_bin = supplier_bin.drop([
    #    c for c in supplier_bin.columns
    #    if c not in supplier_distribution.index
    #    or c == 'RareSupplier'
    # ], axis=1)
    df = pd.concat([df, supplier_bin], axis=1)
    df = df.drop(['supplier'], axis=1)
    return df


def supplier2(train, test, supplier_distribution):

    train_suppliers = train[['tube_assembly_id', 'supplier']]
    test_suppliers = test[['tube_assembly_id', 'supplier']]

    train_suppliers.loc[:, 'train'] = True
    test_suppliers.loc[:, 'train'] = False

    suppliers = pd.concat([train_suppliers, test_suppliers])

    suppliers.loc[:, 'supplier'] = suppliers['supplier'] \
        .apply(utils.rare_category, args=(supplier_distribution, ),
               cutoff=1, value='RareSupplier')

    suppliers_bin = pd.get_dummies(suppliers['supplier'])
    suppliers_dummy = pd.concat([suppliers, suppliers_bin], axis=1)

    msk = suppliers_dummy['train']
    suppliers_dummy.drop(['train', 'supplier'], axis=1, inplace=True)

    #return suppliers_dummy[msk], suppliers_dummy[~msk]

    # print()
    # print('train shape', train.shape)
    # print('test shape', test.shape)
    #print('train dummy columns', suppliers_dummy.columns)
    #print('test dummy columns', suppliers_dummy.columns)
    # print()
    train_supplier = pd.concat([train, suppliers_dummy[msk]], axis=1)
    test_supplier = pd.concat([test, suppliers_dummy[~msk]], axis=1)
    # train_supplier = pd.merge(train, suppliers_dummy[msk], on='tube_assembly_id',
    #                           how='left')
    # test_supplier = pd.merge(test, suppliers_dummy[~msk], on='tube_assembly_id',
    #                          how='left')
    # print('train supplier shape', train_supplier.shape)
    # print('test supplier shape', test_supplier.shape)
    # print()
    return train_supplier, test_supplier


def xgboost_model(train, labels, test):

    params = {}
    params['objective'] = 'reg:linear'
    params['eta'] = 0.02
    #params['gamma'] = 10
    params['min_child_weight'] = 20
    params['subsample'] = 0.7
    params['colsample_bytree'] = 0.75
    params['max_delta_step'] = 5
    params['silent'] = 1
    params['max_depth'] = 25
    params['alpha'] = 0.5
    params['lambda'] = 10
    params['eval_metric'] = 'rmse'

    xgtrain = xgb.DMatrix(train, label=labels)
    xgtest = xgb.DMatrix(test)

    # 0.252876 at num_rounds == 1200
    # 0.258725 at num_rounds == 500 cv
    # 0.260958 at num_rounds == 120 cv
    num_rounds = 2000
    m = xgb.train(list(params.items()), xgtrain, num_rounds)
    return m, np.expm1(m.predict(xgtest))


def prepare_data(df, supplier_distribution):

    df = bracket_pricing(df)
    df = quote_date(df)

    #df = supplier(df, supplier_distribution)

    # additional
    df.loc[:, 'inv_annual_usage'] = df['annual_usage'].apply(utils.make_inv)
    #df.loc[:, 'annual_usage_quantity'] = np.log(df['annual_usage'] / df['quantity'])

    df.loc[:, 'inv_min_order_quantity'] = \
        df['min_order_quantity'].apply(utils.make_inv)

    df.loc[:, 'inv_quantity'] = \
        df['quantity'].apply(utils.make_inv)

    # df.loc[:, 'quant_per_month'] = df['quantity'] \
    #     .apply(lambda x: (12 / x) ** (1/3) if x < 2 else 'None')
    # df.loc[df['quant_per_month'] == 'None', 'quant_per_month'] = \
    #     np.mean(df.loc[df['quant_per_month'] != 'None', 'quant_per_month'])

    #df.loc[:, 'bend_ratio'] = (df['bend_radius'] / df['num_bends']) / df['quantity']**2
    # df.loc[:, 'quantity_ration'] = df['quantity_ratio'] \
    #     .apply(lambda x: 0 if x >= 29 else x)

    # df.loc[:, 'inv_sqrt_quantity'] = \
    #     df['quantity'].apply(lambda x: 1 / x if x else x)
    # df.loc[:, 'inv_sqrt_quantity'] = \
    #     df['quantity'].apply(lambda x: 1 / x if x else x)

    # df.loc[:, 'diameter_quantity'] = (df['diameter'] / df['quantity'])
    # df.loc[:, 'length_quantity'] = (df['length'] / df['quantity'])
    # df.loc[:, 'wall_quantity'] = (df['wall'] / df['quantity'])
    # df.loc[:, 'volume_quantity'] = (df['volume'] / df['quantity'])
    #df.loc[:, 'bend_radius_quantity'] = (df['bend_radius'] / df['quantity'])
    #df.loc[:, 'num_bends_quantity'] = (df['num_bends'] / df['quantity']) ** (1/3)
    #df.loc[:, 'bend_ratio'] = ((df['num_bends'] * df['bend_radius']) / df['quantity']) ** (1/3)

    # train.loc[:, 'inv_diameter'] = train['diameter'].apply(utils.make_inv)
    # test.loc[:, 'inv_diameter'] = test['diameter'].apply(utils.make_inv)
    return df

print('reading train data')
data = pd.read_csv('../competition_data/train_set.csv', parse_dates=[2, ])

data = data[data['quantity'] != 2500]
#data = data[data['annual_usage'] != 150000]

# tube data
tube_data = pd.read_csv('../competition_data/tube.csv')
tube_data = tube.tube(tube_data)
data = pd.merge(data, tube_data, on='tube_assembly_id')

data = specs.specs(data)
data = bill_of_material.bill_of_material(data)

#data.loc[:, 'diameter'] = (data['diameter'] / data['quantity']) ** (1/3)
#data.loc[:, 'length_quantity'] = (data['length'] / data['quantity']) ** (1/3)
#data.loc[:, 'wall_quantity'] = (data['wall'] / data['quantity']) ** (1/3)
#data.loc[:, 'volume_quantity'] = (data['volume'] / data['quantity'])
feature_importance = {}
if not test:
    scores = []
    for train, test in train_test(data, unique_split_label='tube_assembly_id'):

        #print('train', train.head())
        #print('test', test.head())
        supplier_distribution = train['supplier'].value_counts()

        train = prepare_data(train, supplier_distribution)
        test = prepare_data(test, supplier_distribution)

        # print()
        # print('train shape', train.shape)
        # print('test shape', test.shape)
        # print()
        train, test = supplier2(train, test, supplier_distribution)

        #train.loc[:, 'doy_quantity'] = (train['dayofyear'] / train['quantity']) ** (1/3)
        #test.loc[:, 'doy_quantity'] = (test['dayofyear'] / test['quantity']) ** (1/3)
        # print()
        # print('train shape', train.shape)
        # print('test shape', test.shape)
        # print()

        # train = bracket_pricing(train)
        # test = bracket_pricing(test)

        # train = quote_date(train)
        # test = quote_date(test)

        # supplier_distribution = train['supplier'].value_counts()
        # train = supplier(train, supplier_distribution)
        # test = supplier(test, supplier_distribution)

        # # additional
        # train.loc[:, 'inv_annual_usage'] = train['annual_usage'].apply(utils.make_inv)
        # test.loc[:, 'inv_annual_usage'] = test['annual_usage'].apply(utils.make_inv)

        # train.loc[:, 'inv_min_order_quantity'] = \
        #     train['min_order_quantity'].apply(utils.make_inv)
        # test.loc[:, 'inv_min_order_quantity'] = \
        #     test['min_order_quantity'].apply(utils.make_inv)

        # train.loc[:, 'inv_quantity'] = \
        #     train['quantity'].apply(utils.make_inv)
        # test.loc[:, 'inv_quantity'] = \
        #     test['quantity'].apply(utils.make_inv)

        # train.loc[:, 'inv_diameter'] = train['diameter'].apply(utils.make_inv)
        # test.loc[:, 'inv_diameter'] = test['diameter'].apply(utils.make_inv)

        labels = np.log1p(train.cost.values)
        answers = test.cost.values
        train.drop(['cost', 'tube_assembly_id', 'annual_usage',
                    'min_order_quantity', 'supplier'], axis=1, inplace=True)
        test.drop(['cost', 'tube_assembly_id', 'supplier',
                   'min_order_quantity', 'annual_usage',], axis=1, inplace=True)

        # train.drop(['S-0095', 'S-0060', 'S-0097', 'S-0104'],
        #            axis=1, inplace=True)
        # test.drop(['S-0095', 'S-0060', 'S-0097', 'S-0104'],
        #            axis=1, inplace=True)

        #train = train.drop(train.columns[87], axis=1)
        #test = test.drop(test.columns[87], axis=1)

        print('train shape is', train.shape)
        print('test shape is', test.shape)
        train = train.fillna(0)
        test = test.fillna(0)

        train_natural = train.copy()
        test_natural = test.copy()

        train = train.astype(float)
        test = test.astype(float)

        model, predictions = xgboost_model(train, labels, test)
        importance = model.get_fscore()
        importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
        total_importance = sum(i[1] for i in importance)
        for imp in importance[:20]:
            ind = int(imp[0][1:]) - 1
            #print(ind)
            try:
                feature_importance[train_natural.columns[ind]] += 1
            except KeyError:
                feature_importance[train_natural.columns[ind]] = 1
        # print()
        # print('importance')
        # print(importance[:20])
        # print()
        scores.append(utils.rmsle(answers, predictions))
        print('current score is', utils.rmsle(answers, predictions))

        inverse_model, inverse_predictions = xgboost_model(test, np.log1p(answers), train)

        scores.append(utils.rmsle(np.expm1(labels), inverse_predictions))
        print('inverse score is', utils.rmsle(np.expm1(labels), inverse_predictions))

    print('Score is', np.mean(scores))
else:
    test = pd.read_csv('../competition_data/test_set.csv', parse_dates=[3,])

    test = pd.merge(test, tube_data, on='tube_assembly_id')
    test = specs.specs(test)
    test = bill_of_material.bill_of_material(test)

    supplier_distribution = data['supplier'].value_counts()
    train = prepare_data(data, supplier_distribution)
    test = prepare_data(test, supplier_distribution)

    train, test = supplier2(train, test, supplier_distribution)

    train.to_csv('../graph_data/full_train.csv', index=False)

    labels = np.log1p(train.cost.values)
    idx = test['id']

    train = train.drop(['cost', 'tube_assembly_id', 'annual_usage',
                        'min_order_quantity', 'supplier'], axis=1)
    test = test.drop(['tube_assembly_id', 'annual_usage',
                      'min_order_quantity', 'id', 'supplier'], axis=1)

    print('train shape is', train.shape)
    print('test shape is', test.shape)
    train = train.fillna(0)
    test = test.fillna(0)

    train = train.astype(float)
    test = test.astype(float)

    model, predictions = xgboost_model(train, labels, test)
    predictions_df = pd.DataFrame({'id': idx, 'cost': predictions})
    predictions_df.to_csv('../submissions/{}_v{}.csv'.format('xgb', version),
                          index=False)

