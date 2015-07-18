import re
import math
import functools
import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing
from sklearn import svm
from sklearn import decomposition

# stuff to change
# fix dev test data so it's more reflective of actual test
#   - divide by tube assembly id (include unseen tube assemblies)
# inspect data for areas that need attention:
#   - remove outlier in 'annual_usage' (TA-19556)
#   - remove outlier in 'quantity' (TA-07001)

def make_inv(x):
    return 1 / (1 + x)

def make_sqrt_inv(x):
    return math.log(1+x)


def rmsle(labels, predictions):
    try:
        n_preds, n_targets = predictions.shape
        p = predictions.sum(axis=1)
        a = labels.sum(axis=1)
    except ValueError:
        n_preds = len(predictions)
        n_targets = 1
        p = predictions
        a = labels

    sum_squared_error = np.sum((np.log(p+1) - np.log(a+1))**2)
    return np.sqrt(1/(n_preds*n_targets) * sum_squared_error)


def train_test(df, ntest=0.2, label=None):

    # need 20% of data
    # tube asemblies must appear in either train or test but not both
    if not label:
        msk = np.random.rand(len(df)) < 1 - ntest
        train = df[msk]
        test = df[~msk]
    else:
        n_test_rows = round(len(df) * ntest)
        unique_labels = pd.unique(df[label])

        row_count = 0
        df.loc[:, 'test'] = False
        for lab in unique_labels:
            if row_count <= n_test_rows:
                affected_rows = len(df[df[label] == lab])
                df.loc[df[label] == lab, 'test'] = True
                row_count += affected_rows
        train = df[df['test'] == False].reset_index()
        test = df[df['test'] == True].reset_index()
    return train.drop(['index', 'test'], axis=1), test.drop(['index', 'test'], axis=1)


def rare_supplier(x, supplier_distribution):
    try:
        if supplier_distribution[x] < 5:
            return 'RareSupplier'
    except KeyError:
        return np.nan
        #return 'RareSupplier'
    else:
        return x


def rare_material(x, material_distribution):
    if x == '9999' or x == 'NONE':
        return np.nan
    try:
        if material_distribution[x] < 5:
            return 'RareMaterial'
    except ValueError:
        return x
    except KeyError:
        return np.nan
        #return 'RareMaterial'
    else:
        return x


def rare_end_assembly(x, end_assembly_distribution):
    if x == '9999' or x == 'NONE':
        return np.nan
    try:
        if end_assembly_distribution[x] < 5:
            return 'RareEnd'
    except KeyError:
        return np.nan
        #return 'RareEnd'
    else:
        return x


def rare_component(x, dist=None):
    if x == '9999':
        return np.nan
    try:
        if dist[x] < 40:
            return 'RareComponent'
    except (KeyError, ValueError):
        return np.nan
    else:
        return x


def rare_spec(x, spec_distribution):
    try:
        if spec_distribution[x] < 100:
            return 'RareSpec'
    except (KeyError, ValueError):
        return np.nan
    else:
        return x


def rare_component_name(x, dist):
    try:
        if dist[x] < 25:
            return 'RareComponentName'
    except (KeyError, ValueError):
        return np.nan
    else:
        return x


comp_name_re = re.compile(r'^([^ -]+)')
def comp_name_type(x):
    try:
        return comp_name_re.match(x).group(1)
    except TypeError:
        return np.nan


def rare_name_type(x, dist):
    try:
        if dist[x] < 8:
            return 'RareNameType'
    except (KeyError, ValueError):
        return np.nan
    else:
        return x


do_test = False
version = '3.0'

print('reading train data')
data = pd.read_csv('competition_data/train_set.csv', parse_dates=[2,])

# remove some weird data
#data = data[data['annual_usage'] != 150000]
#data = data[data['min_order_quantity'] != 535]

print('handling bill of material data')
bill_data = pd.read_csv('competition_data/bill_of_materials.csv')
component_dist = pd.concat(bill_data[c]
                           for c in bill_data.filter(regex='component')).value_counts()
rare_comp_p = functools.partial(rare_component,
                                dist=component_dist)
trimmed_components = bill_data.filter(regex='component_id_').applymap(rare_comp_p)
trimmed_components = trimmed_components \
    .rename(columns={'component_id_{}'.format(i): 'component_id_{}_trimmed'.format(i)
                     for i in range(1, 9)})
bill_data = pd.merge(bill_data, trimmed_components, left_index=True, right_index=True)
#bill_data.loc[:, ['{}_trimmed'.format(t) for t in trimmed_components.columns]] = trimmed_components

#add total number of components feature
bill_data.loc[:, 'total_components'] = \
    bill_data.filter(regex='quantity').sum(axis=1)
bill_data['total_components'].fillna(0, inplace=True)
bill_data.loc[:, 'has_components'] = (bill_data['total_components'] > 0).astype(int)

print('expanding component columns')
# make binary component features
# bill_components = pd.concat(
#     bill_data[['tube_assembly_id', c]].rename(columns={c: 'component'})
#     for c in bill_data.filter(regex='component_')
# )

bill_components = pd.concat(
    bill_data[['tube_assembly_id',
               'component_id_{}'.format(i),
               'component_id_{}_trimmed'.format(i),
               'quantity_{}'.format(i)]]
               .rename(columns={'component_id_{}'.format(i): 'component_orig',
                                'component_id_{}_trimmed'.format(i): 'component',
                                'quantity_{}'.format(i): 'quantity'})
    for i in range(1, 9)
)
grouped_bill_components = bill_components.groupby(
    ('tube_assembly_id', 'component'), as_index=False).sum()

bill_components_bin = pd.get_dummies(grouped_bill_components['component'])
bill_comp_bin = pd.concat([grouped_bill_components, bill_components_bin], axis=1)
print('setting component counts')
for ind in range(bill_comp_bin.shape[0]):

    bill_comp_bin.loc[ind, bill_comp_bin.loc[ind, 'component']] = \
        bill_comp_bin.loc[ind, 'quantity']

bill_comp_bin = bill_comp_bin.groupby(bill_comp_bin.tube_assembly_id,
                                      as_index=False).sum()
bill_data = pd.merge(bill_data, bill_comp_bin,
                     on='tube_assembly_id',
                     how='left')
# clean up nans for tube assemblies that don't have components
bill_data = bill_data.fillna(0)

# component names
comps = pd.read_csv('competition_data/components.csv')
comps.loc[:, 'base_name'] = comps['name'].apply(comp_name_type)

comps_base_dist = comps['base_name'].value_counts()
comps.loc[:, 'base_name'] = comps['base_name'].apply(rare_name_type,
                                                     args=(comps_base_dist, ))

comps_base_bin = pd.get_dummies(comps['base_name'])
comps_base = pd.concat([comps, comps_base_bin], axis=1)
bill_comps_base = pd.merge(bill_components, comps_base, left_on='component_orig',
                           right_on='component_id', how='left')
grouped_bill_comps_base = bill_comps_base.groupby(bill_comps_base.tube_assembly_id).sum()
grouped_bill_comps_base = grouped_bill_comps_base.drop(['RareNameType'], axis=1)
bill_data = pd.merge(bill_data, grouped_bill_comps_base,
                     left_on='tube_assembly_id',
                     right_index='tube_assembly_id',
                     how='left')

component_name_dist = comps['name'].value_counts()
comps.loc[:, 'name'] = comps['name'].apply(rare_component_name,
                                           args=(component_name_dist, ))

comp_names_bin = pd.get_dummies(comps['name'])
comps_bin = pd.concat([comps, comp_names_bin], axis=1)
bill_comps = pd.merge(bill_components, comps_bin, left_on='component_orig',
                      right_on='component_id', how='left')
grouped_bill_comps = bill_comps.groupby(bill_comps.tube_assembly_id).sum()
bill_data = pd.merge(bill_data, grouped_bill_comps,
                     left_on='tube_assembly_id',
                     right_index='tube_assembly_id',
                     how='left')
bill_data = bill_data.fillna(0)



# removing old columns
bill_data = bill_data.drop(['component_id_1', 'quantity_1',
                            'component_id_2', 'quantity_2',
                            'component_id_3', 'quantity_3',
                            'component_id_4', 'quantity_4',
                            'component_id_5', 'quantity_5',
                            'component_id_6', 'quantity_6',
                            'component_id_7', 'quantity_7',
                            'component_id_8', 'quantity_8',
                            'component_id_1_trimmed',
                            'component_id_2_trimmed',
                            'component_id_3_trimmed',
                            'component_id_4_trimmed',
                            'component_id_5_trimmed',
                            'component_id_6_trimmed',
                            'component_id_7_trimmed',
                            'component_id_8_trimmed',
                            'quantity_x', 'quantity_y'], axis=1)

# spec data
specs = pd.read_csv('competition_data/specs.csv')
specs.loc[:, 'total_specs'] = specs.filter(regex='spec\d').count(axis=1)
specs.loc[:, 'has_specs'] = (specs['total_specs'] > 0).astype(int)
reduced_specs = specs[['tube_assembly_id', 'total_specs', 'has_specs']]

# spec_data = pd.concat(
#     specs[['tube_assembly_id', c]].rename(columns={c: 'spec'})
#     for c in specs.filter(regex='spec\d')
# )
# spec_distribution = spec_data['spec'].value_counts()
# spec_data.loc[:, 'spec'] = spec_data['spec'].apply(rare_spec, args=(spec_distribution, ))

# spec_data_bin = pd.get_dummies(spec_data['spec'])
# spec_bin = pd.concat([spec_data, spec_data_bin], axis=1)
# spec_bin = spec_bin.groupby(spec_bin.tube_assembly_id).sum()
# #print('spec data bin shape', spec_data_bin.shape)
# #print('spec bin shape', spec_bin.shape)
# #print('spec shape is', specs.shape)
# specs = pd.merge(specs, spec_bin, left_on='tube_assembly_id',
#                  right_index=True)
# #print('spec shape after merge', specs.shape)

# # remove old spec columns
# specs = specs.drop(['spec{}'.format(i) for i in range(1, 11)], axis=1)
# reduced_specs = specs


print('handling tube data')
tube_data = pd.read_csv('competition_data/tube.csv')
# #data = pd.merge(data, tube_data, on='tube_assembly_id')

# end 1x and 2x
end_mapper = {'Y': True, 'N': False}
tube_data.loc[:, 'end_a_1x'] = tube_data['end_a_1x'].map(end_mapper).astype(int)
tube_data.loc[:, 'end_a_2x'] = tube_data['end_a_2x'].map(end_mapper).astype(int)
tube_data.loc[:, 'end_x_1x'] = tube_data['end_x_1x'].map(end_mapper).astype(int)
tube_data.loc[:, 'end_x_2x'] = tube_data['end_x_2x'].map(end_mapper).astype(int)

# material id
material_distribution = tube_data['material_id'].value_counts()
tube_data.loc[:, 'material_id'] = \
    tube_data['material_id'].apply(rare_material,
                                   args=(material_distribution, ))
materials_bin = pd.get_dummies(tube_data['material_id'])
tube_data = pd.concat([tube_data, materials_bin], axis=1)

# end assemblies
tube_ends = tube_data[['end_a', 'end_x']]
end_a_dist = tube_ends['end_a'].value_counts()
end_x_dist = tube_ends['end_x'].value_counts()

tube_ends.loc[:, 'end_a'] = tube_ends['end_a'] \
    .apply(rare_end_assembly, args=(end_a_dist, ))
tube_ends.loc[:, 'end_x'] = tube_ends['end_x'] \
    .apply(rare_end_assembly, args=(end_x_dist, ))

end_cols = pd.get_dummies(tube_ends)
tube_data = pd.concat([tube_data, end_cols], axis=1)

end_assemblies = pd.concat(
    tube_data[['tube_assembly_id', c]].rename(columns={c: 'end'})
    for c in ['end_a', 'end_x']
)
end_distribution = end_assemblies['end'].value_counts()
end_assemblies.loc[:, 'end'] = end_assemblies['end'] \
    .apply(rare_end_assembly, args=(end_distribution, ))

end_assemblies_bin = pd.get_dummies(end_assemblies['end'])
end_ass_bin = pd.concat([end_assemblies, end_assemblies_bin], axis=1)
end_ass_bin = end_ass_bin.groupby(end_ass_bin.tube_assembly_id).sum()
tube_data = pd.merge(tube_data, end_ass_bin, left_on='tube_assembly_id',
                     right_index=True)

# end forms
forming_map = {'No': False, 'Yes': True}
end_forms = pd.read_csv('competition_data/tube_end_form.csv')

tube_data = pd.merge(tube_data, end_forms, left_on='end_a',
                     right_on='end_form_id', how='left')
tube_data = tube_data.rename(columns={'forming': 'forming_a'})
tube_data.loc[:, 'forming_a'] = tube_data['forming_a'] \
    .fillna('No').map(forming_map).astype(int)
tube_data = tube_data.drop(['end_form_id'], axis=1)

tube_data = pd.merge(tube_data, end_forms, left_on='end_x',
                     right_on='end_form_id', how='left')
tube_data = tube_data.rename(columns={'forming': 'forming_x'})
tube_data.loc[:, 'forming_x'] = tube_data['forming_x'] \
    .fillna('No').map(forming_map).astype(int)
tube_data = tube_data.drop(['end_form_id'], axis=1)

tube_data = tube_data.drop(['material_id', 'end_a', 'end_x'], axis=1)


print('making train/test data')
if do_test:
    train = data
    test = pd.read_csv('competition_data/test_set.csv', parse_dates=[3,])
    #test = pd.merge(test, tube_data, on='tube_assembly_id')
else:
    train, test = train_test(data, label='tube_assembly_id')

bracket_pricing_map = {'Yes': True, 'No': False}
train = pd.merge(train, tube_data, on='tube_assembly_id', how='left')
train = pd.merge(train, bill_data, on='tube_assembly_id', how='left')
train = pd.merge(train, reduced_specs, on='tube_assembly_id', how='left')
train.loc[:, 'bracket_pricing'] = train['bracket_pricing'] \
    .map(bracket_pricing_map).astype(int)

test = pd.merge(test, tube_data, on='tube_assembly_id', how='left')
test = pd.merge(test, bill_data, on='tube_assembly_id', how='left')
test = pd.merge(test, reduced_specs, on='tube_assembly_id', how='left')
test.loc[:, 'bracket_pricing'] = test['bracket_pricing'] \
    .map(bracket_pricing_map).astype(int)

supplier_distribution = train['supplier'].value_counts()
train.loc[:, 'supplier'] = train['supplier'] \
    .apply(rare_supplier, args=(supplier_distribution, ))
test.loc[:, 'supplier'] = test['supplier'] \
    .apply(rare_supplier, args=(supplier_distribution, ))
print('trainsformed supplier')

print('setting date columns')
train.loc[:, 'year'] = train.quote_date.dt.year
train.loc[:, 'month'] = train.quote_date.dt.month
train.loc[:, 'dayofyear'] = train.quote_date.dt.dayofyear
train.loc[:, 'dayofweek'] = train.quote_date.dt.dayofweek
train.loc[:, 'day'] = train.quote_date.dt.day

test.loc[:, 'year'] = test.quote_date.dt.year
test.loc[:, 'month'] = test.quote_date.dt.month
test.loc[:, 'dayofyear'] = test.quote_date.dt.dayofyear
test.loc[:, 'dayofweek'] = test.quote_date.dt.dayofweek
test.loc[:, 'day'] = test.quote_date.dt.day
print('set date columns')


train.loc[:, 'inv_annual_usage'] = train['annual_usage'].apply(lambda x: make_inv(x))
#train.loc[:, 'inv_sqrt_quantity'] = train['quantity'].apply(lambda x: make_sqrt_inv(x))
#train.loc[:, 'inv_quantity'] = train['quantity'].apply(lambda x: make_inv(x))
#train.loc[:, 'inv_min_order_quantity'] = train['min_order_quantity'].apply(lambda x: make_inv(x))
# train.loc[:, 'material_volume'] = \
#    ((math.pi*(train['diameter']/2)**2) - math.pi*((train['diameter']-train['wall'])/2)**2) * train['length']
# train.loc[:, 'material_volume'] = 1 / np.log(1 + train['material_volume'])
#train.loc[:, 'has_brackets'] = (train['num_bracket'] > 0).astype(int)
#train.loc[:, 'has_bends'] = (train['num_bends'] > 0).astype(int)
train.loc[:, 'has_boss'] = (train['num_boss'] > 0).astype(int)

test.loc[:, 'inv_annual_usage'] = test['annual_usage'].apply(lambda x: make_inv(x))
#test.loc[:, 'inv_sqrt_quantity'] = test['quantity'].apply(lambda x: make_sqrt_inv(x))
#test.loc[:, 'inv_quantity'] = test['quantity'].apply(lambda x: make_inv(x))
#test.loc[:, 'inv_min_order_quantity'] = test['annual_usage'].apply(lambda x: make_inv(x))
# test.loc[:, 'material_volume'] = \
#    ((math.pi*(test['diameter']/2)**2) - math.pi*((test['diameter']-test['wall'])/2)**2) * test['length']
# test.loc[:, 'material_volume'] = 1 / np.log(1 + test['material_volume'])
#test.loc[:, 'has_brackets'] = (test['num_bracket'] > 0).astype(int)
#test.loc[:, 'has_bends'] = (test['num_bends'] > 0).astype(int)
test.loc[:, 'has_boss'] = (test['num_boss'] > 0).astype(int)


# labels
labels = np.log1p(train.cost.values)
train = train.drop(['quote_date',
                    'cost',
                    'tube_assembly_id',
                    'supplier'], axis=1)
test = test.drop(['quote_date',
                  'tube_assembly_id',
                  'supplier'], axis=1)
if not do_test:
    answers = test.cost.values
    test = test.drop(['cost'], axis=1)

else:
    idx = test['id']
    test = test.drop(['id'], axis=1)

print('train shape is', train.shape)
print('test shape is', test.shape)

#train = train.fillna(0)
#test = test.fillna(0)

train = np.array(train)
test = np.array(test)

# print('converting categorical variables')
# for i in [2, 9, 10, 11, 12]:
#     lbl = preprocessing.LabelEncoder()
#     lbl.fit(list(train[:, i]) + list(test[:, i]))
#     train[:, i] = lbl.transform(train[:, i])
#     test[:, i] = lbl.transform(test[:, i])

train = train.astype(float)
test = test.astype(float)

# print('training RF model')
# model = 'RF'
# rfr = ensemble.RandomForestRegressor(n_estimators=50,
#                                      max_depth=15,
#                                      min_weight_fraction_leaf=0.0,
#                                      random_state=42,
#                                      n_jobs=-1)
# rfr.fit(train, labels)
# print('predicting with RF model')
# predictions = rfr.predict(test)
# predictions = np.expm1(predictions)

print('traing gbr model')
model = 'gbr'
gbr = ensemble.GradientBoostingRegressor(n_estimators=1000,
                                         learning_rate=0.1,
                                         subsample=0.8,
                                         max_depth=15,
                                         random_state=42,
                                         loss='lad')
gbr.fit(train, labels)
print('predicting with gbr model')
predictions = gbr.predict(test)
predictions = np.expm1(predictions)


if do_test:
    print('writing predictions to file')
    df_predictions = pd.DataFrame({'id': idx, 'cost': predictions})
    df_predictions.to_csv('submissions/{}_v{}.csv'.format(model, version),
                          index=False)
else:
    print('Score is', rmsle(answers, predictions))
    print()





# supplier_dist = data['supplier'].value_counts()
# def rare_supplier(x):
#     try:
#         if supplier_dist[x] < 10:
#             return 'RareSupplier'
#     except KeyError:
#         return 'RareSupplier'
#     return x

# material_dist = data['material_id'].value_counts()
# def rare_material(x):
#     try:
#         if material_dist[x] < 10:
#             return 'RareMaterial'
#     except ValueError:
#         return x
#     except KeyError:
#         return 'RareMaterial'
#     return x

# end_assembly_dist = pd.concat([data['end_a'], data['end_x']], axis=0).value_counts()
# def rare_end_assembly(x):
#     try:
#         if end_assembly_dist[x] < 10:
#             return 'RareEnd'
#     except KeyError:
#         return 'RareEnd'
#     return x


# print('trainsforming data')
# # rare suppliers
# data.loc[:, 'supplier'] = data['supplier'].apply(lambda x: rare_supplier(x))
# print('trainsformed supplier')

# data.loc[:, 'material_id'] = data['material_id'].apply(lambda x: rare_material(x))
# print('transformed material_id')

# data.loc[:, 'end_a'] = data['end_a'].apply(lambda x: rare_end_assembly(x))
# data.loc[:, 'end_x'] = data['end_x'].apply(lambda x: rare_end_assembly(x))
# print('transformed end_a and end_x')

# print('setting date columns')
# data.loc[:,'year'] = data.quote_date.dt.year
# data.loc[:,'month'] = data.quote_date.dt.month
# data.loc[:,'dayofyear'] = data.quote_date.dt.dayofyear
# data.loc[:,'dayofweek'] = data.quote_date.dt.dayofweek
# data.loc[:,'day'] = data.quote_date.dt.day
# print('set train date columns')

# if do_test:
#     print('transforming test data')
#     test.loc[:, 'supplier'] = test['supplier'].apply(lambda x: rare_supplier(x))
#     print('transformed suppliers')

#     test.loc[:, 'material_id'] = test['material_id'].apply(lambda x: rare_material(x))
#     print('transformed material_id')

#     test.loc[:, 'end_a'] = test['end_a'].apply(lambda x: rare_end_assembly(x))
#     test.loc[:, 'end_x'] = test['end_x'].apply(lambda x: rare_end_assembly(x))
#     print('transformed end_a and end_x')

#     print('setting date columns')
#     test.loc[:, 'year'] = test.quote_date.dt.year
#     test.loc[:, 'month'] = test.quote_date.dt.month
#     test.loc[:, 'dayofyear'] = test.quote_date.dt.dayofyear
#     test.loc[:, 'dayofweek'] = test.quote_date.dt.dayofweek
#     test.loc[:, 'day'] = test.quote_date.dt.day

# #data.loc[:, 'invQuant'] = 1 / (1 + data['quantity'])

# suppliers = pd.get_dummies(data['supplier'])
# data = pd.concat([data, suppliers], axis=1)

# materials = pd.get_dummies(data['material_id'])
# data = pd.concat([data, materials], axis=1)

# ends = pd.get_dummies(data[['end_a', 'end_x']])
# data = pd.concat([data, ends], axis=1)

# if do_test:
#     test_suppliers = pd.get_dummies(test['supplier'])
#     test = pd.concat([test, test_suppliers], axis=1)

#     test_materials = pd.get_dummies(test['material_id'])
#     test = pd.concat([test, test_materials], axis=1)

#     test_ends = pd.get_dummies(test[['end_a', 'end_x']])
#     test = pd.concat([test, test_ends], axis=1)

# # # end connectors
# # # print('computing end connector features')
# # # for end_id, count in zip(end_assembly_dist.index, end_assembly_dist):
# # #     col = rare_end_assembly(end_id)
# # #     if col != 'None' and col not in data:
# # #         data.loc[:, col] = 0

# # # for i in range(len(data)):
# # #     end_a = data.loc[i, 'end_a']
# # #     end_x = data.loc[i, 'end_x']

# # #     if end_a in data and data.loc[i, end_a] is 0:
# # #         data.loc[i, end_a] = 1

# # #     if end_x in data and data.loc[i, end_x] is 0:
# # #         data.loc[i, end_x] = 1


# # # # make train and dev test sets
# if not do_test:
#     print('making train and dev test sets')
#     msk = np.random.rand(len(data)) < 0.8
#     train = data[msk]
#     dev = data[~msk]
#     print('made train and dev test sets')
# else:
#     train = data

# # # separate date parts
# # print('setting date columns')
# # train.loc[:,'year'] = train.quote_date.dt.year
# # train.loc[:,'month'] = train.quote_date.dt.month
# # train.loc[:,'dayofyear'] = train.quote_date.dt.dayofyear
# # train.loc[:,'dayofweek'] = train.quote_date.dt.dayofweek
# # train.loc[:,'day'] = train.quote_date.dt.day
# # print('set train date columns')

# # print('setting dev date columns')
# # dev.loc[:,'year'] = dev.quote_date.dt.year
# # dev.loc[:,'month'] = dev.quote_date.dt.month
# # dev.loc[:,'dayofyear'] = dev.quote_date.dt.dayofyear
# # dev.loc[:,'dayofweek'] = dev.quote_date.dt.dayofweek
# # dev.loc[:,'day'] = dev.quote_date.dt.day
# # print('set dev date columns')
# # print('set date columns')

# # # suppliers = pd.get_dummies(train['supplier'] + dev['supplier'])
# # # train = pd.concat([train, suppliers], axis=1)
# # # dev = pd.concat([dev, suppliers], axis=1)

# # # materials = pd.get_dummies(train['material_id'] + dev['material_id'])
# # # train = pd.concat([train, materials], axis=1)
# # # dev = pd.concat([dev, materials], axis=1)

# # labels
# print('getting labels')
# train_labels = np.log1p(train.cost.values)
# train = train.drop(['quote_date', 'cost', 'tube_assembly_id', 'supplier', 'material_id', 'end_a', 'end_x'], axis=1)

# if not do_test:
#     dev_labels = dev.cost.values
#     dev = dev.drop(['quote_date', 'cost', 'tube_assembly_id', 'supplier', 'material_id', 'end_a', 'end_x'], axis=1)
#     dev = np.array(dev)
# else:
#     idx = test['id']

#     # add missing training columns back in
#     missing = set(train.columns.values) - set(test.columns.values)
#     for col in missing:
#         test.loc[:, col] = 0

#     test = test.drop(['quote_date', 'id', 'tube_assembly_id', 'supplier', 'material_id', 'end_a', 'end_x'], axis=1)
#     test = np.array(test)

# train = np.array(train)

# # # print('droping unnecessary columns')
# # ## PUT 'end_a', 'end_x' back into model

# # # print('converting to np array')
# # # train = np.array(train)
# # # dev = np.array(dev)

# print('converting categorical variables')
# # # label encode categorical variables
# for i in [2, 9, 10, 11, 12]:
# # for i in [3, 11, 12, 13, 14]:
#     lbl = preprocessing.LabelEncoder()
#     if do_test:
#         lbl.fit(list(train[:, i]))
#     else:
#         lbl.fit(list(train[:,i]) + list(dev[:,i]))
#     train[:,i] = lbl.transform(train[:,i])
#     if do_test:
#         test[:, i] = lbl.transform(test[:, i])
#     else:
#         dev[:,i] = lbl.transform(dev[:,i])

# # # handle end assemblies separately
# # # lbl = preprocessing.LabelEncoder()
# # # lbl.fit(list(train[:,13]) + list(train[:,14]) + list(dev[:,13]) + list(dev[:,14]))
# # # train[:,13] = lbl.transform(train[:,13])
# # # train[:,14] = lbl.transform(train[:,14])
# # # dev[:,13] = lbl.transform(dev[:,13])
# # # dev[:,14] = lbl.transform(dev[:,14])

# # # onehot encode where appropriate
# # # enc = preprocessing.OneHotEncoder(categorical_features=[0], sparse=False)
# # # enc.fit(np.vstack((train, dev)))
# # # train = enc.transform(train)
# # # dev = enc.transform(dev)

# train = train.astype(float)
# if do_test:
#     test = test.astype(float)
# else:
#     dev = dev.astype(float)

# # # decomposition
# # pca = decomposition.PCA(n_components=20)
# # pca.fit(train)
# # train = pca.transform(train)
# # dev = pca.transform(dev)

# test_set = test if do_test else dev

# print('training RF model')
# rfr = ensemble.RandomForestRegressor(n_estimators=100, max_depth=25, min_weight_fraction_leaf=0.0,
#                                      random_state=42, n_jobs=-1)
# rfr.fit(train, train_labels)
# print('predicting dev values')
# rf_predictions = rfr.predict(dev)
# rf_predictions = np.expm1(rf_predictions)

# # print('training gradient boost regressor')
# # gbr = ensemble.GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=25, random_state=42,
# #                                          loss='lad')
# # gbr.fit(train, train_labels)
# # print('predictin dev values')
# # gbr_predictions = gbr.predict(test_set)
# # gbr_predictions = np.expm1(gbr_predictions)

# # # print('training svr')
# # # svr = svm.SVR(kernel='linear', C=1.0)
# # # svr.fit(train, train_labels)
# # # print('predicting dev values')
# # # svr_predictions = svr.predict(dev)

# # # score
# if do_test:
#     print('writing predictions to file')
#     df_predictions = pd.DataFrame({'id': idx, 'cost': gbr_predictions})
#     df_predictions.to_csv('submissions/{}_v{}.csv'.format('gbr', version), index=False)
# else:
#     print('The RF rmsle is', rmsle(dev_labels, rf_predictions))
#     print()
    # print('The boosting rmsle is', rmsle(dev_labels, gbr_predictions))
    # print()
#print('The svr rmsle is', rmsle(dev_labels, svr_predictions))
