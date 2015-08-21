import numpy as np
import math
import random
import itertools
import collections


random.seed(a=42)

def train_test(df, base_split=0.25, recombine=0.5, unique_split_label=None):

    # divide df into sections equal to base_split %
    # if unique split label, ensure that all splits contain
    #   unique values for this label
    # recombine splits to equal recombine % of df
    #random.seed(a=42)

    if not unique_split_label:
        msk = np.random.rand(len(df)) < 1 - recombine
        train = df[msk]
        test = df[~msk]
    else:
        n_test_rows = round(len(df) * base_split)
        n_splits = math.floor(1/base_split)

        unique_label_counts = df[unique_split_label].value_counts()
        unique_labels = sorted(unique_label_counts.index.tolist())
        # print('ordered dict part')

        for i in range(n_splits-1):
            ctr = 0
            group_labels = []
            while ctr < n_test_rows:
                group_index = random.randint(0, len(unique_labels) - 1)
                new_group = unique_labels[group_index]

                unique_labels[group_index], unique_labels[-1] = \
                    unique_labels[-1], unique_labels[group_index]
                unique_labels.pop()

                group_labels.append(new_group)
                ctr += unique_label_counts[new_group]

            df.loc[df['tube_assembly_id'].isin(group_labels), 'split'] = i

        # make the leftovers the last group
        df.loc[df['tube_assembly_id'].isin(unique_labels), 'split'] = n_splits - 1

        for i in range(1, n_splits):
            msk = df['split'].isin((0, i))
            yield df[msk].drop(['split'], axis=1), df[~msk].drop(['split'], axis=1)
