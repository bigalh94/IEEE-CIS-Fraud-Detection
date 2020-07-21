import warnings
warnings.filterwarnings("ignore")

from numba import jit
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import graphviz

from tqdm import tqdm_notebook

from tqdm import tqdm_notebook as tqdm
from IPython.display import display

import os
import gc
import random


pd.options.display.max_rows = 10000
pd.options.display.max_columns = 10000
pd.options.display.max_colwidth = 1000

def count_uniques(train, test, pair):
    unique_train = []
    unique_test = []

    for value in train[pair[0]].unique():
        unique_train.append(train[pair[1]][train[pair[0]] == value].value_counts().shape[0])

    for value in test[pair[0]].unique():
        unique_test.append(test[pair[1]][test[pair[0]] == value].value_counts().shape[0])

    pair_values_train = pd.Series(data=unique_train, index=train[pair[0]].unique())
    pair_values_test = pd.Series(data=unique_test, index=test[pair[0]].unique())

    return pair_values_train, pair_values_test

def fill_card_nans(train, test, pair_values_train, pair_values_test, pair):
    print(f'In train{[pair[1]]} there are {train[pair[1]].isna().sum()} NaNs' )
    print(f'In test{[pair[1]]} there are {test[pair[1]].isna().sum()} NaNs' )

    print('Filling train...')

    for value in pair_values_train[pair_values_train == 1].index:
        train[pair[1]][train[pair[0]] == value] = train[pair[1]][train[pair[0]] == value].value_counts().index[0]

    print('Filling test...')

    for value in pair_values_test[pair_values_test == 1].index:
        test[pair[1]][test[pair[0]] == value] = test[pair[1]][test[pair[0]] == value].value_counts().index[0]

    print(f'In train{[pair[1]]} there are {train[pair[1]].isna().sum()} NaNs' )
    print(f'In test{[pair[1]]} there are {test[pair[1]].isna().sum()} NaNs' )

    return train, test

def nans_distribution(train, test, unique_train, unique_test, pair):
    train_nans_per_category = []
    test_nans_per_category = []

    for value in unique_train.unique():
        train_nans_per_category.append(train[train[pair[0]].isin(list(unique_train[unique_train == value].index))][pair[1]].isna().sum())

    for value in unique_test.unique():
        test_nans_per_category.append(test[test[pair[0]].isin(list(unique_test[unique_test == value].index))][pair[1]].isna().sum())

    pair_values_train = pd.Series(data=train_nans_per_category, index=unique_train.unique())
    pair_values_test = pd.Series(data=test_nans_per_category, index=unique_test.unique())

    return pair_values_train, pair_values_test

    # safe downcast
    def sd(col, max_loss_limit=0.001, avg_loss_limit=0.001, na_loss_limit=0, n_uniq_loss_limit=0, fillna=0):
        """
        max_loss_limit - don't allow any float to lose precision more than this value. Any values are ok for GBT algorithms as long as you don't unique values.
                         See https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Precision_limitations_on_decimal_values_in_[0,_1]
        avg_loss_limit - same but calculates avg throughout the series.
        na_loss_limit - not really useful.
        n_uniq_loss_limit - very important parameter. If you have a float field with very high cardinality you can set this value to something like n_records * 0.01 in order to allow some field relaxing.
        """
        is_float = str(col.dtypes)[:5] == 'float'
        na_count = col.isna().sum()
        n_uniq = col.nunique(dropna=False)
        try_types = ['float16', 'float32']

        if na_count <= na_loss_limit:
            try_types = ['int8', 'int16', 'float16', 'int32', 'float32']

        for type in try_types:
            col_tmp = col

            # float to int conversion => try to round to minimize casting error
            if is_float and (str(type)[:3] == 'int'):
                col_tmp = col_tmp.copy().fillna(fillna).round()

            col_tmp = col_tmp.astype(type)
            max_loss = (col_tmp - col).abs().max()
            avg_loss = (col_tmp - col).abs().mean()
            na_loss = np.abs(na_count - col_tmp.isna().sum())
            n_uniq_loss = np.abs(n_uniq - col_tmp.nunique(dropna=False))

            if max_loss <= max_loss_limit and avg_loss <= avg_loss_limit and na_loss <= na_loss_limit and n_uniq_loss <= n_uniq_loss_limit:
                return col_tmp

        # field can't be converted
        return col


def reduce_mem_usage_sd(df, deep=True, verbose=False, obj_to_cat=False):
    numerics = ['int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=deep).sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes

        # collect stats
        na_count = df[col].isna().sum()
        n_uniq = df[col].nunique(dropna=False)

        # numerics
        if col_type in numerics:
            df[col] = sd(df[col])

        # strings
        if (col_type == 'object') and obj_to_cat:
            df[col] = df[col].astype('category')

        if verbose:
            print(f'Column {col}: {col_type} -> {df[col].dtypes}, na_count={na_count}, n_uniq={n_uniq}')
        new_na_count = df[col].isna().sum()
        if (na_count != new_na_count):
            print(f'Warning: column {col}, {col_type} -> {df[col].dtypes} lost na values. Before: {na_count}, after: {new_na_count}')
        new_n_uniq = df[col].nunique(dropna=False)
        if (n_uniq != new_n_uniq):
            print(f'Warning: column {col}, {col_type} -> {df[col].dtypes} lost unique values. Before: {n_uniq}, after: {new_n_uniq}')

    end_mem = df.memory_usage(deep=deep).sum() / 1024 ** 2
    percent = 100 * (start_mem - end_mem) / start_mem
    if verbose:
        print('Mem. usage decreased from {:5.2f} Mb to {:5.2f} Mb ({:.1f}% reduction)'.format(start_mem, end_mem, percent))
    return df

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
