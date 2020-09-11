# utility functions for feature engineering
import numpy as np
import pandas as pd
import os
import gc


def reduce_mem_usage(df, verbose=True):
    # reduce dataframe size
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in reduce_cols:
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
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
            start_mem - end_mem) / start_mem))
    return df


def freq_encode(train_data, test_data, columns):
    # Returns a DataFrame with encoded columns
    encoded_cols = []
    nsamples = train_data.shape[0]
    for col in columns:
        freqs_cat = train_data.groupby(col)[col].count() / nsamples
        encoded_col_train = train_data[col].map(freqs_cat)
        encoded_col_test = test_data[col].map(freqs_cat)
        encoded_col = pd.concat([encoded_col_train, encoded_col_test], axis=0)
        encoded_col[encoded_col.isnull()] = 0
        encoded_cols.append(pd.DataFrame({'freq_' + col: encoded_col}))
    all_encoded = pd.concat(encoded_cols, axis=1)
    print(all_encoded.shape)
    return (all_encoded.iloc[:train_data.shape[0], :],
            all_encoded.iloc[train_data.shape[0]:, :])


def do_countuniq(df, group_cols, counted, show_max=False, show_agg=True):
    # count unique values
    agg_name = '{}_by_{}_countuniq'.format(('_'.join(group_cols)), (counted))
    if show_agg:
        print("\nCounting unqiue ", counted, " by ", group_cols, '... and saved in', agg_name)
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].nunique().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print(agg_name + " max value = ", df[agg_name].max())

    # df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return (df)


def timeblock_frequency_encoding(train_df, test_df, periods, columns,
                                 with_proportions=False, only_proportions=False):
    # count frequency by time period
    for period in periods:
        for col in columns:
            new_col = col + '_' + period
            train_df[new_col] = train_df[col].astype(str) + '_' + train_df[period].astype(str)
            test_df[new_col] = test_df[col].astype(str) + '_' + test_df[period].astype(str)

            temp_df = pd.concat([train_df[[new_col]], test_df[[new_col]]])
            fq_encode = temp_df[new_col].value_counts().to_dict()

            train_df[new_col] = train_df[new_col].map(fq_encode)
            test_df[new_col] = test_df[new_col].map(fq_encode)

            if only_proportions:
                train_df[new_col] = train_df[new_col] / train_df[period + '_total']
                test_df[new_col] = test_df[new_col] / test_df[period + '_total']

            if with_proportions:
                train_df[new_col + '_proportions'] = train_df[new_col] / train_df[period + '_total']
                test_df[new_col + '_proportions'] = test_df[new_col] / test_df[period + '_total']

    return train_df, test_df


if __name__ == '__main__':
    pass
