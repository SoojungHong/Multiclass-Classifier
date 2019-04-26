# libraries
import pandas as pd
from typing import Any, Union, Dict
from pandas import Series, DataFrame
from pandas.core.generic import NDFrame
import collections


# functions
def convert_to_date(unix_time):
    result_ms = pd.to_datetime(unix_time, unit='ms')
    str(result_ms)
    return result_ms


def is_columns_same(df, column1, column2):
    is_same = df[column1].equals(df[column2])
    return is_same


def count_string_frequency(input_all_strings):
    str_count: Dict[Any, int] = dict()
    for str in input_all_strings:
        if str in str_count:
            str_count[str] += 1
        else:
            str_count[str] = 1

    for key, value in str_count.items():
        print("% s : % d" % (key, value))

    return str_count


def n_most_common_in_series(series, n):
    d = collections.Counter(series)
    n_most_common = d.most_common(n)
    return n_most_common


def column_factorize(target_column):
    return pd.factorize(target_column)[0]


def sort_dataframe_with_column(df, column):
    return df.sort_values(column)


def get_all_queries_contains(query, df, query_column):
    contain_df = df[df[query_column].str.contains(query)]
    queries_contains = contain_df[query_column]
    return queries_contains  #queries_contains.unique()


def get_long_queries_contains(query, df, query_column):
    all_queries = df[query_column].unique()
    for i in range(len(all_queries)):
        if all_queries[i] == query:
            return query


def count_different_label(df, col_name):
    from collections import Counter
    print(Counter(df[col_name]))
