import pandas as pd
import numpy as np 

def left_merge(data1, data2, on):
    if type(on) != list:
        on = [on]
    if (set(on) & set(data2.columns)) != set(on):
        data2_temp = data2.reset_index()
    else:
        data2_temp = data2.copy()
    columns = [f for f in data2.columns if f not in on]
    result = data1.merge(data2_temp,on=on,how='left')
    result = result[columns]
    return result


def concat(L):
    """
    tools for concat new dataframe
    """
    result = None
    for l in L:
        if l is None:
            continue
        if result is None:
            result = l
        else:
            try:
                result[l.columns.tolist()] = l
            except Exception as err:
                print(err)
                print(l.head())
    return result

def name2feature(df, feature_space):
    assert isinstance(feature_space, list)
    '''get default parameters, parse dict op, remove delecol'''
    for key in feature_space:
        if key.startswith('COUNT'):
            '''assert value is [c1,c2,c3,c4]'''
            i = key.split('_')[-1]
            df = count_encode(df, i)
        elif key.startswith('CROSSCOUNT'):
            '''assert value is [[c1,c2,c3],[c4,c5,c6]]'''
            i , j = key.split('_')[-2:]
            df = cross_count_encode(df, [i, j])
        elif key.startswith('AGG'):
            '''assert value is [[n1,n2,n3],[c1,c2,c3]]'''
            stat, i, j =  key.split('_')[-3:]
            df = agg_encode(df, i, j, [stat])
    return df
    
def count_encode(df, col):
    """
    tools for count encode
    """
    df['count_{}'.format(col)] = df.groupby(col)[col].transform('count')
    return df


def cross_count_encode(df, col_list):
    """
    tools for multy thread bi_count
    """
    assert isinstance(col_list, list)
    assert len(col_list) >= 2
    name = "count_"+ '_'.join(col_list)
    df[name] = df.groupby(col_list)[col_list[0]].transform('count')
    return df


def agg_encode(df, num_col, col, stat_list = ['min', 'max', 'mean', 'median', 'var']):
    agg_dict = {}
    for i in stat_list:
        agg_dict['AGG_{}_{}_{}'.format(i, num_col, col)] = i
    agg_result = df.groupby([col])[num_col].agg(agg_dict)
    r = left_merge(df, agg_result, on = [col])
    df = concat([df, r])
    return df