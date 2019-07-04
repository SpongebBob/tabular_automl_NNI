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


def get_default_parameters(df, RECEIVED_PARAMS):
    assert isinstance(RECEIVED_PARAMS, dict)
    '''get default parameters, parse dict op'''
    for key in RECEIVED_PARAMS.keys():
        if key == 'count':
            '''assert value is [c1,c2,c3,c4]'''
            for i in RECEIVED_PARAMS[key]:
                df = count_encode(df, i)
        elif key == 'bicount':
            '''assert value is [[c1,c2,c3],[c4,c5,c6]]'''
            for i in RECEIVED_PARAMS[key][0]:
                for j in RECEIVED_PARAMS[key][1]:
                    df = cross_count_encode(df, [i, j])
        elif key == 'aggregate':
            '''assert value is [[n1,n2,n3],[c1,c2,c3]]'''
            for i in RECEIVED_PARAMS[key][0]:
                for j in RECEIVED_PARAMS[key][1]:
                    df = agg_encode(df, i, j)
        else:
            raise RuntimeError('Not supported feature engeriner method!')
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


def agg_encode(df, num_col, col):
    agg_dict = {}
    agg_dict['AGG_min_{}_{}'.format(num_col, col)] = 'min'
    agg_dict['AGG_max_{}_{}'.format(num_col, col)] = 'max'
    agg_dict['AGG_mean_{}_{}'.format(num_col, col)] = 'mean'
    agg_dict['AGG_median_{}_{}'.format(num_col, col)] = 'median'
    agg_dict['AGG_var_{}_{}'.format(num_col, col)] = 'var'
    agg_result = df.groupby([col])[num_col].agg(agg_dict)
    r = left_merge(df, agg_result, on = [col])
    df = concat([df, r])
    return df