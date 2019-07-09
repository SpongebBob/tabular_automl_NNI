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


def get_default_parameters(df, default_space, deletcol = [], sample_col = []):
    assert isinstance(default_space, dict)
    '''get default parameters, parse dict op, remove delecol'''
    topk = 25
    for key in default_space.keys():
        if key == 'count':
            '''assert value is [c1,c2,c3,c4]'''
            count_num = 0
            for i in default_space[key]:
                if 'count_{}'.format(i) in deletcol or count_num >= topk:
                    continue
                df = count_encode(df, i)
                if 'count_{}'.format(i) in deletcol:
                    continue
                count_num += 1
            
        elif key == 'bicount':
            '''assert value is [[c1,c2,c3],[c4,c5,c6]]'''
            cross_count_num = 0
            for i in default_space[key][0]:
                for j in default_space[key][1]:
                    if "count_"+ '_'.join([i,j]) in deletcol or cross_count_num >= topk:
                        continue
                    df = cross_count_encode(df, [i, j])
                    if  "count_"+ '_'.join([i,j]) in sample_col:
                        continue
                    cross_count_num += 1
        elif key == 'aggregate':
            '''assert value is [[n1,n2,n3],[c1,c2,c3]]'''
            agg_num = 0
            for i in default_space[key][0]:
                for j in default_space[key][1]:
                    stat_list = []
                    sample_num = 0
                    for stat in ['min', 'max', 'mean', 'median', 'var']:
                        if 'AGG_{}_{}_{}'.format(stat, i, j) in deletcol or agg_num >= topk:
                            continue
                        if 'AGG_{}_{}_{}'.format(stat, i, j) in sample_col:
                            sample_num += 1
                        stat_list.append(stat)
                    if len(stat_list) <= 0:
                        continue
                    df = agg_encode(df, i, j)
                    agg_num += len(stat_list) - sample_num
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


def agg_encode(df, num_col, col, stat_list = ['min', 'max', 'mean', 'median', 'var']):
    agg_dict = {}
    for i in stat_list:
        agg_dict['AGG_{}_{}_{}'.format(i, num_col, col)] = i
    agg_result = df.groupby([col])[num_col].agg(agg_dict)
    r = left_merge(df, agg_result, on = [col])
    df = concat([df, r])
    return df