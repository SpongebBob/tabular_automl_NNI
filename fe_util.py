import pandas as pd
import numpy as np 
from sklearn.model_selection import KFold
from sklearn.decomposition import TruncatedSVD

def left_merge(data1, data2, on):
    """
    merge util for dataframe
    """
    if type(on) != list:
        on = [on]
    if (set(on) & set(data2.columns)) != set(on):
        data2_temp = data2.reset_index()
    else:
        data2_temp = data2.copy()
    columns = [f for f in data2.columns if f not in on]
    result = data1.merge(data2_temp, on = on, how='left')
    result = result[columns]
    return result


def concat(L):
    """
    tools for concat some dataframes into a new dataframe.
    """
    result = None
    for l in L:
        if l is None:
            continue
        if result is None:
            result = l
        else:
            result[l.columns.tolist()] = l
    return result

def name2feature(df, feature_space, target_name='label'):
    assert isinstance(feature_space, list)
    '''get default parameters, parse dict op, remove delecol'''
    for key in feature_space:
        if key.startswith('COUNT'):
            '''assert value is [c1,c2,c3,c4]'''
            i = key.split('_')[-1]
            df = count_encode(df, i)
        elif key.startswith('CROSSCOUNT'):
            '''assert value is [[c1,c2,c3],[c4,c5,c6]]'''
            i, j = key.split('_')[-2:]
            df = cross_count_encode(df, [i, j])
        elif key.startswith('AGG'):
            '''assert value is [[n1,n2,n3],[c1,c2,c3]]'''
            stat, i, j =  key.split('_')[-3:]
            df = agg_encode(df, i, j, [stat])
        elif key.startswith('NUNIQUE'):
            '''assert value is [[c1,c2,c3],[c4,c5,c6]]'''
            i, j =  key.split('_')[-2:]
            df = agg_nunique_encode(df, i, j)
        elif key.startswith('HISTSTAT'):
            '''assert value is [[c1,c2,c3],[c4,c5,c6]]'''
            i, j =  key.split('_')[-2:]
            df = agg_histgram_encode(df, i, j)
        elif key.startswith('TARGET'):
            '''assert value is [c1,c2,c3,c4]'''
            i = key.split('_')[-1]
            df = target_encoding(df, i, target_name)
        elif key.startswith('EMBEDDING'):
            '''assert value is [m1]'''
            i = key.split('_')[-1]
            df = embedding_encode(df, i)
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

def agg_nunique_encode(df, id_col, col):
    """
    get id group_by(id) nunique
    """
    agg_dict = {}
    agg_dict['NUNIQUE_{}_{}'.format(id_col, col)] = 'nunique'
    agg_result = df.groupby([col])[id_col].agg(agg_dict)
    r = left_merge(df, agg_result, on = [col])
    df = concat([df, r])
    return df

def agg_histgram_encode(df, id_col, col, stat_list = ['min', 'max', 'mean', 'median', 'var']):
    """
    get id group_by(id) histgram statitics
    """
    agg_dict = {}
    for i in stat_list:
        agg_dict['HISTSTAT_{}_{}_{}'.format(i, id_col, col)] = i
    df['temp_count'] = df.groupby(id_col)[id_col].transform('count')
    agg_result = df.groupby([col])['temp_count'].agg(agg_dict)
    r = left_merge(df, agg_result, on = [col])
    df = concat([df, r])
    del df['temp_count']
    return df


def base_embedding(x, model, size):
    """
    embedding helper for bagofwords
    """
    vec = np.zeros(size)
    x = [item for item in x if model.wv.__contains__(item)]
    for item in x:
        vec += model.wv[str(item)]
    if len(x) == 0:
        return vec
    else:
        return vec / len(x)


def embedding_encode(df, col):
    """
    This is the tool for multi-categories embedding encode.
    embedding for one single multi-categories column.
    """
    from gensim.models.word2vec import Word2Vec
    input_ = df[col].fillna('NA').apply(lambda x: str(x).split(' '))
    model = Word2Vec(input_, size=12, min_count=2, iter=5, window=5, workers=4)
    data_vec = []
    for row in input_:
        data_vec.append(base_embedding(row, model, size=12))
    svdT = TruncatedSVD(n_components=6)
    data_vec = svdT.fit_transform(data_vec)
    column_names = []
    for i in range(6):
        column_names.append('embedding_{}_{}'.format(col, i))
    data_vec = pd.DataFrame(data_vec, columns=column_names)
    df = pd.concat([df, data_vec], axis=1)
    return df

def add_noise(series, noise_level):
    """
    target encoding smooth
    """
    return series * (1 + noise_level * np.random.randn(len(series)))

def add_smooth(series, p, a = 1):
    """
    target encoding smooth
    """
    return (series.sum() + p / series.count() + a)

def target_encoding(df, col, target_name='label'):
    """
    target encoding  using 5 k-fold with smooth
    """
    df[col] = df[col].fillna('-9999999')
    mean_of_target = df[target_name].mean()

    kf = KFold(n_splits = 5, shuffle = True, random_state=2019) 
    col_mean_name = "target_{}".format(col)
    X = df[df[target_name].isnull() == False].reset_index(drop=True)
    X_te = df[df[target_name].isnull()].reset_index(drop=True)
    X.loc[:, col_mean_name] = np.nan
    
    for tr_ind, val_ind in kf.split(X):
        X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]
        X.loc[df.index[val_ind], col_mean_name] = X_val[col].map(X_tr.groupby(col)[target_name].apply(lambda x: add_smooth(x, 0.5, 1)))

    tr_agg =  X[[col, target_name]].groupby([col])[target_name].apply(lambda x: add_smooth(x, 0.5, 1)).reset_index()#['label'].mean().reset_index()
    tr_agg.columns = [col, col_mean_name]

    X_te = X_te.merge(tr_agg, on = [col], how = 'left')
    _s = np.array(pd.concat([X[col_mean_name], X_te[col_mean_name]]).fillna(mean_of_target))
    df[col_mean_name] =  _s
    return df

