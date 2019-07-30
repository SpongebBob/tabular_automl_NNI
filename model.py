# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np 
import pandas as pd 
import lightgbm as lgb 
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, roc_curve,log_loss


def get_fea_importance(clf, feature_name):
    gain = clf.feature_importance('gain')
    split_= clf.feature_importance('split')
    importance_df = pd.DataFrame({
        'feature_name':clf.feature_name(),
        'split': split_,
        'gain': gain, 
        'gain_percent':100 *gain / gain.sum(),
        'split_percent':100 *split_ / split_.sum(),
        })
    importance_df['feature_score'] =  0.3*  importance_df['gain_percent'] + 0.7 * importance_df['split_percent'] 
    importance_df.loc[importance_df['split'] ==0, 'feature_score'] = 0
    importance_df['feature_score'] = importance_df['feature_score'] / importance_df['feature_score'].sum()
    importance_df = importance_df.sort_values('feature_score',ascending=False)
    return importance_df


def train_test_split(X, y, test_size, random_state=2018):
    sss = list(StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state).split(X, y))
    X_train = np.take(X, sss[0][0], axis=0)
    X_test = np.take(X, sss[0][1], axis=0)
    y_train = np.take(y, sss[0][0], axis=0)
    y_test = np.take(y, sss[0][1], axis=0)
    return [X_train, X_test, y_train, y_test]


def lgb_model_train( df, _epoch=1000, target_name='Label', id_index='Id'):
    df = df.loc[df[target_name].isnull()==False]
    feature_name = [i for i in df.columns if i not in [target_name, id_index]]
    for i in feature_name:
        if df[i].dtypes == 'object':
            if df[i].fillna('na').nunique() < 12:
                df.loc[:,i] = df.loc[:,i].fillna('na').astype('category')
            else:
                df.loc[:,i] = LabelEncoder().fit_transform(df.loc[:,i].fillna('na').astype(str))
    params_lgb = {
            "objective": "binary", 
            "metric":"auc", 
            'verbose': -1, 
            "seed": 1024, 
            'num_threads': 4,
            'num_leaves':64, 
            'learning_rate': 0.05,
            'min_data': 200, 
            'bagging_fraction': 0.5,
            'feature_fraction': 0.5,
            'max_depth': -1 ,
    }
    X_train, X_val, y_train, y_val = train_test_split(df[feature_name], df[target_name].values, 0.15, 1024)
    del df
    gc.collect()

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    #del X_train, X_val, y_train, y_val
    gc.collect()
    clf = lgb.train(
        params_lgb, lgb_train, valid_sets=lgb_val, valid_names='eval', 
        verbose_eval=50, early_stopping_rounds=100, num_boost_round=_epoch)

    fea_importance_now = get_fea_importance(clf, feature_name)
    val_auc = roc_auc_score(y_val,  clf.predict(X_val, num_iteration=clf.best_iteration))
    return fea_importance_now, val_auc