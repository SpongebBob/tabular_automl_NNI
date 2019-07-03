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

import nni
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import logging
import numpy as np
import pandas as pd
from fe_util import *


LOG = logging.getLogger('sklearn_classification')

def load_data():
    '''Load dataset, use 20newsgroups dataset'''
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=99, test_size=0.25)

    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    return X_train, X_test, y_train, y_test



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

def unit_test():
    return 


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    
    try:
        # get parameters from tuner
        RECEIVED_PARAMS = nni.get_next_parameter()
        # list is a column_name generate from tunner
        if isinstance(RECEIVED_PARAMS, list):
            use_col = RECEIVED_PARAMS
        else:
            # do feature genereation here.
            PARAMS = get_default_parameters(X_train, RECEIVED_PARAMS)
            
        LOG.debug(RECEIVED_PARAMS)
        PARAMS.update(RECEIVED_PARAMS)
        LOG.debug(PARAMS) 
        feature_imp = pd.DataFrame({
            'feature_name':['col1','col2','col3'],
            'feature_score':['0.5','0.3','0.2'],
        })
        nni.report_final_result({
            "default":0.555, 
            "feature_importance":feature_imp
        })

        #model = get_model(PARAMS)
        #run(X_train, X_test, y_train, y_test, model)
    except Exception as exception:
        LOG.exception(exception)
        raise
