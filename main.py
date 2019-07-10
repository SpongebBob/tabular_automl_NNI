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
import json
from fe_util import *
from model import *

LOG = logging.getLogger('sklearn_classification')

def unit_test_fe():
    with open('search_space.json', 'r') as myfile:
        data=myfile.read()
    df = pd.read_csv('train.tiny.csv')
    json_config = json.loads(data)
    #result = get_default_parameters(df, json_config)
    result = name2feature(df, ["AGG_min_I9_C3", "COUNT_C20", "CROSSCOUNT_C1_C11"])
    feature_imp, val_score = lgb_model_train(result,  _epoch = 1000, target_name = 'Label', id_index = 'Id')
    print(feature_imp)
    print(val_score)
    exit()
    

if __name__ == '__main__':
    #unit_test_fe()
    file_name = 'train.tiny.csv'
    target_name = 'Label'
    id_index = 'Id'
    try:
        # get parameters from tuner
        RECEIVED_PARAMS = nni.get_next_parameter()
        # list is a column_name generate from tunner
        df = pd.read_csv(file_name)
        if 'delete_feature' in RECEIVED_PARAMS.keys():
            delete_col = RECEIVED_PARAMS['delete_feature']
        else:
            delete_col = []
        if 'sample_feature' in RECEIVED_PARAMS.keys():
            sample_col = RECEIVED_PARAMS['sample_feature']
        else:
            sample_col = []
        df = name2feature(df, sample_col)
        # df = get_default_parameters(df, RECEIVED_PARAMS['default_space'], delete_col, sample_col)
        # if 'sample_feature' in RECEIVED_PARAMS.keys():
        #     use_col = RECEIVED_PARAMS['sample_feature']
        #     use_col += [target_name]
        # else:
        #use_col = list(df.columns) + [target_name]
        print(df.columns)
        LOG.debug(RECEIVED_PARAMS)
        #PARAMS.update(RECEIVED_PARAMS)
        #LOG.debug(use_col) 
        # attention ID and other Lable need to be specifiy
        feature_imp, val_score = lgb_model_train(df,  _epoch = 1000, target_name = target_name, id_index = id_index)
        nni.report_final_result({
            "default":val_score , 
            "feature_importance":feature_imp
        })
    except Exception as exception:
        LOG.exception(exception)
        raise
