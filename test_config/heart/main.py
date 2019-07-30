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
import logging
import numpy as np
import pandas as pd
import json
import sys
from sklearn.preprocessing import LabelEncoder

sys.path.append('../../')

from fe_util import *
from model import *

logger = logging.getLogger('auto-fe-examples')

if __name__ == '__main__':
    file_name = '~/Downloads/heart.dat'
    target_name = 'Label'
    id_index = 'Id'

    # get parameters from tuner
    RECEIVED_PARAMS = nni.get_next_parameter()
    logger.info("Received params:\n", RECEIVED_PARAMS)
    
    # list is a column_name generate from tuner
    df = pd.read_csv(file_name, sep = ' ')
    df.columns = [
        'c1', 'c2', 'c3', 'n4', 'n5', 'n6', 'c7', 'n8',\
        "n9", "n10", "n11", "n12", "n13", 'Label'
    ]
    df['Label'] = df['Label'] -1 #LabelEncoder().fit_transform(df['Label'])
    
    if 'sample_feature' in RECEIVED_PARAMS.keys():
        sample_col = RECEIVED_PARAMS['sample_feature']
    else:
        sample_col = []
    
    # raw feaure + sample_feature
    df = name2feature(df, sample_col, target_name)
    feature_imp, val_score = lgb_model_train(df,  _epoch = 1000, target_name = target_name, id_index = id_index)
    nni.report_final_result({
        "default":val_score, 
        "feature_importance":feature_imp
    })

