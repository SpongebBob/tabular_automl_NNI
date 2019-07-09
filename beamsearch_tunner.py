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

import copy
import json
import logging
import random
import numpy as np
from enum import Enum, unique

from nni.tuner import Tuner
from nni.utils import extract_scalar_reward

logger = logging.getLogger('autofe-tunner')



class OptimizeMode(Enum):
    Minimize = 'minimize'
    Maximize = 'maximize'


class BeamTuner(Tuner):
    def __init__(self, optimize_mode, feature_percent = 0.6, topk = 120):
        '''
        Beamsearch tunner used given default seacrch_space to get the top N features.
        '''
        self.count = -1
        self.optimize_mode = OptimizeMode(optimize_mode)
        # first trial feaure_importance
        self.search_space = None
        # deleted search_space
        self.deleta_feature = None
        # defautlt seach_space
        self.default_space = None 
        self.topk = topk
        self.feature_percent = feature_percent
        logger.debug('init aufo-fe done.')
        return
        

    def generate_parameters(self, parameter_id, **kwargs):
        """
        Generate used_features and deletad features.
        """
        self.count += 1
        if self.count == 0:
            return {'default_space': self.default_space}
        else:
            importance_df = self.search_space
            importance_df = importance_df[importance_df.feature_score !=0]
            feature_list = list(importance_df.feature_name)
            
            feature_probablity = list(importance_df.feature_score)
            # gen_parameter_from_distribution default number is sqrt 
            sample_size = min(128, int(len(importance_df) * self.feature_percent))
            sample_feature = np.random.choice(
                feature_list, 
                size=sample_size, 
                p=feature_probablity, 
                replace=False
                )
            gen_feature = list(sample_feature)
            self.deleta_feature = set([i for i in feature_list if i not in gen_feature])
            generate_result = {
                'sample_feature': gen_feature, 
                'default_space': self.default_space, 
                'default_importance': self.search_space,
                'delete_feature': self.deleta_feature
            }
            return generate_result  


    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        '''
        Record an observation of the objective function
        parameter_id : int
        parameters : dict of parameters
        value: final metrics of the trial, including reward
        '''
        # get the feature importance
        if self.search_space is None:
            self.search_space = value['feature_importance']
        reward = extract_scalar_reward(value)
        if self.optimize_mode is OptimizeMode.Minimize:
            reward = -reward

        logger.debug('receive trial result is:\n')
        logger.debug(str(parameters))
        logger.debug(str(reward))
        return

    def update_search_space(self, data):
        '''
        Input: data, search space object.
        {
            'op1' : [col1, col2, ....]
            'op2' : [col1, col2, ....]
            'op1_op2' : [col1, col2, ....]
        }
        '''
        self.default_space = data


if __name__ =='__main__':
    with open('search_space.json', 'r') as myfile:
        data=myfile.read()
    tuner = BeamTuner(OptimizeMode.Maximize)
    tuner.update_search_space(data)
    config = tuner.generate_parameters(0)
    with open('./data.json', 'w') as outfile:
        json.dump(config, outfile)
    tuner.receive_trial_result(0, config, 0.99)
