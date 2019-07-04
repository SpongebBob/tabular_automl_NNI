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


class CustomerTuner(Tuner):
    def __init__(self, optimize_mode, feature_percent = 0.6):
        self.count = -1
        self.optimize_mode = OptimizeMode(optimize_mode)
        self.search_space = None
        self.feature_percent = feature_percent
        self.default_space = []
        logger.debug('init aufo-fe done.')
        return


    def generate_parameters(self, parameter_id, **kwargs):
        """Returns a set of trial graph config, as a serializable object.
        parameter_id : int
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
            sample_feature = np.random.choice(
                feature_list, 
                size = int(len(importance_df) * self.feature_percent), 
                p=feature_probablity, 
                replace = False
                )
            gen_feature = list(sample_feature)
            r = {'sample_feature': gen_feature, 'default_space': self.default_space}
            return r  


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

        #indiv = Individual(graph_loads(parameters), result=reward)
        #self.population.append(indiv)
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
        #self.search_space = data
        self.default_space = data


if __name__ =='__main__':
    tuner = CustomerTuner(OptimizeMode.Maximize)
    config = tuner.generate_parameters(0)
    with open('./data.json', 'w') as outfile:
        json.dump(config, outfile)
    tuner.receive_trial_result(0, config, 0.99)
