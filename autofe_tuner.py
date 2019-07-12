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
from itertools import combinations

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
        self.epoch_importance = []
        logger.debug('init aufo-fe done.')
        return


    def generate_parameters(self, parameter_id, **kwargs):
        """Returns a set of trial graph config, as a serializable object.
        parameter_id : int
        """
        self.count += 1
        if self.count == 0:
            return {'sample_feature': []}
        else:           
            sample_size = min(128, int(len(self.candidate_feature) * self.feature_percent))
            print("candiata", self.candidate_feature)
            sample_feature = np.random.choice(
                self.candidate_feature, 
                size = sample_size, 
                p = self.estimate_sample_prob, 
                replace = False
                )
            gen_feature = list(sample_feature)
            print(gen_feature)
            r = {'sample_feature': gen_feature}
            return r  


    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        '''
        Record an observation of the objective function
        parameter_id : int
        parameters : dict of parameters
        value: final metrics of the trial, including reward
        '''
        # get the default feature importance
        if self.search_space is None:
            self.search_space = value['feature_importance']
            self.estimate_sample_prob = self.estimate_candidate_probility()
        else:
            self.epoch_importance = self.epoch_importance.append(value['feature_importance'])
            # TODO
            self.update_candidate_probility()
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
        #self.search_space = data
        self.default_space = data
        self.candidate_feature = self.json2space(data)


    def update_candidate_probility(self):
        """
        Using true_imp score to modify candidate probility.
        """
        # get last importance
        last_epoch_importance = self.epoch_importance[-1]
        last_sample_feature = list(last_epoch_importance.feature_name)
        last_sample_feature_score = list(last_epoch_importance.score)
        #self.
        return 

    def estimate_candidate_probility(self):
        """
        estimate_candidate_probility use history feature importance, first run importance.
        """
        raw_score_dict = self.impdf2dict()
        gen_prob = []
        for i in self.candidate_feature:
            _feature = i.split('_')
            score = [raw_score_dict[i] for i in _feature if i in raw_score_dict.keys()]
            if len(_feature) == 1:
                gen_prob.append(np.mean(score))
            else:
                generate_score = np.mean(score) * 0.9 # TODO
                gen_prob.append(generate_score)
        return gen_prob
 
    
    def impdf2dict(self):
        d= dict([(i,j) for i,j in zip(self.search_space.feature_name, self.search_space.feature_score)])
        return d 

    def json2space(self, default_space):
        """
        You Need to add name format and parse mthod in fe_util.py,
        if you want to add more feature generated methods.
        """
        result = []
        for key in default_space.keys():
            if key == 'count':
                for i in default_space[key]:
                    name = 'COUNT_{}'.format(i)
                    result.append(name)         
            elif key == 'crosscount':
                for i in default_space[key][0]:
                    for j in default_space[key][1]:
                        if i == j:
                            continue
                        cross = [i,j] 
                        cross.sort()
                        name = "CROSSCOUNT_"+ '_'.join(cross)
            elif key == 'aggregate':
                for i in default_space[key][0]:
                    for j in default_space[key][1]:
                        for stat in ['min', 'max', 'mean', 'median', 'var']:
                            name = 'AGG_{}_{}_{}'.format(stat, i, j)
                            result.append(name)
            else:
                raise RuntimeError('Not supported feature engeriner method!')
        return result

# if __name__ =='__main__':
#     with open('search_space.json', 'r') as myfile:
#         data=myfile.read()
#     data = json.loads(data)
#     tuner = CustomerTuner(OptimizeMode.Maximize)
#     tuner.update_search_space(data)
#     #print(tuner.candidate_feature)
#     config = tuner.generate_parameters(0)
#     #print(config)
#     with open('./data.json', 'w') as outfile:
#         json.dump(config, outfile)
#     tuner.receive_trial_result(0, config, 0.99)