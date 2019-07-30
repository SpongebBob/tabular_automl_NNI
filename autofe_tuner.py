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
from nni.utils import extract_scalar_reward, OptimizeMode

from const import FeatureType, AGGREGATE_TYPE

logger = logging.getLogger('autofe-tuner')


class AutoFETuner(Tuner):
    def __init__(self, optimize_mode = 'maximize', feature_percent = 0.6):
        """Initlization function
        count : 
        optimize_mode : contains "Maximize" or "Minimize" mode.
        search_space : define which features that tuner need to search
        feature_percent : @mengjiao
        default_space : @mengjiao 
        epoch_importance : @mengjiao
        estimate_sample_prob : @mengjiao
        """
        self.count = -1
        self.optimize_mode = OptimizeMode(optimize_mode)
        self.search_space = None
        self.feature_percent = feature_percent
        self.default_space = []
        self.epoch_importance = []
        self.estimate_sample_prob = None

        logger.debug('init aufo-fe done.')


    def generate_parameters(self, parameter_id, **kwargs):
        """Returns a set of trial graph config, as a serializable object.
        parameter_id : int
        """
        self.count += 1
        if self.count == 0:
            return {'sample_feature': []}
        else:
            sample_p = np.array(self.estimate_sample_prob) / np.sum(self.estimate_sample_prob)
            sample_size = min(128, int(len(self.candidate_feature) * self.feature_percent))
            sample_feature = np.random.choice(
                self.candidate_feature, 
                size = sample_size, 
                p = sample_p, 
                replace = False
                )
            gen_feature = list(sample_feature)
            r = {'sample_feature': gen_feature}
            return r  


    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        '''
        Record an observation of the objective function
        parameter_id : int
        parameters : dict of parameters
        value: final metrics of the trial
        '''
        # get the default feature importance

        if self.search_space is None:
            self.search_space = value['feature_importance']
            self.estimate_sample_prob = self.estimate_candidate_probility()
        else:
            self.epoch_importance.append(value['feature_importance'])
            # TODO
            self.update_candidate_probility()
        reward = extract_scalar_reward(value)
        if self.optimize_mode is OptimizeMode.Minimize:
            reward = -reward

        logger.info('receive trial result is:\n')
        logger.info(str(parameters))
        logger.info(str(reward))
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
        self.candidate_feature = self.json2space(data)


    def update_candidate_probility(self):
        """
        Using true_imp score to modify candidate probility.
        """
        # get last importance
        last_epoch_importance = self.epoch_importance[-1]
        last_sample_feature = list(last_epoch_importance.feature_name)
        for index, f in enumerate(self.candidate_feature):
            if f in last_sample_feature:
                score = max(float(last_epoch_importance[last_epoch_importance.feature_name == f]['feature_score']), 0.00001)
                self.estimate_sample_prob[index] = score
        
        logger.debug("Debug UPDATE ", self.estimate_sample_prob)


    def estimate_candidate_probility(self):
        """
        estimate_candidate_probility use history feature importance, first run importance.
        """
        raw_score_dict = self.impdf2dict()
        logger.debug("DEBUG feature importance\n", raw_score_dict)

        gen_prob = []
        for i in self.candidate_feature:
            _feature = i.split('_')
            score = [raw_score_dict[i] for i in _feature if i in raw_score_dict.keys()]
            if len(score) == 1:
                gen_prob.append(np.mean(score))
            else:
                generate_score = np.mean(score) * 0.9 # TODO
                gen_prob.append(generate_score)
        return gen_prob


    def impdf2dict(self):
        return dict([(i,j) for i,j in zip(self.search_space.feature_name, self.search_space.feature_score)])


    def json2space(self, default_space):
        """
        parse json to search_space 
        """
        result = []
        for key in default_space.keys():
            if key == FeatureType.COUNT:
                for i in default_space[key]:
                    name = (FeatureType.COUNT + '_{}').format(i)
                    result.append(name)         
            
            elif key == FeatureType.CROSSCOUNT:
                for i in default_space[key][0]:
                    for j in default_space[key][1]:
                        if i == j:
                            continue
                        cross = [i,j] 
                        cross.sort()
                        name = (FeatureType.CROSSCOUNT + '_') + '_'.join(cross)
                        result.append(name)         
                        
            
            elif key == FeatureType.AGGREGATE:
                for i in default_space[key][0]:
                    for j in default_space[key][1]:
                        for stat in AGGREGATE_TYPE:
                            name = (FeatureType.AGGREGATE + '_{}_{}_{}').format(stat, i, j)
                            result.append(name)
            
            elif key == FeatureType.NUNIQUE:
                for i in default_space[key][0]:
                    for j in default_space[key][1]:
                        name = (FeatureType.NUNIQUE + '_{}_{}').format(i, j)
                        result.append(name)
            
            elif key == FeatureType.HISTSTAT:
                for i in default_space[key][0]:
                    for j in default_space[key][1]:
                        name = (FeatureType.HISTSTAT + '_{}_{}').format(i, j)
                        result.append(name)
            
            elif key == FeatureType.TARGET:
                for i in default_space[key]:
                    name = (FeatureType.TARGET + '_{}').format(i)
                    result.append(name) 
            
            elif key == FeatureType.EMBEDDING:
                for i in default_space[key]:
                    name = (FeatureType.EMBEDDING + '_{}').format(i)
                    result.append(name) 
            
            else:
                raise RuntimeError('feature ' + str(key) + ' Not supported now')
        return result