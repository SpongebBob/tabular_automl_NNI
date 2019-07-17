# How to write a Trial and Tunner for tabular data running on NNI?
*Tabular data is an arrangement of data in rows and columns, or possibly in a more complex structure. Usualy we treat columns as features, rows as data. AutoML for tabular data automates feature engineering  feature selection, and hyper tunning on a wide range of tabular data primitives — such as numbers, catgories, multi-catgories, timestamps etc.*

In this example, it shows that how a simple automl freamwork working on nni.

Trial receives the generated and selected feature configure from Tuner, and send intermediate result to Assessor and some key inofrmation to Tuner, such as metrics, feature importance information.

Tunner receives the key inofrmation to Tuner, such as metrics, feature importance information from trails.
then decide what feature to be generated and selected in next step.

So when user want to write a tabular automl tool running on NNI, she/he should:

**1)Have an original Trial could run**,

Trial's code could be any machine learning code that could run in local. He/She need to parse one fixed format feature name in order to generated a feature. Here we use `main.py` as example:

```python
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
    result = name2feature(df, ["AGG_min_I9_C3", "COUNT_C20", "CROSSCOUNT_C1_C11"])
    feature_imp, val_score = lgb_model_train(result,  _epoch = 1000, target_name = 'Label', id_index = 'Id')
    print(feature_imp)
    print(val_score)
    exit()
    

if __name__ == '__main__':
    file_name = 'train.tiny.csv'
    target_name = 'Label'
    id_index = 'Id'
    try:
        # get parameters from tuner
        RECEIVED_PARAMS = nni.get_next_parameter()
        print(RECEIVED_PARAMS)
        # list is a column_name generate from tunner
        df = pd.read_csv(file_name)
        if 'sample_feature' in RECEIVED_PARAMS.keys():
            sample_col = RECEIVED_PARAMS['sample_feature']
        else:
            sample_col = []
        df = name2feature(df, sample_col)
        LOG.debug(RECEIVED_PARAMS)
        feature_imp, val_score = lgb_model_train(df,  _epoch = 1000, target_name = target_name, id_index = id_index)
        nni.report_final_result({
            "default":val_score , 
            "feature_importance":feature_imp
        })
    except Exception as exception:
        LOG.exception(exception)
        raise

```

**2)Give an search space**,

Search space is defined by json, following fomart: 
```json
{
    "1-order-op" : [col1, col2],
    "2-order-op" : [col1, col2], [col3, col4]
}
```
Tunner receives this seach space, and generats the feature space. Every trial selected original feature and some generated feature. 


**3)Get configure from Tuner**

User import `nni` and use `nni.get_next_parameter()` to receive configure. 

```python
...
if 'sample_feature' in RECEIVED_PARAMS.keys():
            sample_col = RECEIVED_PARAMS['sample_feature']
        else:
            sample_col = []
        # raw feaure + sample_feature
        df = name2feature(df, sample_col)
...
```


**4)  Send result and key information to tunner**

Use `nni.report_final_result` to send final result to Tuner. Please noted **15** line in the following code.

```python

feature_imp, val_score = lgb_model_train(df,  _epoch = 1000, target_name = target_name, id_index = id_index)
nni.report_final_result({
    "default":val_score , 
    "feature_importance":feature_imp
})
```

**4)  Define your own feature engineer method**

If you want to add a feature engineer operation, you should follow this instruction.
Firstly, add json2space code in the tunner. 
```python
...
if key == 'opname':
    # give a fixed formax opname_colname, make sure that "_" is not in column name.
    name = 'opname_{}'.format(colname)
result.apped(name)
...	
```
Seconly, add name2feature code in the trail.
```python
...
if gen_name.startwith('opname'):
    col = parse(gen_name) #get the operated col name, such as count_col1 return col1
    df[gen_name] = df[col].apply(lambda x: fe_opname(x))
...
```

