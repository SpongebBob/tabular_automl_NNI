# How to use NNI to do Automatic Feature Engeering?

## What is Tabular Data?

*Tabular data is an arrangement of data in rows and columns, or possibly in a more complex structure. Usually we treat columns as features, rows as data. AutoML for tabular data including automatic feature generation, feature selection, and hyper tunning on a wide range of tabular data primitives — such as numbers, categories, multi-categories, timestamps etc.*

## Quick Start

In this example, we will shows that how to do automatic feature engineering on nni.

We treat the automatic feature engineering(auto-fe) as a two steps task. *feature generation exploration* and *feature selection*.

We give a simple example.

The tuner call *AutoFETuner* first will generate a command that to ask *Trial* the *feature_importance* of original feature. *Trial* will return the *feature_importance* to *Tuner* in the first iteration. Then *AutoFETuner* will estimate a feature importance ranking and decide what feature to be generated, according to the definition of search space.

In the following iterations(2nd +), *AutoFETuner* updates the estimated feature importance ranking.

If you are interested in contributing to the *AutoFETuner* algorithm, such as Reinforcement Learning(RL) and genetic algorithm (GA),you are welcomed to propose proposal and pull request.  Interface `update_candidate_probility()` can be used to update feature sample probability and `epoch_importance` maintains the all iterations feature importance.

*Trial* receives the the configure contains selected feature configure from *Tuner*, then *Trial* will generate these feature by *fe_util*, which is a general sdk to generate features. After evaluate performance by adding these features, *Trial* will report the final metric to the Tuner.


So when user want to write a tabular autoML tool running on NNI, she/he should:

**1) Have an Trial code to run**

Trial's code could be any machine learning code. 
Here we use `main.py` as example:

```diff
import nni


if __name__ == '__main__':
    file_name = 'train.tiny.csv'
    target_name = 'Label'
    id_index = 'Id'

    # read original data from csv file
    df = pd.read_csv(file_name)

    # get parameters from tuner
+   RECEIVED_FEATURE_CANDIDATES = nni.get_next_parameter()

+    if 'sample_feature' in RECEIVED_FEATURE_CANDIDATES.keys():
+        sample_col = RECEIVED_FEATURE_CANDIDATES['sample_feature']
+    # return 'feature_importance' to tuner in first iteration
+    else:
+        sample_col = []
+    df = name2feature(df, sample_col)

    feature_imp, val_score = lgb_model_train(df,  _epoch = 1000, target_name = target_name, id_index = id_index)

+    # send final result to Tuner
+    nni.report_final_result({
+        "default":val_score , 
+        "feature_importance":feature_imp
    })
```

**2) Define a search space**

Search space could be defined in a json file, format as following: 

```json
{
    "1-order-op" : [
            col1,
            col2
        ],
    "2-order-op" : [
        [
            col1,
            col2
        ], [
            col3, 
            col4
        ]
    ]
}
```
We provide `count encoding`,`target encoding`,`embedding encoding` for `1-order-op` examples.
We provide `cross count encoding`, `aggerate statistics(min max var mean median nunique)`, `histgram aggerate statistics` for `2-order-op` examples.
All operations above are classic feature enginner methods, and the detail in [here](./AutoFEOp.md). 

*Tuner* receives this search space, and generates the feature calling SDK *fe_util*.

For example, we want to search the features which is a frequency encoding (value count) features on columns name {col1, col2}, in the following way:

```json
{
    "COUNT" : [
        col1,
        col2
    ],
}
```

For example, we can define a cross frequency encoding (value count on cross dims) method on columns {col1, col2} × {col3, col4} in the following way:

```json
{
    "CROSSCOUNT" : [
        [
            col1,
            col2
        ],
        [
            col3,
            col4
        ],
    ]
}
```

**3) Get configure from Tuner**

User import `nni` and use `nni.get_next_parameter()` to receive configure. 

```python
...
RECEIVED_PARAMS = nni.get_next_parameter()
if 'sample_feature' in RECEIVED_PARAMS.keys():
            sample_col = RECEIVED_PARAMS['sample_feature']
else:
    sample_col = []
# raw_feature + sample_feature
df = name2feature(df, sample_col)
...
```


**4)  Send result metric and feature importance to tunner**

Use `nni.report_final_result` to send final result to Tuner. Please noted **15** line in the following code.

```python

feature_imp, val_score = lgb_model_train(df,  _epoch = 1000, target_name = target_name, id_index = id_index)
nni.report_final_result({
    "default":val_score , 
    "feature_importance":feature_imp
})
```

**4) Extend the SDK of feature engineer method**

If you want to add a feature engineer operation, you should follow the  instruction in [here](./AutoFEOp.md). 

# Benchmark

We test some binary-classfiaction benchmarks which from open-resource.

The experiment setting is given in the `./test_config/test_name/search_sapce.json` :

The baseline and the result as following:

|  Dataset   | baseline auc  | automl auc| dataset link| 
|  ----  | ----  | ----  | ----  |
| Cretio Tiny  | 0.7516 | 0.7760 |[data link](https://labs.criteo.com/category/dataset/) |
| titanic  | 0.8700 | 0.8867 |[data link](https://www.kaggle.com/c/titanic/data) |
| Heart |0.9178| 0.9501|[data link](http://archive.ics.uci.edu/ml/datasets/Statlog+%28Heart%29)|
| Cancer |0.7089 | 0.7846 | [data link](http://archive.ics.uci.edu/ml/datasets/Breast+Cancer)|
| Haberman |0.6568 | 0.6948 | [data link](http://archive.ics.uci.edu/ml/machine-learning-databases/haberman/)|

