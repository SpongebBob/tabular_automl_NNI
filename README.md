# How to use NNI to do Automatic Feature Engineering?

## What is Tabular Data?

*Tabular data is an arrangement of data in rows and columns, or possibly in a more complex structure. Usually, we treat columns as features, rows as data. AutoML for tabular data including automatic feature generation, feature selection, and hyper tunning on a wide range of tabular data primitives — such as numbers, categories, multi-categories, timestamps, etc.*

## Quick Start

In this example, we will show how to do automatic feature engineering on nni.

We treat the automatic feature engineering(auto-fe) as a two steps task. *feature generation exploration* and *feature selection*.

We give a simple example.

The tuner call *AutoFETuner* first will generate a command that to ask *Trial* the *feature_importance* of original feature. *Trial* will return the *feature_importance* to *Tuner* in the first iteration. Then *AutoFETuner* will estimate a feature importance ranking and decide what feature to be generated, according to the definition of search space.

In the following iterations, *AutoFETuner* updates the estimated feature importance ranking.

If you are interested in contributing to the *AutoFETuner* algorithm, such as Reinforcement Learning(RL) and genetic algorithm (GA), you are welcomed to propose proposal and pull request.  Interface `update_candidate_probility()` can be used to update feature sample probability and `epoch_importance` maintains the all iterations feature importance.

*Trial* receives the configure contains selected feature configure from *Tuner*, then *Trial* will generate these feature by *fe_util*, which is a general SDK to generate features. After evaluating performance by adding these features, *Trial* will report the final metric to the Tuner.


So when user wants to write a tabular autoML tool running on NNI, she/he should:

**1) Have a Trial code to run**

Trial's code could be any machine learning code. 
Here we use `main.py` as an example:

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

Search space could be defined in a JSON file, format as following: 

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
We provide `count encoding`, `target encoding`, `embedding encoding` for `1-order-op`.
We provide `cross count encoding`, `aggerate statistics(min max var mean median nunique)`, `histgram aggerate statistics` for `2-order-op`.
All operations above are classic feature engineer methods, and the detail in [here](./AutoFEOp.md). 

*Tuner* receives this search space and generates the feature by calling generator in *fe_util*.

For example, we want to search the features which are a frequency encoding (value count) features on columns name {col1, col2}, in the following way:

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


**4)  Send final metric and feature importances to tuner**

Use `nni.report_final_result` to send final result to Tuner. Please noted **15** line in the following code.

```python

feature_imp, val_score = lgb_model_train(df,  _epoch = 1000, target_name = target_name, id_index = id_index)
nni.report_final_result({
    "default":val_score , 
    "feature_importance":feature_imp
})
```

**5) Extend the SDK of feature engineer method**

If you want to add a feature engineer operation, you should follow the instruction in [here](./AutoFEOp.md). 

**6) Run expeirment**

```
nnictl create --config config.yml
```

# Test Example

We test some binary-classification benchmarks which come from public resources.

The experiment setting is given in the `./benchmark/benchmark_name/search_sapce.json` :

The baseline and the result as following:


|  Dataset   | baseline auc  | automl auc| number of cat|  number of num |  dataset link| 
|  ----  | ----  | ----  | ----  |  ----  | ----  | 
| Cretio| 0.7516 | 0.7760 | 13 | 26| [data link](https://labs.criteo.com/category/dataset/) |
| titanic  | 0.8700 | 0.8867 | 9 | 1 |  [data link](https://www.kaggle.com/c/titanic/data) |
| Heart |0.9178| 0.9501| 4 | 9|  [data link](http://archive.ics.uci.edu/ml/datasets/Statlog+%28Heart%29)|
| Cancer |0.7089 | 0.7846 |9 | 0|  [data link](http://archive.ics.uci.edu/ml/datasets/Breast+Cancer)|
| Haberman |0.6568 | 0.6948 | 2 | 1|   [data link](http://archive.ics.uci.edu/ml/machine-learning-databases/haberman/)|

