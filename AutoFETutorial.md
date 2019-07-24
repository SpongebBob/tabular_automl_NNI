# How to use NNI to do Automatic Feature Engeering?

## What is Tabular Data?

*Tabular data is an arrangement of data in rows and columns, or possibly in a more complex structure. Usually we treat columns as features, rows as data. AutoML for tabular data including automatic feature generation, feature selection, and hyper tunning on a wide range of tabular data primitives — such as numbers, categories, multi-categories, timestamps etc.*

## Quick Start

In this example, we will shows that how to do automatic feature engeering on nni.

The tuner call *AutoFETuner* first will generate a command that to ask *Trial* the *feature_importance* of original feature. *Trial* will return the *feature_importance* to *Tuner* in the first iteration. Then *AutoFETuner* will decide what feature to be generated, accroding to the definiton of search space.

*Trial* receives the the configure contains selected feature configure from *Tuner*, then *Trial* will generate these feature by *fe_util*, which is a general sdk to generate features. After evaluate performence by adding these features, *Trial* will report the final metric to the Tuner.


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

**2) Give an search space**

Search space is defined by json, following format: 
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
We provude `cross count encoding`, `aggerate statistics(min max var mean median nunique)`, `histgram aggerate statistics` for `2-order-op` examples.
ALL operations above are classic feature enginner methods. 

Tuner receives this search space, and generates the feature space. Every trial selected original feature and some generated feature. 

For example, we can define a frequency encoding (value count) method on columns {col1, col2} in the following way:
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


**3)Get configure from Tuner**

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
Firstly, add json2space code in the tuner. 
```python
...
if key == 'opname':
    # give a fixed format opname_colname, make sure that "_" is not in column name.
    name = 'opname_{}'.format(colname)
result.append(name)
...	
```
Seconly, add name2feature code in the trail.
```python
...
if gen_name.startwith('opname'):
    col = parse(gen_name) 
    #get the operated col name, such as count_col1 return col1
    df[gen_name] = df[col].apply(lambda x: fe_opname(x))
...
```

# Test example results on some binary classification dataset.
|  Dataset   | baseline result  | automl result| 
|  ----  | ----  | ----  |
| Cretio Tiny  | 0.7516 | 0.7760 |
| titanic  | 0.8700 | 0.8867 |
| talkingdata  | 000 | 000 |

