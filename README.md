# How to write a Trial and Tuner for tabular data running on NNI?
*Tabular data is an arrangement of data in rows and columns, or possibly in a more complex structure. Usually we treat columns as features, rows as data. AutoML for tabular data automates feature engineering  feature selection, and hyper tunning on a wide range of tabular data primitives — such as numbers, categories, multi-categories, timestamps etc.*

In this example, it shows that how a simple autoML framework working on nni.

Trial receives the generated and selected feature configure from Tuner, and send intermediate result to Assessor and some key information to Tuner, such as metrics, feature importance information.

Tuner receives the key information to Tuner, such as metrics, feature importance information from trails.
then decide what feature to be generated and selected in next step.

So when user want to write a tabular autoML tool running on NNI, she/he should:

**1)Have an original Trial could run**,

Trial's code could be any machine learning code that could run in local. He/She need to parse one fixed format feature name in order to generated a feature. Here we use `main.py` as example:

```python
import nni


if __name__ == '__main__':
    file_name = 'train.tiny.csv'
    target_name = 'Label'
    id_index = 'Id'
    try:
        # get parameters from tuner
        RECEIVED_PARAMS = nni.get_next_parameter()
        print(RECEIVED_PARAMS)
        # list is a column_name generate from tuner
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
| Cretio  | 000 | 000 |
| talkingdata  | 000 | 000 |

