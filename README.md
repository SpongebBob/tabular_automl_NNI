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
TODO
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
TODO
```


**4)  Send result and key information to tunner**

Use `nni.report_final_result` to send final result to Tuner. Please noted **15** line in the following code.

```python
TODO
```

**4)  Define your own feature engineer method **

If you want to add a feature engineer operation, you should follow this instruction.
Firstly, add json2space code in the tunner. 
```python
...
if key == 'opname':
    # give a fixed formax opname_colname, make sure "_" is not in column name.
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

