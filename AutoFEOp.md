# How to use the Op from SDK for feature engineering?

We offer many operations which could automatically generate new features, which list as following:

`1-order-op` : 
`count`, `target encoding`, `embedding encoding`

`2-order-op` :
`cross count encoding`, `aggregate statistics(min max var mean median nunique)` , `histgram aggregate statistics`

## Operation Definition

### count
Preformed on category columns(CAT).

Count encoding is based on replacing categories with their counts computed on the train set, also named frequency encoding.
[count encoding](https://wrosinski.github.io/fe_categorical_encoding/)

### target
Preformed on category columns(CAT).

Target encoding is based on encoding categorical variable values with mean of target variable per value. A statistic (here - mean) of target variable can be computed for every group in the train set and afterwards merged to validation and test sets to capture relationships between a group and the target.

When using target variable, is is very important not to leak any information into the validation set. Every such feature should be computed on the training set and then only merged or concatenated with the validation and test subsets. Even though target variable is present in the validation set, it cannot be used for any kind of such computation or overly optimistic estimate of validation error will be given.

If KFold is used, features based on target should be computed in-fold. If a single split is performed, then this should be done after splitting the data into train and validation set.

What is more, smoothing can be added to avoid setting certain categories to 0. Adding random noise is another way of avoiding possible overfit.

When done properly, target encoding is the best encoding for both linear and non-linear models.

[target encoding reference](https://wrosinski.github.io/fe_categorical_encoding/)

### embedding
Preformed on multi-category columns(Multi-CAT).

We can treat multi-category columns as natural language sentence, thus bag of words(BOW) can be utilized.

However it has no semantics and very sparse.

Thus word embedding is a good choice. First we train embedding and get the mean embedding for one row. Then use SVD for dimensionality reduction into 6 dims.

### crosscount
Preformed on several category columns(CAT).

Feature cross is important in some task such as Click Through Rate(CTR). 

Cross count is count encoding on more than one dimension.


### aggregate min/max/var/mean/
Preformed on one category column(CAT) and one numerical column(NUM). 

We group the data by the instances and then every instance is represented by only one row. The key point of group by operations is to decide the aggregation functions of the features. For numerical features, average, sum, min, max functions are usually convenient options, whereas for categorical features it more complicated.

### nunique (aggregate)
Preformed on one category column(CAT) and one category column(CAT).

For categorical features, nunique() functions are usually convenient options, return a group distinct observations.

### histstat (aggregate) 
Preformed on one category column(CAT) and one category column(CAT).

For categorical features, we can take average, sum, min, max aggregate functions on the histogram of group result.

## How to use it?
We just need to decide use which Op and the corresponding column name in search_space.json like:

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

# How to define an Op by yourself?



Firstly, add json2space code in the tuner. 
```python
...
if key == 'OP_NAME':
    # give a fixed format opname_colname, make sure that "_" is not in column name.
    name = 'OP_NAME_{}'.format(colname)
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

Thirdly, add `fe_opname` implement in `fe_util.py`
```python
...
def fe_opnam(df, col):
    # do some things to df[col]
    # DIY
    return df[col]
...
```