# How to use the Op from SDK for feature engineering?

We offer many operations which could automaticlly generate new features, which list as following:

`1-order-op` : 
`count`, `target encoding`, `embedding encoding`

`2-order-op` :
`cross count encoding`, `aggerate statistics(min max var mean median nunique)` , `histgram aggerate statistics`

## Operation Definition

### count
@mengjiao could you give a definition here? also what's kind of column could use it? such as "CAT" or "TIME"?

### target
@mengjiao same here.

### embedding
@mengjiao same here.

### crosscount
@mengjiao same here.

### aggregate
@mengjiao same here.

### nunique
@mengjiao same here.

### histstat
@mengjiao same here.

## How to use it?



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