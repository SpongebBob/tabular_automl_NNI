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