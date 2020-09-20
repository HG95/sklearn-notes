# sklearn.metrics.recall_score用法

**召回率recall**

```python
sklearn.metrics.precision_score(y_true, 
                                y_pred, 
                                labels=None, 
                                pos_label=1, 
                                average='binary', 
                                sample_weight=None, 
                                zero_division='warn'
                               )
```



**Parameters**:

- **y_true** :**1d array-like, or label indicator array / sparse matrix**
- **y_pred** :**1d array-like, or label indicator array / sparse matrix**
- **average** : 计算类型 **string, [None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]**
  average参数定义了该指标的计算方法，二分类时average参数默认是binary，多分类时，可选参数有micro、macro、weighted和samples。
- **sample_weight** : 样本权重

参数average

| **选项** |                 **含义**                 |
| :------: | :--------------------------------------: |
|  binary  |                  二分类                  |
|  micro   |           统计全局TP和FP来计算           |
|  macro   | 计算每个标签的未加权均值（不考虑不平衡） |
| weighted |  计算每个标签等等加权均值（考虑不平衡）  |
| samples  |          计算每个实例找出其均值          |
| **None** |             返回每类的精确度             |

<br>

**Returns**:

- **precision**:**float (if average is not None) or array of float, shape = [n_unique_labels]**

<br>

**Examples**:



```python
>>> from sklearn.metrics import recall_score
>>> y_true = [0, 1, 2, 0, 1, 2]
>>> y_pred = [0, 2, 1, 0, 0, 1]
>>> recall_score(y_true, y_pred, average='macro')
0.33...
>>> recall_score(y_true, y_pred, average='micro')
0.33...
>>> recall_score(y_true, y_pred, average='weighted')
0.33...
>>> recall_score(y_true, y_pred, average=None)
array([1., 0., 0.])
>>> y_true = [0, 0, 0, 0, 0, 0]
>>> recall_score(y_true, y_pred, average=None)
array([0.5, 0. , 0. ])
>>> recall_score(y_true, y_pred, average=None, zero_division=1)
array([0.5, 1. , 1. ])
```

