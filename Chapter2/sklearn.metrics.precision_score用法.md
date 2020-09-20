# sklearn.metrics.precision_score用法

**精确度 precision** :所有的测量点到测量点集合的均值非常接近，与测量点的方差有关。就是说各个点紧密的聚合在一起。

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

<br>

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



```python
>>> from sklearn.metrics import precision_score
>>> y_true = [0, 1, 2, 0, 1, 2]
>>> y_pred = [0, 2, 1, 0, 0, 1]

>>> precision_score(y_true, y_pred, average='macro')
0.22...

>>> precision_score(y_true, y_pred, average='micro')
0.33...

>>> precision_score(y_true, y_pred, average='weighted')
0.22...

>>> precision_score(y_true, y_pred, average=None)
array([0.66..., 0.        , 0.        ])

>>> y_pred = [0, 0, 0, 0, 0, 0]
>>> precision_score(y_true, y_pred, average=None)
array([0.33..., 0.        , 0.        ])

>>> precision_score(y_true, y_pred, average=None, zero_division=1)
array([0.33..., 1.        , 1.        ])
```



`micro`、`macro`、`weighted `以及样本不均时加入`sample_weight`参数的计算方法。

以三分类模型举例。首先我们生成一组数据：

```python
import numpy as np

y_true = np.array([-1]*30 + [0]*240 + [1]*30)
y_pred = np.array([-1]*10 + [0]*10 + [1]*10 + 
                  [-1]*40 + [0]*160 + [1]*40 + 
                  [-1]*5 + [0]*5 + [1]*20)
```

数据分为-1、0、1三类，真实数据y_true中，一共有30个-1，240个0，30个1。然后我们生成真实数据y_true和预测数据y_pred的混淆矩阵，之后的演示中我们会用到混淆矩阵的数据：

```python
confusion_matrix(y_true, y_pred)

#array([[ 10,  10,  10],
#       [ 40, 160,  40],
#       [  5,   5,  20]], dtype=int64)
```

由混淆矩阵我们可以计算出真正类数TP、假正类数FP、假负类数FN，如下：

|      |  TP  |  FN  |  FP  |
| :--: | :--: | :--: | :--: |
|  -1  |  10  |  20  |  45  |
|  0   | 160  |  80  |  15  |
|  1   |  20  |  10  |  50  |

以`precision_score`的计算为例，`accuracy_score`、`recall_score`、`f1_score`等均可以此类推。

sklearn包中计算`precision_score`

```
klearn.metrics.precision_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’, 
                                sample_weight=None)
```

其中，average参数定义了该指标的计算方法，二分类时average参数默认是binary，多分类时，可选参数有micro、macro、weighted和samples。samples的用法我也不是很明确，所以本文只讲解micro、macro、weighted。

<br>

## **1 不加`sample_weight`**

**1.1 micro**

`micro`算法是指把所有的类放在一起算，具体到`precision`，就是把所有类的 TP 加和，再除以所有类的 TP 和 FN 的加和。因此`micro`方法下的`precision`和`recall`都等于`accuracy`。

<center>
    <img src="https://raw.githubusercontent.com/HG1227/image/master/img_tuchuang/20200102155210.png"/>
</center>

**1.2 macro**

macro方法就是先分别求出每个类的precision再算术平均。

<center>
    <img src="https://raw.githubusercontent.com/HG1227/image/master/img_tuchuang/20200102155327.png"/>
</center>



**1.3 weighted**

前面提到的macro算法是取算术平均，weighted算法就是在macro算法的改良版，不再是取算术平均、乘以固定weight（也就是1/3）了，而是乘以该类在总样本数中的占比。计算一下每个类的占比：

```python
>>> w_neg1, w_0, w_pos1 = np.bincount(y_true+1) / len(y_true)
>>> print(w_neg1, w_0, w_pos1)
0.1 0.8 0.1
```

然后手算一下weighted方法下的precision：

<center>
    <img src="https://raw.githubusercontent.com/HG1227/image/master/img_tuchuang/20200102155548.png"/>
</center>



<br>

## **2 加入`sample weight`**

当样本不均衡时，比如本文举出的样本，中间的0占80%，1和-1各占10%，每个类数量差距很大，我们可以选择加入sample_weight来调整我们的样本。

首先我们使用sklearn里的`compute_sample_weight`函数来计算sample_weight：

```python
sw = compute_sample_weight(class_weight='balanced',y=y_true)
```

sw 是一个和 y_true 的 shape 相同的数据，每一个数代表该样本所在的 sample_weight。它的具体计算方法是 : 总样本数 /（类数 \* 每个类的个数），比如一个值为-1的样本，它的sample_weight就是300 / (3 \* 30)。



使用sample_weight计算出的混淆矩阵如下：

```python
>>> cm =confusion_matrix(y_true, y_pred, sample_weight=sw)
>>> cm
array([[33.33333333, 33.33333333, 33.33333333],
       [16.66666667, 66.66666667, 16.66666667],
       [16.66666667, 16.66666667, 66.66666667]])
```



由该混淆矩阵可以得到TP、FN、FP:

|      |  TP   |  FN   |  FP   |
| :--: | :---: | :---: | :---: |
|  -1  | 33.3  | 66.67 | 33.33 |
|  0   | 66.67 | 33.33 |  50   |
|  1   | 66.67 | 33.33 |  50   |

三种precision的计算方法和第一节中计算的一样，就不多介绍了。使用sklearn的函数时，把sw作为函数的sample_weight参数输入即可。



## 参考

1.  <a href="https://zhuanlan.zhihu.com/p/59862986" target="">详解sklearn的多分类模型评价指标</a>

