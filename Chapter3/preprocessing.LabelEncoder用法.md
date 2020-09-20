# preprocessing.LabelEncoder用法

在机器学习中，大多数算法，譬如逻辑回归，支持向量机SVM，k近邻算法等都只能够处理数值型数据，不能处理文字，在sklearn当中，除了专用来处理文字的算法，其他算法在fit的时候全部要求输入数组或矩阵，也不能够导入文字型数据（其实手写决策树和普斯贝叶斯可以处理文字，但是sklearn中规定必须导入数值型）  

然而在现实中，许多标签和特征在数据收集完毕的时候，都不是以数字来表现的。比如说，学历的取值可以是["小学"，“初中”，“高中”，"大学"]，付费方式可能包含["支付宝"，“现金”，“微信”]等等。在这种情况下，为了让数据适应算法和库，我们必须将数据进行编码，即是说，将文字型数据转换为数值型  



```python
class sklearn.preprocessing.LabelEncoder
```

**Attributes**：

**`classes_`**

*array of shape (n_class,)*

Holds the label for each class



**Methods**

|                                                              |                                             |
| ------------------------------------------------------------ | ------------------------------------------- |
| [`fit`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html?highlight=labelencoder#sklearn.preprocessing.LabelEncoder.fit)(self, y) | Fit label encoder                           |
| [`fit_transform`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html?highlight=labelencoder#sklearn.preprocessing.LabelEncoder.fit_transform)(self, y) | Fit label encoder and return encoded labels |
| [`get_params`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html?highlight=labelencoder#sklearn.preprocessing.LabelEncoder.get_params)(self[, deep]) | Get parameters for this estimator.          |
| [`inverse_transform`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html?highlight=labelencoder#sklearn.preprocessing.LabelEncoder.inverse_transform)(self, y) | Transform labels back to original encoding. |
| [`transform`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html?highlight=labelencoder#sklearn.preprocessing.LabelEncoder.transform)(self, y) | Transform labels to normalized encoding.    |

LabelEncoder 可以将标签分配一个0—n_classes-1之间的编码将各种标签分配一个可数的连续编号：

**Example**

```python
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

lb=le.fit(["paris", "paris", "tokyo", "amsterdam"])
lb.classes_
# array(['amsterdam', 'paris', 'tokyo'], dtype='<U9')

lb.transform(["tokyo", "tokyo", "paris", "amsterdam", "amsterdam"]) 
# array([2, 2, 1, 0, 0], dtype=int64)
```

