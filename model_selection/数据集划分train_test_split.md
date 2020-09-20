# 数据集划分train_test_split

`train_test_split` 是交叉验证中常用的函数，功能是从样本中随机的按比例选取`train_data`和`test_data`，形式为：

```python
sklearn.model_selection.train_test_split(train_data,
                                         train_target,
                                         test_siz=, 
                                         random_state=,
                                         shuffle)
```

参数

- `train_data`：所要划分的样本特征集
- `train_target`：所要划分的样本结果
- `test_size`：样本占比，如果是整数的话就是样本的数量
- `random_state`：是随机数的种子。
- `shuffle` : bool, default=True

返回值 ：划分后的数据集

随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
随机数的产生取决于种子，随机数和种子之间的关系遵从以下两个规则：
种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。

# 案例

```python
import numpy as np
from sklearn.model_selection import train_test_split


X = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
y = [0, 1, 2, 3, 4]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=0)
```



