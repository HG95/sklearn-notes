# preprocessing.Binarizer用法

二值数据是使用值将数据转化为二值，大于阈值设置为1，小于阈值设置为0。这个过程被叫做二分数据或阈值转换。在生成明确值或特征工程增加属性时候使用，使用scikit-learn 中的`Binarizer`类实现。

```python
sklearn.preprocessing.Binarizer(threshold=0.0, 
                                copy=True
                               )
```



**Examples**

```
from sklearn.preprocessing import Binarizer
X = [[ 1., -1.,  2.],
     [ 2.,  0.,  0.],
     [ 0.,  1., -1.]]
transformer = Binarizer().fit(X)  # fit does nothing.
transformer

transformer.transform(X)
```

二值化器(binarizer)的阈值是可以被调节的:

```python
from  sklearn.preprocessing import  Binarizer
from  sklearn import preprocessing

X = [[ 1., -1.,  2.],
      [ 2.,  0.,  0.],
      [ 0.,  1., -1.]]

transform = Binarizer(threshold=0.0)
newX=transform.fit_transform(X)
# print(mm)

# transform = Binarizer(threshold=0.0).fit(X)
# newX = transform.transform(X)

binarizer = preprocessing.Binarizer().fit(X)  # fit does nothing
print(binarizer)
#Binarizer(copy=True, threshold=0.0)

print(binarizer.transform(X))
'''
[[1. 0. 1.]
 [1. 0. 0.]
 [0. 1. 0.]]
'''

binarizer = preprocessing.Binarizer(threshold=1.1)
print(binarizer.transform(X))
'''
[[0. 0. 1.]
 [1. 0. 0.]
 [0. 0. 0.]]
'''

```

