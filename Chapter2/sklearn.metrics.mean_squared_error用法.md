均方误差

该指标计算的是拟合数据和原始数据对应样本点的误差的 平方和的均值，其值越小说明拟合效果越好





<center>
    <img src="https://raw.githubusercontent.com/HG1227/image/master/img_tuchuang/20200113164809.png"/>
</center>



```python
metrics.mean_squared_error(y_true, 
                           y_pred,                         
                           sample_weight=None, 
                           multioutput=’uniform_average’)
```

参数：
y_true：真实值。

y_pred：预测值。

sample_weight：样本权值。

multioutput：多维输入输出，默认为’uniform_average’，计算所有元素的均方误差，返回为一个标量；也可选‘raw_values’，计算对应列的均方误差，返回一个与列数相等的一维数组。


```python
from sklearn.metrics import mean_squared_error
y_true = [3, -1, 2, 7]
y_pred = [2, 0.0, 2, 8]
mean_squared_error(y_true, y_pred)
# 结果为：0.75
y_true = [[0.5, 1],[-1, 1],[7, -6]]
y_pred = [[0, 2],[-1, 2],[8, -5]]
mean_squared_error(y_true, y_pred)
# 结果为：0.7083333333333334
mean_squared_error(y_true, y_pred, multioutput='raw_values')
# 结果为：array([0.41666667, 1.        ])
mean_squared_error(y_true, y_pred, multioutput=[0.3, 0.7])
# 结果为：0.825
# multioutput=[0.3, 0.7]返回将array([0.41666667, 1.        ])按照0.3*0.41666667+0.7*1.0计算所得的结果
mean_squared_error(y_true, y_pred, multioutput='uniform_average')
# 结果为：0.7083333333333334
```

# RMSE(均方根误差)

## 定义

即 均方误差开平方

RMSE，全称是Root Mean Square Error，即均方根误差，它表示预测值和观测值之间差异（称为残差）的样本标准差。均方根误差为了说明样本的离散程度。做非线性拟合时, RMSE越小越好。

## 标准差与均方根误差的区别

- 标准差是用来衡量一组数自身的离散程度，而均方根误差是用来衡量观测值同真值之间的偏差，它们的研究对象和研究目的不同，但是计算过程类似。
- 均方根误差算的是观测值与其真值，或者观测值与其模拟值之间的偏差，而不是观测值与其平均值之间的偏差。