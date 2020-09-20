# sklearn.metrics.mean_absolute_error用法

平均绝对误差(MAE)

- Mean Absolute Error ，平均绝对误差
- 它表示预测值和观测值之间绝对误差的平均值。
- 是绝对误差的平均值
- 能更好地反映预测值误差的实际情况.
- 用于评估预测结果和真实数据集的接近程度的程度 ，其其值越小说明拟合效果越好


$$
M A E(X, h)=\frac{1}{m} \sum_{i=1}^{m}\left|h\left(x_{i}\right)-y_{i}\right|
$$

```python
sklearn.metrics.mean_absolute_error(y_true, 
                                    y_pred, 
                                    *, 
                                    sample_weight=None, 
                                    multioutput='uniform_average')
```

参数

- **y_true** ：*array-like of shape (n_samples,) or (n_samples, n_outputs)*

  Ground truth (correct) target values.

- **y_pred** ： array-like of shape (n_samples,) or (n_samples, n_outputs)

  Estimated target values.

- **sample_weight** ：array-like of shape (n_samples,), optional
  样本权重

- **multioutput** ：**string in [‘raw_values’, ‘uniform_average’] or array-like of shape (n_outputs)**
  定义多个输出值的聚合。类似数组的值定义用于平均错误的权重。
  **‘raw_values’ :** Returns a full set of errors in case of multioutput input.
  **‘uniform_average’ :** Errors of all outputs are averaged with uniform weight. 

返回值：

- **loss**：**float or ndarray of floats**

  If multioutput is ‘raw_values’, then mean absolute error is returned for each output separately. If multioutput is ‘uniform_average’ or an ndarray of weights, then the weighted average of all output errors is returned.

  MAE output is non-negative floating point. The best value is 0.0.





**Examples**

```python
from sklearn.metrics import mean_absolute_error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
mean_absolute_error(y_true, y_pred)
# 0.5

y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]
mean_absolute_error(y_true, y_pred)
# 0.75

mean_absolute_error(y_true, y_pred, multioutput='uniform_average')
# 0.75

mean_absolute_error(y_true, y_pred, multioutput='raw_values')
# array([0.5, 1. ])

mean_absolute_error(y_true, y_pred, multioutput=[0.3, 0.7])
# 0.85 
# 0.5*0.3+1*0.7=0.85
```



