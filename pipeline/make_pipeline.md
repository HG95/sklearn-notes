# make_pipeline

用 `Pipeline`类构建管道时语法有点麻烦，我们通常不需要为每一个步骤提供用户指定的名称，这种情况下，就可以用`make_pipeline` 函数创建管道，它可以为我们创建管道并根据每个步骤所属的类为其自动命名。

```python
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(MinMaxScaler(),SVC())
```

一般来说，自动命名的步骤名称是类名称的小写版本，如果多个步骤属于同一个类，则会附加一个数字。 

# 案例

```python
# 加载数据
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv(
    "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data", header=None)
# 做基本的数据预处理

X = df.iloc[:, 2:].values
y = df.iloc[:, 1].values
le = LabelEncoder()  # 将M-B等字符串编码成计算机能识别的0-1
y = le.fit_transform(y)
le.transform(['M', 'B'])

# 数据切分8：2
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=1)
```

```python
# 把所有的操作全部封在一个管道pipeline内形成一个工作流：
# 标准化+PCA+逻辑回归

# make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

pipe_lr1 = make_pipeline(StandardScaler(), 
                         PCA( n_components=2), 
                         LogisticRegression(random_state=1)
                        )
pipe_lr1.fit(X_train, y_train)
y_pred1 = pipe_lr1.predict(X_test)
print("Test Accuracy: %.3f" % pipe_lr1.score(X_test, y_test))
```

结果：

```
Test Accuracy: 0.956
```

---

`make_pipeline`  同样可以和交叉验证等相结合 。



# 参考

- <a href="https://mp.weixin.qq.com/s/LV020zM9EPwABLDP04NSZA" target="_blank">Datawhale:常用数据分析方法：方差分析及实现！</a>  
- <a href="https://blog.csdn.net/elma_tww/article/details/88427695" target="_blank">《Python机器学习基础教程》构建管道(make_pipeline)</a> 