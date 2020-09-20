# Pipeline

当我们对训练集应用各种预处理操作时（特征标准化、主成分分析等等）， 我们都需要对测试集重复利用这些参数，以免出现数据泄露（data leakage）。

pipeline 实现了对全部步骤的流式化封装和管理（streaming workflows with pipelines），可以很方便地使参数集在新数据集（比如测试集）上被**重复使用**。

Pipeline可以将许多算法模型串联起来，比如将特征提取、归一化、分类组织在一起形成一个典型的机器学习问题工作流。

pipeline 可以用于下面几处：

- 模块化 Feature Transform，只需写很少的代码就能将新的 Feature 更新到训练集中。
- 自动化 Grid Search，只要预先设定好使用的 Model 和参数的候选，就能自动搜索并记录最佳的 Model。
- 自动化 Ensemble Generation，每隔一段时间将现有最好的 K 个 Model 拿来做 Ensemble。





```python
sklearn.pipeline.make_pipeline(*steps, **kwargs)
```

Parameters

- `steps` : 步骤：列表(list)
  被连接的（名称，变换）元组（实现拟合/变换）的列表，按照它们被连接的顺序，最后一个对象是估计器(estimator)。
- `memory`: 内存参数,Instance of sklearn.external.joblib.Memory or string, optional (default=None)
- 属性, name_steps:bunch object，具有属性访问权限的字典
  只读属性以用户给定的名称访问任何步骤参数。键是步骤名称，值是步骤参数。或者也可以直接通过”.步骤名称”获取

funcution

- Pipline的方法都是执行各个学习器中对应的方法,如果该学习器没有该方法,会报错
- 假设该Pipline共有n个学习器
- `transform` ,依次执行各个学习器的transform方法
- `fit`:依次对前n-1个学习器执行fit和transform方法,第n个学习器(最后一个学习器)执行fit方法
- `predict`: 执行第n个学习器的`predict`方法
- `score`： 执行第 n 个学习器的score方法
- `set_params`: 设置第n个学习器的参数
- `get_param` :,获取第n个学习器的参数



# 举例

:star: 问题是要对数据集 Breast Cancer Wisconsin 进行分类，

它包含 569 个样本，第一列 ID，第二列类别(M=恶性肿瘤，B=良性肿瘤)，第 3-32 列是实数值的特征。

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

我们要用 **Pipeline** 对训练集和测试集进行如下操作：

- 先用 `StandardScaler` 对数据集每一列做标准化处理，（是 transformer）
- 再用 `PCA` 将原始的 30 维度特征压缩的 2 维度，（是 transformer）
- 最后再用模型 `LogisticRegression`。（是 Estimator）

**调用 Pipeline 时**，输入由元组构成的列表，每个元组第一个值为变量名，元组第二个元素是 sklearn 中的 transformer 或 Estimator。

注意中间每一步是 **transformer**，即它们必须包含 fit 和 transform 方法，或者 `fit_transform`。

最后一步是一个 **Estimator**，即最后一步模型要有 fit 方法，可以没有 transform 方法。

然后用 **Pipeline.fit**对训练集进行训练，`pipe_lr.fit(X_train, y_train)`
再直接用 **Pipeline.score** 对测试集进行预测并评分 `pipe_lr.score(X_test, y_test)`

把所有的操作全部封在一个管道pipeline内形成一个工作流：

标准化+PCA+逻辑回归

```python
# Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe_lr2 = Pipeline([['std', StandardScaler()], ['pca', PCA(n_components=2)], [
                    'lr', LogisticRegression(random_state=1)]])
pipe_lr2.fit(X_train, y_train)
y_pred2 = pipe_lr2.predict(X_test)
print("Test Accuracy: %.3f" % pipe_lr2.score(X_test, y_test))
```

结果：Test Accuracy: 0.956

## Pipeline 的工作方式：

当管道 `Pipeline` 执行 `fit` 方法时，
首先 `StandardScaler` 执行 `fit `和 `transform` 方法，
然后将转换后的数据输入给 `PCA`，
`PCA` 同样执行 `fit` 和 `transform` 方法，
再将数据输入给 `LogisticRegression`，进行训练。

<img src="https://raw.githubusercontent.com/HG1227/image/master/img_tuchuang/20200708175055.png" alt="162d6d3f6b4fcbc2 (1)" style="zoom:67%;" />



注意中间每一步是**transformer**，即它们必须包含 fit 和 transform 方法，或者`fit_transform`。

最后一步是一个**Estimator**，即最后一步模型要有 fit 方法，可以没有 transform 方法。



:star:  当然，还可以用来选择特征，也可以应用 `K-fold cross validation`



# 与交叉验证结合

## k 折交叉验证

```python
from sklearn.model_selection import cross_val_score

scores1 = cross_val_score(estimator=pipe_lr1, X=X_train,
                          y=y_train, cv=10, n_jobs=1)
print("CV accuracy scores:%s" % scores1)
print("CV accuracy:%.3f +/-%.3f" % (np.mean(scores1), np.std(scores1)))
```

结果：

```
CV accuracy scores:[0.93478261 0.93478261 0.95652174 0.95652174 0.93478261 0.95555556
 0.97777778 0.93333333 0.95555556 0.95555556]
CV accuracy:0.950 +/-0.014
```

## 分层 k 折交叉验证

```python
# 分层 k折交叉验证

from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=10, random_state=1).split(X_train, y_train)
scores2 = []
for k, (train, test) in enumerate(kfold):
    pipe_lr1.fit(X_train[train], y_train[train])
    score = pipe_lr1.score(X_train[test], y_train[test])
    scores2.append(score)
    print('Fold:%2d,Class dist.:%s,Acc:%.3f' %
          (k+1, np.bincount(y_train[train]), score))
    
print('\nCV accuracy :%.3f +/-%.3f' % (np.mean(scores2), np.std(scores2)))
```

结果：

```python
Fold: 1,Class dist.:[256 153],Acc:0.935
Fold: 2,Class dist.:[256 153],Acc:0.935
Fold: 3,Class dist.:[256 153],Acc:0.957
Fold: 4,Class dist.:[256 153],Acc:0.957
Fold: 5,Class dist.:[256 153],Acc:0.935
Fold: 6,Class dist.:[257 153],Acc:0.956
Fold: 7,Class dist.:[257 153],Acc:0.978
Fold: 8,Class dist.:[257 153],Acc:0.933
Fold: 9,Class dist.:[257 153],Acc:0.956
Fold:10,Class dist.:[257 153],Acc:0.956

CV accuracy :0.950 +/-0.014
```



# 参考

- <a href="https://mp.weixin.qq.com/s/LV020zM9EPwABLDP04NSZA" target="_blank">Datawhale:常用数据分析方法：方差分析及实现！</a>  
- <a href="https://zhuanlan.zhihu.com/p/42368821" target="_blank">利用sklearn中pipeline构建机器学习工作流</a> 
- <a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.pipeline" target="_blank">sklearn.pipeline.Pipeline </a>  
- <a herf="" target="_blank"></a>
- <a herf="" target="_blank"></a>