# 学习曲线learning_curve

`learning_curve` 是展示不同数据量，算法学习得分

确定交叉验证的针对不同训练集大小的训练和测试分数。

交叉验证生成器将整个数据集拆分为训练和测试数据中的 k 次。 具有不同大小的训练集的子集将用于训练估计器，并为每个训练子集大小和测试集计算分数。 之后，对于每个训练子集大小，将对所有k次运行的得分进行平均。

## 函数

```python
sklearn.model_selection.learning_curve(estimator, 
                                       X, y, *, 
                                       groups=None, 
                                       train_sizes=array([0.1, 0.33, 0.55, 0.78, 1. ]), 
                                       cv=None, 
                                       scoring=None, 
                                       exploit_incremental_learning=False, 
                                       n_jobs=None, 
                                       pre_dispatch='all', 
                                       verbose=0, 
                                       shuffle=False, 
                                       random_state=None, 
                                       error_score=nan, 
                                       return_times=False)
```

**参数** 

- `estimator`：实现“ fit”和“ predict”方法的对象类型
  每次验证都会克隆的该类型的对象。

- `X`：数组类，形状（n_samples，n_features）
  训练向量，其中n_samples是样本数，n_features是特征数。

- `y`：数组类，形状（n_samples）或（n_samples，n_features），可选
  相对于X的目标进行分类或回归；无监督学习无。

- `groups`：数组类，形状为（n_samples，），可选
  将数据集拆分为训练/测试集时使用的样本的标签分组。仅用于连接交叉验证实例组（例如GroupKFold）。

- `train_sizes`：数组类，形状（n_ticks），dtype float或int
  训练示例的相对或绝对数量，将用于生成学习曲线。如果dtype为float，则视为训练集最大尺寸的一部分（由所选的验证方法确定），即，它必须在（0，1]之内，否则将被解释为绝对大小注意，为了进行分类，样本的数量通常必须足够大，以包含每个类中的至少一个样本（默认值：np.linspace（0.1，1.0，5））

- `cv`：int，交叉验证生成器或可迭代的，可选的
  确定交叉验证拆分策略。cv的可能输入是：
  - `None`，要使用默认的三折交叉验证（v0.22版本中将改为五折）
    整数，用于指定（分层）KFold中的折叠数，
  - `CV splitter`
  - 可迭代的集（训练，测试）拆分为索引数组。
    对于整数/无输入，如果估计器是分类器，y是二进制或多类，则使用StratifiedKFold。在所有其他情况下，都使用KFold。
- `scoring`：字符串，可调用或无，可选，默认：None
  字符串（参阅model evaluation documentation）或带有签名scorer(estimator, X, y)的计分器可调用对象/函数。

- `exploit_incremental_learning`：布尔值，可选，默认值：False
  如果估算器支持增量学习，此参数将用于加快拟合不同训练集大小的速度。

- `n_jobs`：int或None，可选（默认=None）
  要并行运行的作业数。None表示1。 -1表示使用所有处理器。有关更多详细信息，请参见词汇表。

- `pre_dispatch`：整数或字符串，可选
  并行执行的预调度作业数（默认为全部）。该选项可以减少分配的内存。该字符串可以是“ 2 * n_jobs”之类的表达式。

- `verbose`：整数，可选
  控制详细程度：越高，消息越多。

- `shuffle`：布尔值，可选
  是否在基于`train_sizes`为前缀之前对训练数据进行洗牌。

- `random_state`：int，RandomState实例或无，可选（默认=None）
  如果为int，则random_state是随机数生成器使用的种子；否则为false。如果是RandomState实例，则random_state是随机数生成器；如果为None，则随机数生成器是np.random使用的RandomState实例。在shuffle为True时使用。

- `error_score`：`raise` | `raise-deprecating` 或数字
  如果估算器拟合中出现错误，则分配给分数的值。如果设置为“ raise”，则会引发错误。如果设置为“raise-deprecating”，则会在出现错误之前打印FutureWarning。如果给出数值，则引发FitFailedWarning。此参数不会影响重新安装步骤，这将始终引发错误。默认值为“不赞成使用”，但从0.22版开始，它将更改为np.nan。


**返回值**

- `train_sizes_abs`：数组，形状（n_unique_ticks，），dtype int
  已用于生成学习曲线的训练示例数。 请注意，ticks的数量可能少于n_ticks，因为重复的条目将被删除。

- `train_scores`：数组，形状（n_ticks，n_cv_folds）
  训练集得分。

- `test_scores`：数组，形状（n_ticks，n_cv_folds）
  测试集得分。

  

## 案例

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use("ggplot")
warnings.filterwarnings("ignore")

# 加载数据
df = pd.read_csv(
    "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data", header=None)
# 做基本的数据预处理

X = df.iloc[:, 2:].values
y = df.iloc[:, 1].values
le = LabelEncoder()  # 将M-B等字符串编码成计算机能识别的0-1
y = le.fit_transform(y)
le.transform(['M', 'B'])

# 数据切分8：2

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=1)
```

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


# 用学习曲线诊断偏差与方差
from sklearn.model_selection import learning_curve


pipe_lr3 = make_pipeline(
    StandardScaler(), LogisticRegression(random_state=1, penalty='l2'))

train_sizes, train_scores, test_scores = learning_curve(
    estimator=pipe_lr3, X=X_train, y=y_train, train_sizes=np.linspace(0.1, 1, 10), cv=10, n_jobs=1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)


# 在画训练集的曲线时：横轴为 train_sizes，纵轴为 train_scores_mean；

# 画测试集的曲线时：横轴为train_sizes，纵轴为test_scores_mean。

plt.plot(train_sizes, train_mean, color='blue', marker='o',
         markersize=5, label='training accuracy')
plt.fill_between(train_sizes, train_mean+train_std,
                 train_mean-train_std, alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean, color='red', marker='s',
         markersize=5, label='validation accuracy')
plt.fill_between(train_sizes, test_mean+test_std,
                 test_mean-test_std, alpha=0.15, color='red')

plt.xlabel("Number of training samples")
plt.ylabel("Accuracy")
plt.legend(loc='lower right')
plt.ylim([0.8, 1.02])
plt.savefig('te.png', dpi=300)
```

<center><img src="https://raw.githubusercontent.com/HG1227/image/master/img_tuchuang/20200708205742.png" alt="te" style="zoom:60%;" /></center>





# 参考

- <a href="https://blog.csdn.net/gracejpw/article/details/102370364" target="_blank">sklearn中的学习曲线learning_curve函数</a> 
- <a href="https://blog.csdn.net/qq_36523839/article/details/82556932" target="_blank">绘制学习曲线——plot_learning_curve</a> 
- <a href="https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection" target="_blank">sklearn.model_selection.learning_curve</a> 