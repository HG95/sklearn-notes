# 模型融合Blending

## 概念

Blending与Stacking大致相同，只是Blending的主要区别在于训练集不是通过K-Fold的CV策略来获得预测值从而生成第二阶段模型的特征，而是建立一个Holdout集。简单来说，Blending直接用不相交的数据集用于不同层的训练。

## Blending 流程

模型融合有许多方法，简单的有平均融合，加权融合，投票融合等方法；较为复杂的就是Blending和Stacking了。

`Blending`相较于`Stacking`来说要简单一些，其流程大致分为以下几步：

1. 将数据划分为训练集和测试集(test_set)，其中训练集需要再次划分为训练集(train_set)和验证集(val_set)；
2. 创建第一层的多个模型，这些模型可以使同质的也可以是异质的；
3. 使用train_set训练步骤2中的多个模型，然后用训练好的模型预测val_set和test_set得到val_predict, test_predict1；
4. 创建第二层的模型,使用val_predict作为训练集训练第二层的模型；
5. 使用第二层训练好的模型对第二层测试集test_predict1进行预测，该结果为整个测试集的结果

## Blending 图解

<img src=".\img\2019052109512454.png" alt="2019052109512454" style="zoom:80%;" />

## Blending与Stacking对比

Blending的优点在于：

1.比stacking简单（因为不用进行k次的交叉验证来获得stacker feature）

2.避开了一个信息泄露问题：generlizers和stacker使用了不一样的数据集

3.在团队建模过程中，不需要给队友分享自己的随机种子

而缺点在于：

1.使用了很少的数据（是划分hold-out作为测试集，并非cv）

2.blender可能会过拟合（其实大概率是第一点导致的）

3.stacking使用多次的CV会比较稳健



## python 实现

```python
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.datasets.samples_generator import make_blobs

'''创建训练的数据集'''
data, target = make_blobs(n_samples=50000, centers=2, random_state=0, cluster_std=0.60)

'''模型融合中使用到的各个单模型'''
clfs = [RandomForestClassifier(n_estimators=5, n_jobs=-1, criterion='gini'),
        RandomForestClassifier(n_estimators=5, n_jobs=-1, criterion='entropy'),
        ExtraTreesClassifier(n_estimators=5, n_jobs=-1, criterion='gini'),
        ExtraTreesClassifier(n_estimators=5, n_jobs=-1, criterion='entropy'),
        GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=5)]

'''切分一部分数据作为测试集'''
X, X_predict, y, y_predict = train_test_split(data, target, test_size=0.33, random_state=2017)

'''5折stacking'''
n_folds = 5
skf = list(StratifiedKFold(y, n_folds))

'''切分训练数据集为d1,d2两部分'''
X_d1, X_d2, y_d1, y_d2 = train_test_split(X, y, test_size=0.5, random_state=2017)
dataset_d1 = np.zeros((X_d2.shape[0], len(clfs)))
dataset_d2 = np.zeros((X_predict.shape[0], len(clfs)))

for j, clf in enumerate(clfs):
    '''依次训练各个单模型'''
    # print(j, clf)
    '''使用第1个部分作为预测，第2部分来训练模型，获得其预测的输出作为第2部分的新特征。'''
    # X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
    clf.fit(X_d1, y_d1)
    y_submission = clf.predict_proba(X_d2)[:, 1]
    dataset_d1[:, j] = y_submission
    '''对于测试集，直接用这k个模型的预测值作为新的特征。'''
    dataset_d2[:, j] = clf.predict_proba(X_predict)[:, 1]
    print("val auc Score: %f" % roc_auc_score(y_predict, dataset_d2[:, j]))

'''融合使用的模型'''
# clf = LogisticRegression()
clf = GradientBoostingClassifier(learning_rate=0.02, subsample=0.5, max_depth=6, n_estimators=30)
clf.fit(dataset_d1, y_d2)
y_submission = clf.predict_proba(dataset_d2)[:, 1]

print("Linear stretch of predictions to [0,1]")
y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
print("blend result")
print("val auc Score: %f" % (roc_auc_score(y_predict, y_submission)))
```





参考

- <a href="https://blog.csdn.net/sinat_35821976/article/details/83622594" target="_blank">图解Blending&Stacking</a>



