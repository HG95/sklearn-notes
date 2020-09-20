# feature_selection.SelectFromModel用法

## Embedded嵌入法  

嵌入法是一种让算法自己决定使用哪些特征的方法，即特征选择和算法训练同时进行。在使用嵌入法时，我们先使用某些机器学习的算法和模型进行训练，得到各个特征的权值系数，根据权值系数从大到小选择特征。这些权值系数往往代表了特征对于模型的某种贡献或某种重要性，比如决策树和树的集成模型中的`feature_importances_`属性，可以列出各个特征对树的建立的贡献，我们就可以基于这种贡献的评估，找出对模型建立最有用的特征。因此相比于过滤法，嵌入法的结果会更加精确到模型的效用本身，对于提高模型效力有更好的效果  。

<center>
    <img src="https://raw.githubusercontent.com/HG1227/image/master/img_tuchuang/20200106155344.png"/>
</center>



过滤法中使用的统计量可以使用统计知识和常识来查找范围（如p值应当低于显著性水平0.05），而嵌入法中使用的权值系数却没有这样的范围可找——我们可以说，权值系数为0的特征对模型丝毫没有作用，但当大量特征都对模型有贡献且贡献不一时，我们就很难去界定一个有效的临界值。这种情况下，模型权值系数就是我们的超参数，我们或许需要学习曲线，或者根据模型本身的某些性质去判断这个超参数的最佳值究竟应该是多少。  



## `feature_selection.SelectFromModel `

```python
lass sklearn.feature_selection.SelectFromModel (estimator, 
                                                threshold=None, 
                                                prefit=False, 
                                                norm_order=1,
                                                max_features=None
                                               )
```

SelectFromModel是一个元变换器，可以与任何在拟合后具有`coef_`，`feature_importances_`属性或参数中可选惩罚项的评估器一起使用（比如随机森林和树模型就具有属性`feature_importances_`，逻辑回归就带有`l1`和`l2`惩罚项，线性支持向量机也支持`l2`惩罚项）  

对于有`feature_importances_`的模型来说，若重要性低于提供的阈值参数，则认为这些特征不重要并被移除。`feature_importances_`的取值范围是[0,1]，如果设置阈值很小，比如0.001，就可以删除那些对标签预测完全没贡献的特征。如果设置得很接近1，可能只有一两个特征能够被留下  

| 参数         | 说明                                                         |
| ------------ | ------------------------------------------------------------ |
| estimator    | 使用的模型评估器，只要是带feature_importances_或者coef_属性，或带有l1和l2惩罚 项的模型都可以使用 |
| threshold    | 特征重要性的阈值，重要性低于这个阈值的特征都将被删除         |
| prefit       | 默认False，判断是否将实例化后的模型直接传递给构造函数。如果为True，则必须直接 调用fit和transform，不能使用fit_transform，并且SelectFromModel不能与 cross_val_score，GridSearchCV和克隆估计器的类似实用程序一起使用。 |
| norm_order   | k可输入非零整数，正无穷，负无穷，默认值为1 在评估器的coef_属性高于一维的情况下，用于过滤低于阈值的系数的向量的范数的阶 数。 |
| max_features | 在阈值设定下，要选择的最大特征数。要禁用阈值并仅根据max_features选择，请设置 threshold = -np.inf |

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier as RFC

RFC_ = RFC(n_estimators =10,random_state=0)
X_embedded = SelectFromModel(RFC_,threshold=0.005).fit_transform(X,y)
#在这里我只想取出来有限的特征。0.005这个阈值对于有780个特征的数据来说，是非常高的阈值，因为平均每个特征
#只能够分到大约0.001的feature_importances_

X_embedded.shape
#模型的维度明显被降低了
#同样的，我们也可以画学习曲线来找最佳阈值

#======【TIME WARNING：10 mins】======#
import numpy as np
import matplotlib.pyplot as plt

RFC_.fit(X,y).feature_importances_
threshold = np.linspace(0,(RFC_.fit(X,y).feature_importances_).max(),20)
score = []
for i in threshold:
    X_embedded = SelectFromModel(RFC_,threshold=i).fit_transform(X,y)
    once = cross_val_score(RFC_,X_embedded,y,cv=5).mean()
    score.append(once)
plt.plot(threshold,score)
plt.show()
```

<center>
    <img src="https://raw.githubusercontent.com/HG1227/image/master/img_tuchuang/20200106154856.png"/>
</center>

