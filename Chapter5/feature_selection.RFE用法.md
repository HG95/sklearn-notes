# feature_selection.RFE用法

包装法也是一个特征选择和算法训练同时进行的方法，与嵌入法十分相似，它也是依赖于算法自身的选择，比如
`coef_`属性或 `feature_importances_` 属性来完成特征选择。但不同的是，我们往往使用一个目标函数作为黑盒来帮助我们选取特征，而不是自己输入某个评估指标或统计量的阈值。包装法在初始特征集上训练评估器，并且通过`coef_`属性或通过`feature_importances_` 属性获得每个特征的重要性。然后，从当前的一组特征中修剪最不重要的特征。在修剪的集合上递归地重复该过程，直到最终到达所需数量的要选择的特征。区别于过滤法和嵌入法的一次训练解决所有问题，包装法要使用特征子集进行多次训练，因此它所需要的计算成本是最高的。  

<center>
    <img src="https://raw.githubusercontent.com/HG1227/image/master/img_tuchuang/20200106155541.png"/>
</center>

注意，在这个图中的“算法”，指的不是我们最终用来导入数据的分类或回归算法（即不是随机森林），而是专业的据挖掘算法，即我们的目标函数。这些数据挖掘算法的核心功能就是选取最佳特征子集。

最典型的目标函数是递归特征消除法（`Recursive feature elimination`, 简写为RFE）。它是一种贪婪的优化算法，旨在找到性能最佳的特征子集。 它反复创建模型，并在每次迭代时保留最佳特征或剔除最差特征，下一次迭代时，它会使用上一次建模中没有被选中的特征来构建下一个模型，直到所有特征都耗尽为止。 然后，它根据自己保留或剔除特征的顺序来对特征进行排名，最终选出一个最佳子集。

包装法的效果是所有特征选择方法中最利于提升模型表现的，它可以使用很少的特征达到很优秀的效果。除此之外，在特征数目相同时，包装法和嵌入法的效果能够匹敌，不过它比嵌入法算得更见缓慢，所以也不适用于太大型的数据。相比之下，包装法是最能保证模型效果的特征选择方法  

```python
class sklearn.feature_selection.RFE (estimator, 
                                     n_features_to_select=None, 
                                     step=1, 
                                     verbose=0
                                    )
```



参数`estimator`是需要填写的实例化后的评估器，`n_features_to_select`是想要选择的特征个数，`step`表示每次迭代中希望移除的特征个数。除此之外，RFE类有两个很重要的属性，`.support_`：返回所有的特征的是否最后被选中的布尔矩阵，以及 ` .ranking_` 返回特征的按数次迭代中综合重要性的排名  

类`feature_selection.RFECV`会在交叉验证循环中执行RFE以找到最佳数量的特征，增加参数`cv`，其他用法都和RFE一模一样  

```python
from sklearn.feature_selection import RFE
RFC_ = RFC(n_estimators =10,random_state=0)
selector = RFE(RFC_, n_features_to_select=340, step=50).fit(X, y)
selector.support_.sum()
selector.ranking_
X_wrapper = selector.transform(X)
cross_val_score(RFC_,X_wrapper,y,cv=5).mean()
#======【TIME WARNING: 15 mins】======#
score = []
for i in range(1,751,50):
X_wrapper = RFE(RFC_,n_features_to_select=i, step=50).fit_transform(X,y)
once = cross_val_score(RFC_,X_wrapper,y,cv=5).mean()
score.append(once)
plt.figure(figsize=[20,5])
plt.plot(range(1,751,50),score)
plt.xticks(range(1,751,50))
plt.show()
```

<center>
    <img src="https://raw.githubusercontent.com/HG1227/image/master/img_tuchuang/20200106155957.png"/>
</center>

明显能够看出，在包装法下面，应用50个特征时，模型的表现就已经达到了90%以上，比嵌入法和过滤法都高效很多。我们可以放大图像，寻找模型变得非常稳定的点来画进一步的学习曲线（就像我们在嵌入法中做的那样）。如果我们此时追求的是最大化降低模型的运行时间，我们甚至可以直接选择50作为特征的数目，这是一个在缩减了94%的特征的基础上，还能保证模型表现在90%以上的特征组合，不可谓不高效  