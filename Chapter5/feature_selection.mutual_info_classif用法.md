# feature_selection.mutual_info_classif用法

互信息法是用来捕捉每个特征与标签之间的任意关系（包括线性和非线性关系）的过滤方法。和F检验相似，它既
可以做回归也可以做分类，并且包含两个类 `feature_selection.mutual_info_classif`（互信息分类）和
`feature_selection.mutual_info_regression`（互信息回归）。这两个类的用法和参数都和F检验一模一样，不过互信息法比F检验更加强大，F检验只能够找出线性关系，而互信息法可以找出任意关系  

```python
sklearn.feature_selection.mutual_info_classif(X, y, 
                                              discrete_features='auto', 
                                              n_neighbors=3, 
                                              copy=True, 
                                              random_state=None
                                             )
```

`X` :特征矩阵

**`y`** :目标向量

**discrete_features** : **{‘auto’, bool, array_like}, default ‘auto’**