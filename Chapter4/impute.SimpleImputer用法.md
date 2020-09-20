# impute.SimpleImputer用法

```python
sklearn.impute.SimpleImputer (missing_values=nan, 
                              strategy=’mean’, 
                              fill_value=None, 
                              verbose=0,
                              copy=True
                             )
```

这个类是专门用来填补缺失值的。它包括四个重要参数：  

**`missing_values`** 

告诉SimpleImputer，数据中的缺失值长什么样，默认空值np.nan

**`strategy`** 

填补缺失值的策略，默认均值

输入“mean”使用均值填补（**仅对数值型特征可用**）

输入“median"用中值填补（**仅对数值型特征可用**）

输入"most_frequent”用众数填补（**对数值型和字符型特征都可用**）

输入“constant"表示请参考参数“fill_value"中的值（**对数值型和字符型特征都可用**）

**`fill_value`** 

当参数 startegy 为 ”constant" 的时候可用，可输入字符串或数字表示要填充的值，常用 0





**Example**

```python
import pandas as pd
data = pd.read_csv(r"Narrativedata.csv",index_col=0)
data.head()
```

```
	Age	Sex	Embarked	Survived
0	22.0	male	S	No
1	38.0	female	C	Yes
2	26.0	female	S	Yes
3	35.0	female	S	Yes
4	35.0	male	S	No
```

```python
data.info()
```

```
<class 'pandas.core.frame.DataFrame'>
Int64Index: 891 entries, 0 to 890
Data columns (total 4 columns):
Age         714 non-null float64
Sex         891 non-null object
Embarked    889 non-null object
Survived    891 non-null object
dtypes: float64(1), object(3)
memory usage: 34.8+ KB
```

```python
Age = data.loc[:,"Age"].values.reshape(-1,1) #sklearn当中特征矩阵必须是二维

from sklearn.impute import SimpleImputer

imp_mean = SimpleImputer() #实例化，默认均值填补

imp_median = SimpleImputer(strategy="median") #用中位数填补


imp_0 = SimpleImputer(strategy="constant",fill_value=0) #用0填补

imp_mean = imp_mean.fit_transform(Age) #fit_transform一步完成调取结果
imp_median = imp_median.fit_transform(Age)

imp_0 = imp_0.fit_transform(Age)

imp_mean[:20]
imp_median[:20]
imp_0[:20]

#在这里我们使用中位数填补Age
data.loc[:,"Age"] = imp_median
data.info()

#使用众数填补Embarked
Embarked = data.loc[:,"Embarked"].values.reshape(-1,1)

imp_mode = SimpleImputer(strategy = "most_frequent")
data.loc[:,"Embarked"] = imp_mode.fit_transform(Embarked)
data.info()
```



## BONUS：用Pandas和Numpy进行填补其实更加简单  

```python
import pandas as pd
data = pd.read_csv(r"Narrativedata.csv",index_col=0)
data.head()
data.loc[:,"Age"] = data.loc[:,"Age"].fillna(data.loc[:,"Age"].median())
#.fillna 在DataFrame里面直接进行填补

data.dropna(axis=0,inplace=True)
#.dropna(axis=0)删除所有有缺失值的行，.dropna(axis=1)删除所有有缺失值的列
#参数inplace，为True表示在原数据集上进行修改，为False表示生成一个复制对象，不修改原数据，默认False
```

