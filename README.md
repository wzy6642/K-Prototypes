# K-Prototypes
改进的K-Prototypes聚类算法，参考文献[Determining the number of clusters using information entropy for mixed data](http://jiyeliang.net/Cms_Data/Contents/SXU_JYL/Folders/JournalPapers/~contents/BD5238EEQPQYXZPP/Determining%20the%20number%20of%20clusters%20using%20information%20entropy%20for%20mixed%20data(2012)PR.pdf)

## 使用方法：
```python
from k_prototypes import K_Prototypes
label, n_center, c_center = K_Prototypes(random_seed=2020, data=data, num_numerical=num_numerical_features,
                                         num_category=num_category_features, max_iters=10, mode=3, n=N)
```

## 参数说明：
| 参数名称 | 参数类型 | 参数意义 |
|:---:|:---:|:---:|
| n | int | 聚类中心的个数 |
| data | DataFrame | 用于聚类的样本 |
| random_seed | int | 随机数种子 |
| num_numerical | int | 数值特征个数 |
| num_category | int | 类别特征个数 |
| max_iters | int | 最大迭代次数 |
| mode | int | 计算模式：1-K_Modes，2-K_Means，其他-K_Prototypes |

## 返回值说明：
| 参数名称 | 参数类型 | 参数意义 |
|:---:|:---:|:---:|
| newlabel | list | 最终的聚类结果 |
| center_numerical | DataFrame | 数值型特征的聚类中心点 |
| center_category | DataFrame | 类别型特征的聚类中心点 |

## 在[GTD数据集](https://www.start.umd.edu/gtd/access/)上的效果对比：
| 算法名称 | 每一类别样本个数 | Calinski-Harabaz Index |
|:---:|:---:|:---:|
| 本文的K-Prototypes算法 | 90/81/58/48/23 | 2.2126 |
| K-Means算法 | 106/86/61/36/11 | 0.7589 |
| K-Modes算法 | 72/65/57/55/51 | 1.6840 |
| [开源包的K_Prototypes算法](https://github.com/nicodv/kmodes) | 67/65/61/54/53 | 0.7630 |
