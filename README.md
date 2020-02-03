# K-Prototypes
改进的K-Prototypes聚类算法
## 使用方法：
```python
from k_prototypes import K_Prototypes
label, n_center, c_center = K_Prototypes(random_seed=2020, data=data, num_numerical=num_numerical_features,
                                         num_category=num_category_features, max_iters=10, mode=3, n=N)
```
## 在[GTD数据集](https://www.start.umd.edu/gtd/access/)上的效果对比
| 算法名称 | 每一类别样本个数 | Calinski-Harabaz Index |
|:---:|:---:|:---:|
| 本文的K-Prototypes算法 | 90/81/58/48/23 | 2.2126 |
| K-Means算法 | 106/86/61/36/11 | 0.7589 |
| K-Modes算法 | 72/65/57/55/51 | 1.6840 |
| [开源包的K_Prototypes算法](https://github.com/nicodv/kmodes) | 67/65/61/54/53 | 0.7630 |
