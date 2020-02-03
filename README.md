# K-Prototypes
改进的K-Prototypes聚类算法
## 使用方法：
```python
from k_prototypes import K_Prototypes
label, n_center, c_center = K_Prototypes(random_seed=2020, data=data, num_numerical=num_numerical_features,
                                         num_category=num_category_features, max_iters=10, mode=3, n=N)
```
## 效果对比
| 算法名称 | 每一类别样本个数 | Calinski-Harabaz Index |
|:---:|:---:|:---:|
| 本文的K-Prototypes算法 | 0 | 0 |
| K-Means算法 | 0 | 0 |
| K-Modes算法 | 0 | 0 |
| 开源包的K_Prototypes算法 | 0 | 0 |
