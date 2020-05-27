# K-Prototypes
改进的K-Prototypes聚类算法，参考文献[Determining the number of clusters using information entropy for mixed data](http://jiyeliang.net/Cms_Data/Contents/SXU_JYL/Folders/JournalPapers/~contents/BD5238EEQPQYXZPP/Determining%20the%20number%20of%20clusters%20using%20information%20entropy%20for%20mixed%20data(2012)PR.pdf)

## 使用方法：
```python
import k_prototypes as kp
# 导入示例数据
data, data_id, num_numerical_features, num_category_features = kp.Load_Data(demo=True)
# 聚类模型
label, center_numerical, center_category = kp.K_Prototypes(random_seed=2020, n=6, data=data, 
                                                           num_numerical=num_numerical_features, 
                                                           num_category=num_category_features, 
                                                           max_iters = 10, mode=3)
# 模型评价
CUM = kp.CUM_index(data=data, num_category=num_category_features, 
                   num_numerical=num_numerical_features, n=5, label=label, mode=3)
print("K_Prototypes算法的CUM值为：{}".format(CUM))
```
## 控制台打印内容：
```terminal
INFO--当前为第1次迭代  loss: 120
INFO--当前为第2次迭代  loss: 30
INFO--当前为第3次迭代  loss: 35
INFO--当前为第4次迭代  loss: 35
INFO--当前为第5次迭代  loss: 20
INFO--当前为第6次迭代  loss: 16
INFO--当前为第7次迭代  loss: 10
INFO--当前为第8次迭代  loss: 5
INFO--当前为第9次迭代  loss: 3
INFO--当前为第10次迭代 loss: 6
INFO--当前为第11次迭代 loss: 3
INFO--当前为第12次迭代 loss: 2
INFO--当前为第13次迭代 loss: 1
INFO--当前为第14次迭代 loss: 0
INFO--当前为第15次迭代 各类别的样本个数统计结果: [101  94  64  60  43  38]
最终的迭代次数为: 14
K_Prototypes算法的CUM值为：3.0573724471832033
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
| mode | int | 计算模式：1: K_Modes，2: K_Means，其他值: K_Prototypes |

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

## 数据地址：
链接: https://pan.baidu.com/s/1910YyLGiEXJlZjBMo3nbAg 提取码: s9ir
