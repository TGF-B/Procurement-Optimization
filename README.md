# Stress Detection With BernouliNB()
最近很流行玩人格测试，通过做一些题目然后判断出自己属于哪种人格，以便更快地找到适合自己的职业或者对象。在当下的快节奏社会，这确实是一个方便又有趣的测试方式。受此启发，我搭建一个分析情绪的预测模型，通过分析社交平台上的评论内容文本，来判断一下发帖人的情绪倾向。
```python
import pandas as pd
import numpy as np
data=pd.red_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/stress.csv")
data.head()
```
