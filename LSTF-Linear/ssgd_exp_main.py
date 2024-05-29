import pandas as pd
import numpy as np

# data = pd.read_csv("./data/atmosphere_density/Atmosphere_data.csv",engine='python')
# rate = 10
# data = data[::rate]

data = pd.read_csv("./data/atmosphere_density/5T_gap.csv",engine='python')
# data = pd.read_csv("./data/atmosphere_density/5T_gap_1Y.csv",engine='python')
data = data.fillna(data.mean())             # 用列的平均值填充
# data = data.dropna(axis=0)                # 删除有空的行
# data.fillna(method = 'backfill', axis = 0) # 将通过前向填充 (ffill) 方法用同一列的后一个数作为填充
data = data[525600:630720]
data.replace([np.inf, -np.inf], np.nan, inplace=True)

print("nan left? ",np.any(data.isnull()))  # 只要有一个空值便会返回True，否则返回False

data.to_csv("./data/atmosphere_density/5T_gap_1Y.csv",index=None)