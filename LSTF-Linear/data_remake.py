import pandas as pd
import numpy as np

# data = pd.read_csv("./data/atmosphere_density/Atmosphere_data.csv",engine='python')
# rate = 10
# data = data[::rate]

data = pd.read_csv("./data/atmosphere_density/5T_gap.csv",engine='python')
data.replace([np.inf, -np.inf], np.nan, inplace=True)
# data = pd.read_csv("./data/atmosphere_density/1t_gap.csv",engine='python')

#data[data < 0] = np.nan  # 对过滤出来的对象进行赋值替换


# # data = data[525600:630720]        # 5T_gap_1Y
# # data = data[420480:525600]
# # data = data[315280:420480]
# data = data[:1 * int(data.shape[0] / 3)]


# data = data[44926:45214]             #active day
# data = data[102814:103102]           # outer_calm_day
# data = data[1051200:1138800]       #1t间隔，每个月43800点

# data = data[45502:53566]
data = data[464834:569953]
data.iloc[data.iloc[:,1]<=0,1] = np.nan             # 删除有空的行
data = data.dropna(axis=0)

print("nan left? ",np.any(data.isnull()))  # 只要有一个空值便会返回True，否则返回False

data.to_csv("./data/atmosphere_density/2007_5t.csv",index=None)