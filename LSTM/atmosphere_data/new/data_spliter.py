import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 先做数据融合，再做插值，最后做采样
# data = pd.read_csv('./atmosphere_data/new/to070129.csv').iloc[92160:178560,:]

# data1 = pd.read_csv('./atmosphere_data/new/to060730.csv').iloc[263520:,:]

data2 = pd.read_csv('./atmosphere_data/new/to070129.csv').iloc[267840:,:]

data3 = pd.read_csv('./atmosphere_data/new/to070730.csv').iloc[:7200,:]

data = pd.concat([data2, data3])
# data = pd.read_csv('./atmosphere_data/new/to010129.csv')
# data = data.iloc[1440:4321,:]

true_den_col = 1
true_den_name = data.columns[1]
data.iloc[data.iloc[:,true_den_col]<0,true_den_col] = np.nan

data[true_den_name] = data[true_den_name].interpolate(method='polynomial',limit_direction='forward',order=2)

# 这里是用来查找负值用的
raw_true_density = data.iloc[:,1].values.reshape(-1,1)
raw_JR_density = data.iloc[:,4].values.reshape(-1,1)
print('数值检测-------')
ctr = 0
for num in range(raw_true_density.size):
    if(isinstance(raw_true_density[num][0],str)):
        print("True:",raw_true_density[num])
        continue
    if raw_true_density[num]<=0 or raw_JR_density[num]<=0:
        print(type(raw_true_density[num][0]))
        print('当前是第',end='')
        print(num,end='')
        print('组数据，')
        print('TRUE:',raw_true_density[num])
        print('JR:',raw_JR_density[num])
        ctr +=1
print('检测结束------')
print(f"非法数据总量: {ctr}")


if ctr>0:
    data.iloc[data.iloc[:,true_den_col]<0,true_den_col] = np.nan
    data[true_den_name] = data[true_den_name].interpolate(method='linear',limit_direction='both')

data = data.iloc[::2]
data.to_csv('./atmosphere_data/new/Test2.csv',index=False)


# data = pd.read_csv('./atmosphere_data/new/to070129.csv')
# data = data.iloc[:178560,:]
x = range(data.iloc[:,1].values.size)
plt.plot(x,data.iloc[:,1].values)
plt.show()
