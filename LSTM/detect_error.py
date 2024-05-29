import pandas as pd
import numpy as np

true_den_col = 1
dataset = pd.read_csv('./atmosphere_data/new/Test1.csv')
'''
# dataset = dataset[dataset['Grace-A'] > 0]
# 准备做插值处理
# 这里用布林操作批量修改，但会有漏网之鱼(其实是后面的三次样条插值导致又出现了负值)
true_den_name = dataset.columns[1]
dataset.iloc[dataset.iloc[:,true_den_col]<=0,true_den_col] = np.nan


# 对真实值列进行插值
dataset[true_den_name] = dataset[true_den_name].interpolate(method='linear',limit_direction='both')
# 结果保存至新csv文件
dataset.to_csv('./atmosphere_data/new/Trainset.csv',index=False)
'''

# 这里是用来查找负值用的
raw_true_density = dataset.iloc[:,1].values.reshape(-1,1)
raw_JR_density = dataset.iloc[:,4].values.reshape(-1,1)

print(type(raw_true_density[0][0]))
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


'''
#遍历并修改某列元素
for index, row in dataset.iterrows():
    value = row[true_den_name]
    
    # 进行某些操作，例如修改某个特定条件下的元素
    if value <= 0:
        dataset.at[index, true_den_name] = np.nan
'''