from lstm_mdl import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as Data
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

'''专门用来测试并保存测试误差的脚本'''
'''首先导入另一个脚本准备好的数据'''
'''之后再将模型导入并进行测试'''
'''在这个过程中需要将数据进行分批处理，不然会爆显存'''
def density_rmse(std:float,mean:float,y_prime:np.ndarray,y:np.ndarray) -> float:
    y_prime = y_prime*std + mean
    y_prime = np.exp(y_prime)
    diff = y_prime - y
    mse = np.mean(diff**2)
    rmse = np.sqrt(mse)
    return rmse
def density_R(std:float,mean:float,pred:np.ndarray,true:np.ndarray) -> float:
    pred = pred*std + mean
    pred = np.exp(pred)
    pred_mean = np.mean(pred)
    true_mean = np.mean(true)
    pred_unit = pred - pred_mean
    true_unit = true - true_mean
    R = np.sum(pred_unit*true_unit)/(np.sqrt(np.sum(pred_unit**2)*np.sum(true_unit**2)))
    return R

One_day_rmse = 4    #每天生成多少个rmse点
def rmse_cal(label,prediction):
    mse = np.sum((label - prediction) ** 2) / len(label)
    rmse = np.sqrt(mse)
    return rmse
def make_rmse_list(label,prediction,day_point = One_day_rmse):
    if label.size != prediction.size:
        return -1
    label = label.reshape(-1,1)
    prediction = prediction.reshape(-1,1)
    num = int(1440 / day_point)
    rounds = int(label.size / num)
    rmse = []
    rmse1 = 0
    for i in range(rounds):
        y_label = label[i * num:(i+1) * num - 1]
        y_pre = prediction[i * num:(i+1) * num - 1]
        rmse1 = rmse_cal(y_label,y_pre)
        rmse.append(rmse1)
    return rmse

#----------基础参数区
input_size = 4
hidden_size = 128
output_size = 1
num_layers = 2
time_delay = 200
data_path = './atmosphere_data/new/Test4.csv'
sample_dist = 2

device = torch.device("cuda:0" if torch.cuda.is_available() else"cpu")
print(device)

true_den_col = 1
JR_col = 2
AP_col = 8
F107_col = 9
F107A_col = 10
chunk_end = 8000
#载入数据
rawData = pd.read_csv(data_path)

# rawData = rawData.iloc[:chunk_end+1]
raw_true_density = rawData.iloc[:,true_den_col].values.reshape(-1,1) #选取特定列并转为numpy格式，同时进行采样，这里把采样的工作交给其他文件
raw_JR_density = rawData.iloc[:,JR_col].values.reshape(-1,1)
raw_AP_index = rawData.iloc[:,AP_col].values.reshape(-1,1)
raw_F107 = rawData.iloc[:,F107_col].values.reshape(-1,1)
raw_F107L = rawData.iloc[:,F107A_col].values.reshape(-1,1)

dataset_size = raw_true_density.size

log_true_density = np.log(raw_true_density.astype('float32')) #密度要先做对数运算，防止数值过小
true_density_mean = np.mean(log_true_density)
true_density_std = np.std(log_true_density)
true_density = raw_true_density

#JR_density的标准化
JR_density = np.log(raw_JR_density.astype('float32'))
JR_density_mean = np.mean(JR_density)
JR_density_std = np.std(JR_density)
JR_density = (JR_density - JR_density_mean)/JR_density_std

#其余特征的标准化
AP = (raw_AP_index - np.mean(raw_AP_index))/np.std(raw_AP_index)
F107 = (raw_F107 - np.mean(raw_F107))/np.std(raw_F107)
F107L = (raw_F107L - np.mean(raw_F107L))/np.std(raw_F107L)

#再把所有特征合并在一个向量中，并组成一个矩阵(size_of_data,4)
raw_dataset = np.concatenate((JR_density,F107,F107L,AP),axis = 1)
raw_dataset = raw_dataset.astype('float32')

#接下来开始生成训练用的数据集，记住，时间步数为200，故会生成(size_of_data-200,200,4)的输入数据集，以及(size_of_data-200,1,1)的标签
lst_for_stack = []
for i in range(dataset_size-time_delay):
    lst_for_stack.append(raw_dataset[i:i+time_delay,:])

testset = np.stack(lst_for_stack).reshape(-1,time_delay,input_size)
lableset = true_density[time_delay:,:].reshape(-1,1,1)
raw_JR_density = raw_JR_density[time_delay:,:].reshape(-1,1)

#载入模型
lstm_model = lstmNN(input_size,hidden_size,output_size,num_layers)
lstm_model = lstm_model.to(device)
lstm_model.load_state_dict(torch.load('./new_l2_h128_b30_N00_ES1.pth'))


#注意数据逆标准化
lstm_model = lstm_model.eval()  #转成评估模式(evaluation)，专门用来验证，关闭了批处理

testset = torch.tensor(testset)
# 一口气塞不下去
# pred = lstm_model(testset)
testset = Data.TensorDataset(testset)
# 这里不能用num_work，因为不在if __name__ == '__main__'模块里
data_loader =  Data.DataLoader(dataset=testset,batch_size=20,shuffle=False)
pred = []
# 这里可以尝试做一做dataloader，要不然太慢了
for itrs, eachbatch in enumerate(data_loader):
    eachbatch = eachbatch[0].to(device)
    pred_tensor = lstm_model(eachbatch).view(-1,1).cpu().detach().numpy()
    pred.append(pred_tensor)

# 一定要确保形状一致！！！否则会触发广播机制，轻则结果出错，重则爆内存
pred = np.concatenate(pred).reshape(-1,1)
lableset = lableset.reshape(-1,1)
print('ss')
# 这里还没有计算JR的rmse，以及相关系数R
'''
pred_rmse = density_rmse(true_density_std,true_density_mean,pred,lableset)
print("pred_rmse:",pred_rmse)
JR_rmse = density_rmse(JR_density_std,JR_density_mean,raw_JR_density,lableset)
print("JR_rmse:",JR_rmse)
'''

pred_R = density_R(true_density_std,true_density_mean,pred, lableset)
JR_R = density_R(JR_density_std,JR_density_mean,raw_JR_density, lableset)
print("pred_R:",pred_R)
print("JR_R:", JR_R)


pred = pred * true_density_std + true_density_mean
pred = np.exp(pred)
# JR_density = raw_JR_density * JR_density_std + JR_density_mean
# JR_density = np.exp(JR_density)
JR_density = raw_JR_density

print("prediction rmse = ", rmse_cal(pred, lableset))
print("N00 rmse = ", rmse_cal(JR_density, lableset))
pre_rmse = make_rmse_list(lableset, pred)
JR_rmse = make_rmse_list(lableset,JR_density)
days = range(0, len(JR_rmse))
days = [i / One_day_rmse for i in days]
plt.plot(days, JR_rmse, label='JB-2008_rmse', color='g')
plt.plot(days, pre_rmse, label='prediction_rmse', color='r')
plt.xlabel('DAYS')
plt.ylabel('RMSE/ (1e-12*(kg·m^-3))')
plt.legend(loc='upper right')
plt.title("density present")
plt.show()










'''展示时域图像'''

displaypts = pred.size
strtpts = 0
pred = pred[strtpts:strtpts+displaypts]
lableset = lableset[strtpts:displaypts+strtpts]
# pred = pred * true_density_std + true_density_mean
# pred = np.exp(pred)


t = np.arange(start=0,stop=displaypts)
plt.plot(t,pred,label='JB-2008C',color='b')
plt.plot(t,lableset,label='true_value',color='r')
plt.plot(t,JR_density,label='JB-2008',color='g')
plt.legend(loc='best')
plt.xlabel('MINUTES')
plt.ylabel('Density/ (1e-12*(kg·m^-3))')
plt.show()


print("foo bar")


