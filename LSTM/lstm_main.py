import numpy as np
import torch
from lstm_mdl import *
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
import torch.utils.data as Data

# --------模型参数调整区----------#
num_layers = 2      #LSTM层堆叠数量，注意LSTM层时间步数与单次输入的向量长度有关        
hidden_size = 128    #隐藏的h与c向量维度
input_size = 4      #输入的向量维度、包含经验模型(JR)值，F10.7，F10.7A，ap指数
output_size = 1     #输出的向量维度
sampledist = 2      #确保60s采一个点
time_delay = 200    #即序列长度是200
batch_size = 30      #批大小为30
train_ratio = 0.8   #训练集大小占整个数据集的比例，剩下的是验证集
learning_rate = 0.01
num_epoches = 40   #训练轮数
true_col = 1
JR_col = 2
AP_col = 8
F107_col = 9
F107A_col = 10
datapath = './atmosphere_data/new/Trainset.csv'

if __name__ == '__main__':
    # 看gpu是否可用
    device=torch.device("cuda:0" if torch.cuda.is_available() else"cpu")
    print("device= ",device)
    # 三步走
    # step1. 加载数据并做好归一化等工作，同时把数据化为相应形状的张量，并加载进内存，供gpu调度
    # step2. 实例化模型，定义好训练迭代次数，优化器，误差函数，开始训练
    # step3. 结束训练后开始预测，并做好绘图工作(或导出数据，交给matlab绘图)，同时可以注意保存好模型参数数据

    #step1----------
    #注意数据标准化&对密度值取log

    #读取数据，并根据采样率进行样本选取
    rawData = pd.read_csv(datapath)

    raw_true_density = rawData.iloc[:,true_col].values[0::sampledist].reshape(-1,1) #选取特定列并转为numpy格式，同时进行采样
    raw_JR_density = rawData.iloc[:,JR_col].values[0::sampledist].reshape(-1,1)
    raw_AP_index = rawData.iloc[:,AP_col].values[0::sampledist].reshape(-1,1)
    raw_F107 = rawData.iloc[:,F107_col].values[0::sampledist].reshape(-1,1)
    raw_F107L = rawData.iloc[:,F107A_col].values[0::sampledist].reshape(-1,1)
    # raw_N00_density = rawData.iloc[:,2].values[0::sampledist]
    dataset_size = raw_true_density.size
    #上面步骤做完后，得到4个形状为(size_of_data,1)的单独特征向量和1个形状相同的标签向量，接下来要把这4个单独的特征向量合并成矩阵

    #开始进行数据变换，朝输入与输出向量进行变换，并最后转变成张量
    #记住，一个可以被lstm层训练的输入序列长度是200，对应的，经过整个lstm模型最后的输出长度只有1

    #在此之前，先进行标准化。
    #true_density的标准化
    raw_true_density = np.log(raw_true_density.astype('float32')) #密度要先做对数运算，防止数值过小
    true_density_mean = np.mean(raw_true_density)
    true_density_std = np.std(raw_true_density)
    true_density = (raw_true_density - true_density_mean)/true_density_std

    #JR_density的标准化
    raw_JR_density = np.log(raw_JR_density.astype('float32'))
    JR_density = (raw_JR_density - np.mean(raw_JR_density))/np.std(raw_JR_density)

    #AP指数的标准化
    AP = (raw_AP_index - np.mean(raw_AP_index))/np.std(raw_AP_index)

    #F107的标准化
    F107 = (raw_F107 - np.mean(raw_F107))/np.std(raw_F107)

    #F107L的标准化
    F107L = (raw_F107L - np.mean(raw_F107L))/np.std(raw_F107L)

    #再把所有特征合并在一个向量中，并组成一个矩阵(size_of_data,4)
    raw_dataset = np.concatenate((JR_density,F107,F107L,AP),axis = 1)
    raw_dataset = raw_dataset.astype('float32')

    #接下来开始生成训练用的数据集，记住，时间步数为200，故会生成(size_of_data-200,200,4)的输入数据集，以及(size_of_data-200,1,1)的标签
    lst_for_stack = []
    for i in range(dataset_size-time_delay):
        lst_for_stack.append(raw_dataset[i:i+time_delay,:])
    
    dataset = np.stack(lst_for_stack)
    lableset = true_density[time_delay:,:].reshape(-1,1,1)
    
    #划分训练集与验证集
    divisionPnt = int(dataset_size*train_ratio)
    trainset = dataset[:divisionPnt]
    trainlable = lableset[:divisionPnt]
    validationset = dataset[divisionPnt:]
    validationlable = lableset[divisionPnt:]


    #step2----------
    #开始定义模型，优化器，损失函数，以及可用的数据集
    #注意要把这些实体数据都转移到gpu上，才能利用gpu做运算
    #要使用早停法进行训练
    
    train_tensors = Data.TensorDataset(torch.tensor(trainset),torch.tensor(trainlable))
    valid_tensors = Data.TensorDataset(torch.tensor(validationset),torch.tensor(validationlable))
    trainset_loader = Data.DataLoader(dataset=train_tensors, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True) #num_workers=2，用两个线程并行加载数据
    validset_loader = Data.DataLoader(dataset=valid_tensors, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

    lstm_model = lstmNN(input_size,hidden_size,output_size,num_layers)
    lstm_model = lstm_model.to(device)
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate) #这里会自动把优化器创建在gpu上
    loss_function = nn.MSELoss()
    loss_function = loss_function.to(device)

    #注意，这里仍然与论文的处理有差别，论文的梯度更新是放在一轮epoch完成之后进行的
    #若想完全复刻论文，则需要把预测的y与真值y_t都暂存起来，之后一轮epoch完成后，再计算损失函数，同时进行梯度更新
    best_loss = float('inf')
    #这里的早退阈值是后来敲定的
    # threshold = 56
    threshold = 70
    for epoch in range(num_epoches):
        # 先做一个epoch的训练
        for times, (for_input, for_lable) in enumerate(trainset_loader):
            #把数据转移到gpu
            for_input = for_input.to(device)
            for_lable = for_lable.to(device)
            pred = lstm_model(for_input)
            #这里pred形状为(batch_size,1)
            for_lable = for_lable.view(batch_size,output_size)
            loss = loss_function(pred,for_lable)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # 开始利用验证集检验误差，若误差在一定阈值内开始呈现上升趋势，就退出训练
        sum_loss = 0
        with torch.no_grad():
            for times, (valid_input, valid_lable) in enumerate(validset_loader):
                valid_input = valid_input.to(device)
                valid_lable = valid_lable.to(device)
                pred = lstm_model(valid_input)
                valid_lable = valid_lable.view(batch_size,output_size)
                sum_loss += loss_function(pred,valid_lable).item()
        if epoch%10 == 0:
            print(f"Epoch {epoch + 1}/{num_epoches}, Train Loss: {loss:.4f}, Best Valid Loss: {best_loss:.4f}, Current Loss: {sum_loss:.4f}")
        if sum_loss < best_loss:
            best_loss = sum_loss
            print(f"Epoch {epoch+1}, Current Best Loss: {best_loss:.4f}")
            torch.save(lstm_model.state_dict(),'./new_l2_h128_b30_N00_ES1.pth')
            # if best_loss < threshold:
            #     break
        elif best_loss < threshold:
            print(f"Final Loss: {sum_loss:.4f}")
            break
        # else :
        #     print(f"Final Loss: {sum_loss:.4f}")
        #     break







    #模型的保存与载入
    # torch.save(lstm_model.state_dict(),'./model_l2_h128_b30_ES.pth')

    

    # lstm_model = lstmNN(input_size,hidden_size,output_size,num_layers)
    # lstm_model.load_state_dict(torch.load('./model_l2_h128_b30.pth'))

