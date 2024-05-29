import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
import torch.utils.data as Data
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features

seeds = 2023
if seeds > 0:
    torch.manual_seed(seeds)        #设置CPU生成随机数的种子,方便下次复现实验结果。
    np.random.seed(seeds)

"""-------------------------参数区-------------------------------"""
data_name = "5T_gap_1Y_LastYear_0.25"
test_data_name = "2007_5t"
# test_data_name = "outer_active_day"                       #"""test: 1:5T_gap_1Y_LastYear_0.25-0.5  2:5T_gap_1Y_0.25"""
# test_data_name = "LSTM_compare"                           #"""test: 3:outer_calm_day  4:outer_active_day  5:LSTM-compare"""
file_path = './data/atmosphere_density/' + data_name + '.csv'
test_file_path = './data/atmosphere_density/' + test_data_name + '.csv'

Model_name = 'N00'                          #要校正的模型————可选项：N00、N20、Jacchia-ROBERT、JB2008、GEODYN
Models = ['N00','N20','Jacchia-ROBERT','JB2008','GEODYN']
outer_para = ['AP','F10.7','F10.7_81']
if 'N00' == Model_name:
    cols = Models + outer_para
elif 'N20' == Model_name:
    cols = ['Jacchia-ROBERT','JB2008','GEODYN',Model_name] + outer_para
else:
    cols = [Model_name] + outer_para

input_size = len(cols)
batch_size = 32
seq_len = 216
label_len = 108
pre_len = 216
showing_len = 600      #密度展示点数
hidden_size = 2048
output_size = 1
num_epochs = 15
dropout_rate = 0.25
lr = 0.01
num_layers = 2


One_day_rmse = 2        #画图时一天画几个误差点
do_train = 0
do_pred = True
get_future = 1
# if get_future:
#     input_size = input_size+3

assert Model_name in Models
setting = data_name + '_' + Model_name + f"_nl{num_layers}_sl{seq_len}_ll{label_len}_pl{pre_len}" \
                                         f"_hs{hidden_size}_ne{num_epochs}_dr{dropout_rate}_input{input_size}"
checkpoints_path = './checkpoints/byd/' + setting + '.pth'

device=torch.device("cuda:0" if torch.cuda.is_available() else"cpu")
print("device= ",device)

"""-------------------------------函数与类区-----------------------------------"""
class LinearNet(nn.Module):
    def __init__(self, input_size = input_size, hidden_size = hidden_size,output_size = output_size,
                 dropout_rate = dropout_rate,num_layers = num_layers):
        super(LinearNet,self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fm = nn.Linear(hidden_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.num_layers = num_layers

    def forward(self,x):
        if self.num_layers <= 2:
            x = self.fc1(x)
            x = self.dropout(x)
            x = torch.relu(x)
            x = self.fc2(x)
        else:
            x = self.fc1(x)
            x = self.dropout(x)
            x = torch.relu(x)
            for i in range(self.num_layers - 2):
                x = self.fm(x)
                x = self.dropout(x)
                x = torch.relu(x)
            x = self.fc2(x)
        return x

"""
class Metrics():
    def __init__(self):
        self.mse = mean_squared_error
        self.mae = mean_absolute_error
        self.epsilon = np.mean(train.true_density) / 2
        self.epsilon = 0
        self.tol = 1.2  # 单个数据百分比误差上限120%

    def percentage_error(self, pred, true):
        r = (pred - true) / (true + self.epsilon)
        r[r > self.tol] = self.tol / 2
        return r

    def abs_percentage_error(self, pred, true):
        r = abs(self.percentage_error(pred, true))
        return r

    def mean_percentage_error(self, pred, true):
        mpe = np.mean(self.percentage_error(pred, true))
        return mpe

    def mape(self, pred, true):
        return np.mean(self.abs_percentage_error(pred, true))

    def smape(self, y_predict, y_true) -> float:  # [0,200%]
        sm = np.mean(np.abs(y_predict - y_true) / (np.abs(y_true + y_predict) / 2))
        return sm

    def rmse(self, pred, true):
        return np.sqrt(mean_squared_error(pred, true))

    def rmse_percent(self, pred, true):
        r = self.percentage_error(pred, true)
        return np.sqrt(np.mean((r) ** 2))

    def percent_rmse_true(self, rmse, test_mean):
        test_mean = np.mean(test_mean)
        return rmse / test_mean

    def mpe_std_sigma(self, pred, true):
        '''标准百分比误差，用于衡量预测值与真实值之间的离散程度'''
        r = self.percentage_error(pred, true)
        r_bar = np.mean(r)
        n = len(r)
        # return np.sqrt( np.sum( (r-r_bar)**2/(n-1) ) )
        return np.std(r)

    def mape_std_sigma(self, pred, true):
        '''用于衡量预测值与真实值之间的离散程度'''
        r = self.abs_percentage_error(pred, true)
        r_bar = np.mean(r)
        n = len(r)
        # return np.sqrt( np.sum( (r-r_bar)**2/(n-1) ) )
        return np.std(r)

    def improve_rate(self, n00_loss, lstm_loss):
        '''计算误差提升率，正常情况下n00_loss>lstm_loss，参数中大的放在前面'''
        improve_rate = (abs(n00_loss) - abs(lstm_loss)) / abs(n00_loss)
        return improve_rate

    def write_metrics_to_csv(self, pred, true, csv_filename):
        with open(csv_filename, 'a', newline='') as csvfi
"""


class MyDataSet(Data.Dataset):
    def __init__(self,data_x,data_y,get_future = get_future,
                 seq_len = seq_len,label_len = label_len,pred_len = pre_len):
        self.data_x = data_x
        self.data_y = data_y
        self.get_future = get_future
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        if s_end+self.seq_len < self.data_x.shape[0] and self.get_future:
            seq_x = self.data_x[s_begin:s_end,:input_size-3]
            seq_x1 = self.data_x[s_end:s_end+self.seq_len,-3:]
            seq_x = np.concatenate((seq_x,seq_x1),axis=1)
        else:
            seq_x = self.data_x[s_begin:s_end]

        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

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
    for i in range(rounds):
        y_label = label[i * num:(i+1) * num - 1]
        y_pre = prediction[i * num:(i+1) * num - 1]
        rmse1 = rmse_cal(y_label,y_pre)
        rmse.append(rmse1)
    return rmse

def calculate_Ri(label,preds):
    assert label.size == preds.size
    R = np.zeros([label.size,1])
    for i in range(label.size):
        R[i] = preds[i]/label[i] - 1
        if R[i] > 1.2:
            R[i] = 1.2

    avg = R.mean()
    # sjwc = np.sqrt(np.sum((R-avg) ** 2) / (R.size - 1))
    sjwc = np.std(R)
    return avg,sjwc



net = LinearNet()
net.to(device)

if do_train:
    df_raw = pd.read_csv(file_path)
    df_y = df_raw.loc[:,'Grace-A']
    df_raw = df_raw[cols]
    # df_raw = df_raw.drop(columns=['date','Grace-A','timestamp'])
    # df_raw = df_raw.drop(columns=['date', 'Grace-A', 'timestamp','AP','F10.7','F10.7_81'])
    scaler_x,scaler_y = StandardScaler(),StandardScaler()
    data_x = scaler_x.fit_transform(df_raw.values.reshape(-1,len(cols)))
    data_y = scaler_y.fit_transform(df_y.values.reshape(-1,1))
    data_x = torch.tensor(data_x)
    data_y = torch.tensor(data_y)
    dataset = Data.TensorDataset(data_x,data_y)
    dataloader = Data.DataLoader(dataset,batch_size=batch_size,drop_last=True,shuffle=True)

    loss = nn.MSELoss()
    loss.to(device)
    lmin,l = 10,0
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        for step,(x,y) in enumerate(dataloader):
            x = x.float().to(device)
            y = y.float().to(device)
            optimizer.zero_grad()
            pre = net(x)
            l = loss(pre, y)
            l.backward()
            optimizer.step()
        if l<lmin:
            lmin = l
            torch.save(net.state_dict(), checkpoints_path)
            print(f"第{epoch+1}次,loss= {l:.4f} Model Save!")
        else:
            print(f"第{epoch+1}次,loss= {l:.4f}")

    # torch.save(net.state_dict(), checkpoints_path)

if do_pred:
    if not do_train:
        net.load_state_dict(torch.load(checkpoints_path))

    df_raw = pd.read_csv(test_file_path)
    df_y = df_raw.loc[:, 'Grace-A']
    df_raw = df_raw[cols]
    # df_raw = df_raw.drop(columns=['date','Grace-A','timestamp'])
    # df_raw = df_raw.drop(columns=['date', 'Grace-A', 'timestamp', 'AP', 'F10.7', 'F10.7_81'])
    scaler_x, scaler_y = StandardScaler(), StandardScaler()
    data_x = scaler_x.fit_transform(df_raw.values.reshape(-1,len(cols)))
    data_y = scaler_y.fit_transform(df_y.values.reshape(-1,1))
    data_x = torch.tensor(data_x)
    data_y = torch.tensor(data_y)
    dataset = Data.TensorDataset(data_x, data_y)
    dataloader = Data.DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=False)

    net.eval()
    preds = []
    with torch.no_grad():
        for step, (x, y) in enumerate(dataloader):
            x = x.float().to(device)
            pre = net(x)
            preds.append(pre)

    preds = torch.cat(preds,dim=0)
    # preds = preds[:,-1,0].view(-1,output_size).to('cpu').detach().numpy()
    preds = preds.view(-1, output_size).to('cpu').detach().numpy()
    Target_den = df_raw.loc[:preds.size-1,Model_name].values.reshape(-1,1)
    true_den = df_y.loc[:preds.size-1].values.reshape(-1,1)
    preds = scaler_y.inverse_transform(preds)
    if 'outer' in test_data_name:
        print(Model_name + " rmse = ", rmse_cal(Target_den[0:280], true_den[0:280]))
        print("pre rmse = ", rmse_cal(preds[0:280], true_den[0:280]))

    print(Model_name + " rmse = ", rmse_cal(Target_den, true_den))
    print("pre rmse = ", rmse_cal(preds, true_den))

    tar_R = np.corrcoef(Target_den.reshape(1,-1), true_den.reshape(1,-1))
    pred_R = np.corrcoef(preds.reshape(1,-1),true_den.reshape(1,-1))
    print(Model_name + " 相关系数R = ",tar_R[0][1])
    print("pre 相关系数R = ",pred_R[0][1])

    R, cita = calculate_Ri(true_den, Target_den)
    print(Model_name + " R = ", R*100, " sita = ", cita*100)
    R, cita = calculate_Ri(true_den, preds)
    print("prediction R = ", R*100, " sita = ", cita*100)

    Tar_rmse = make_rmse_list(true_den, Target_den, One_day_rmse)
    preds_rmse = make_rmse_list(true_den, preds, One_day_rmse)
    days = range(0, len(Tar_rmse))
    days = [i / One_day_rmse for i in days]
    np.save("./data/atmosphere_density/npy/" + Model_name + test_data_name + '.npy',Tar_rmse)
    np.save(f"./data/atmosphere_density/npy/DLinear_{Model_name}_" + test_data_name + '.npy',preds_rmse)
    plt.figure()
    plt.plot(days, Tar_rmse, label=Model_name + '_rmse', color='g')
    plt.plot(days, preds_rmse, label='prediction_rmse', color='r')
    plt.xlabel('DAYS')
    plt.ylabel('RMSE/ (1e-12*(kg·m^-3))')
    plt.legend(loc='upper right')
    plt.title("rmse present")
    plt.show()

    plt.figure()
    min = range(0,showing_len*5,5)
    plt.plot(preds,label='prediction',color='blue')
    plt.plot(Target_den,label=Model_name,color='green')
    plt.plot(true_den,label="Grace-A",color='r')
    plt.ylabel('Density/ (1e-12*(kg·m^-3))')
    plt.xlabel('minute')
    plt.legend(loc='upper right')
    plt.title("density present")
    plt.show()

