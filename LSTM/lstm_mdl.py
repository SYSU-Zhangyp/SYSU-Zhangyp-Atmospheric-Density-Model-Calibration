import torch
from torch import nn
import torch.utils.data as Data


'''
改进思路：在进入堆叠的LSTM单元之前，经过一个全连接的三层线性层，之后再做一个残差连接
'''


class lstmNN(nn.Module):
    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()

        self.feas0 = nn.Linear(input_size, input_size*4)
        self.feas = nn.Linear(input_size*4, input_size)
        self.drop_ = nn.Dropout(p = 0.4)
        self.relu = nn.ReLU()
        #可以设置归一化层，但尚不清楚归一化层的机制

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
        #注意，batchfirst设置为true之后，输入与输出向量的维度变换为(batch,seq_len,hidden_size)
        self.fc_h2finaloutput = nn.Linear(hidden_size, output_size)

    def forward(self,_x):
        #这里只接收最后的输出h，丢弃c
        x = self.drop_(self.feas(self.feas0(_x)))
        x = x + _x
        x = self.relu(x)

        x, _ = self.lstm(x)  # _x is input, size (seq_len, batch, input_size),now size(batch,seq_len,input_size)
        #解读_x:第几批的第几个时间步的样本(input_size维)的第几个特征
        # x拥有所有时间步的隐藏状态，而不仅仅只有最后一个时间步的
        '''
        #这里决定只用最后一个h，所以不保留这一部分代码
        b, s, h = x.shape  # x is output, size (batch_size,seq_len, hidden_size),解读：第几批的第几个时间步的h(hidden_size维)的第几个元素
        x = x.view(s*b, h)
        x = self.fc_h2finaloutput(x)
        x = x.view(b, s, -1)
        return x
        '''
        #经过线性全连接层，把输出映射成维度是输出维度的最终结果
        #最后的x形状是(batch_size,output_size)
        x = self.fc_h2finaloutput(x[:,-1,:])
        x = x.view(x.shape[0]*x.shape[1], -1)
        return x


# class lstmNN(nn.Module):
#     def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
#         super().__init__()

#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
#         #注意，batchfirst设置为true之后，输入与输出向量的维度变换为(batch,seq_len,hidden_size)
#         self.fc_h2finaloutput = nn.Linear(hidden_size, output_size)

#     def forward(self,_x):
#         #这里只接收最后的输出h，丢弃c
#         x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size),now size(batch,seq_len,input_size)
#         #解读_x:第几批的第几个时间步的样本(input_size维)的第几个特征
#         # x拥有所有时间步的隐藏状态，而不仅仅只有最后一个时间步的
#         '''
#         #这里决定只用最后一个h，所以不保留这一部分代码
#         b, s, h = x.shape  # x is output, size (batch_size,seq_len, hidden_size),解读：第几批的第几个时间步的h(hidden_size维)的第几个元素
#         x = x.view(s*b, h)
#         x = self.fc_h2finaloutput(x)
#         x = x.view(b, s, -1)
#         return x
#         '''
#         #经过线性全连接层，把输出映射成维度是输出维度的最终结果
#         #最后的x形状是(batch_size,output_size)
#         x = self.fc_h2finaloutput(x[:,-1,:])
#         x = x.view(x.shape[0]*x.shape[1], -1)
#         return x