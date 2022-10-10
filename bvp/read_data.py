import scipy
import torch
import torch.nn as nn
import scipy.io as scio
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

# a = scio.loadmat(r"data\20181109-VS\6-link\user1\user1-1-1-1-2-1-1e-07-100-20-100000-L0.mat")
# print(a['velocity_spectrum_ro'].shape)
# # 20*20*T


d_model = 10  # 词嵌入的维度
hidden_size = 20  # lstm隐藏层单元数量
layer_num = 1  # lstm层数

# 输入inputs,维度为[batch_size,max_seq_len]=[3,4],其中0代表填充
# 该input包含3个序列，每个序列的真实长度分别为: 4 3 2

inputs = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 0], [1, 2, 0, 0]])

embedding = nn.Embedding(5, d_model)

# 获取词嵌入后的inputs 当前inputs的维度为[batch_size,max_seq_len,d_model]=[3,4,10]
inputs = embedding(inputs)

# 查看inputs的维度
print(inputs.size())
# print: torch.Size([3, 4, 10])

# 利用“压缩”函数对inputs进行压缩处理，[4,3,2]分别为inputs中序列的真实长度,batch_first=True表示inputs的第一维是batch_size
inputs = nn.utils.rnn.pack_padded_sequence(inputs, lengths=[4, 3, 2], batch_first=True)

# 查看经“压缩”函数处理过的inputs的维度
print(inputs[0].size())
# print: torch.Size([9, 10])


# 定义RNN网络
network = nn.LSTM(input_size=d_model, hidden_size=hidden_size, batch_first=True, num_layers=layer_num)
# 初始化RNN相关门参数
c_0 = torch.zeros((layer_num, 3, hidden_size))
h_0 = torch.zeros((layer_num, 3, hidden_size))  # [rnn层数,batch_size,hidden_size]

# inputs经过RNN网络后得到的结果outputs
output, (h_n, c_n) = network(inputs, (h_0, c_0))

# 查看未经“解压函数”处理的outputs维度
print(output[0].size())
# print: torch.Size([9, 20])

# 利用“解压函数”对outputs进行解压操作,其中batch_first设置与“压缩函数相同”，padding_value为0
output = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, padding_value=0)

# 查看经“解压函数”处理的outputs维度
print(output[0].size())
# print：torch.Size([3, 4, 20])
