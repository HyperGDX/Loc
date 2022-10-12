import torch

a = torch.FloatTensor([[1.0, 2.0]])
b = torch.FloatTensor([[3.0, 4.0]])

c = torch.tensor([])
c = torch.cat((a, b), dim=0)
print(c)
