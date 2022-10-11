import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np


b = np.ones(((10, 10, 6)))


# c = torch.ones((20, 10, 10))
# abc = pad_sequence([a, b, c], batch_first=True, padding_value=0)
# print(abc)
b = np.transpose(b, axes=[2, 0, 1])
print(b.shape)
