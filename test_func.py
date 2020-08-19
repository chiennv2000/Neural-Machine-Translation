import torch

a = torch.tensor([[1], [2], [3]])
temp = a.expand(3, 2)
print(temp)