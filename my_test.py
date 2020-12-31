import torch

a = torch.FloatTensor([[1, 1, 1], [2, 2, 2]])
b = torch.FloatTensor([[3, 3, 3], [4, 4, 4]])
c = torch.FloatTensor([[5, 5, 5], [6, 6, 6]])
res = torch.cat([a, b, c], dim=1)
print(res.size())
print(res)
