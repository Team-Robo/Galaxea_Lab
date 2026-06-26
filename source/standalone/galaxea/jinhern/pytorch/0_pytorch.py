import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"

# print(type(device))

###--------------tensor---------------------###
x = torch.tensor([1, 2, 3], dtype=torch.float32)
print(x)

###--------------ARRAY---------------------###
# array = np.array([1, 2, 3], dtype=np.float32)
# print(array)

w = torch.randn(3, 4, requires_grad=True)
print(w)

y = x + 2
print(y)
z = x @ w
print(z)