import torch
import os
a = []
for i in range(1000):
    t = torch.zeros((1000, 1000, 1000), dtype=torch.float).cuda()
    print(t.device)
    a.append(t)
os.system('nvidia-smi')
del a