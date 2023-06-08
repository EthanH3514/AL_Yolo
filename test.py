import torch
import os

print(torch.cuda.device_count())
print(torch.cuda.is_available())

print(os.getcwd())