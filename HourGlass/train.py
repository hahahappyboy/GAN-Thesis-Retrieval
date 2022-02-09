
import torch
import torch.nn as nn
from model import HourGlass
input = torch.randn((1,64,256,256))


model = HourGlass(64,64)

output = model(input)


print(output.shape)
