import sys, shutil, random, yaml
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from geneartor import SPADEGenerator
from dicriminator import MultiscaleDiscriminator
import argparse

input_semantics = torch.randn((1,12,512,512))
real_image = torch.randn((1,3,512,512))

parser = argparse.ArgumentParser(description='SEAN')
parser.add_argument('--semantic_nc', type=int, default=12)
opt = parser.parse_args()

netG = SPADEGenerator(opt)
netD = MultiscaleDiscriminator(opt)

fake_image = netG(input_semantics, real_image)
fake_concat = torch.cat([input_semantics, fake_image], dim=1)
real_concat = torch.cat([input_semantics, real_image], dim=1)
fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
discriminator_out = netD(fake_and_real)


print(fake_image.shape)
print(discriminator_out)