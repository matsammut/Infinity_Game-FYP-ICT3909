import torch
import numpy as np
from torchvision.transforms import ToTensor
from pifuhd import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pifuhd = PIFuHD().to(device)