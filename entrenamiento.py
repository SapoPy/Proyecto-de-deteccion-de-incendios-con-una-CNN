import os
import sys
import time

import numpy as np
import torch
from torch import nn
import torchvision
from matplotlib import pyplot as plt

from utils import *
from fireyecnn import *

from dataset import *


fireeye_train = FireEyeDataset(DIR_TRAIN, transform=transform_fireeye)
fireeye_test = FireEyeDataset(DIR_TEST,transform=transform_fireeye)

lr = 5e-4
dropout_p = 0.5
batch_size = 4
criterion = nn.CrossEntropyLoss()

epochs = 30

model = CNNModel(dropout_p)

curves = train_model(
    model,
    fireeye_train,
    fireeye_test,
    epochs,
    criterion,
    batch_size,
    lr,
    use_gpu=False,
    data_augmentation=False,
)

show_curves(curves)