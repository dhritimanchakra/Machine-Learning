import torch
import numpy as np
from tqdm import tqdm 
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

@torch.no_grad()
def test(hn,hf,dataset,chunk_size=10,img_index=0,nb_bins=192,H=400,W=400):