import random
import pandas as pd
import numpy as np
import os
from PIL import Image
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from Data_loader import get_data_loader, get_test_loader
from multiprocessing import freeze_support
from torchvision import transforms
import torchvision.models as models
from method import train
from tqdm.auto import tqdm
from Soft_voting import soft_voting
import warnings
warnings.filterwarnings(action='ignore')

from kfold import kfold
from Model import Efficientnet
from config import CFG


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정




# data load
df = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# make Kfold
kfold(df)


def main():
    file_name = sorted(os.listdir('fold_data'))

    for i in range(5):
        val_df = pd.read_csv(f"fold_data/{file_name[2*i]}")
        train_df = pd.read_csv(f"fold_data/{file_name[2*i+1]}")
        
        train_loader, val_loader = get_data_loader(train_df, val_df)
        
        model = Efficientnet()

        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                    first_cycle_steps=200,
                                                    cycle_mult=1.0,
                                                    max_lr=0.005,
                                                    min_lr=0.0001,
                                                    warmup_steps=100,
                                                    gamma=1.0)


        infer_model = train(model, optimizer, train_loader, val_loader, device)
        torch.save(infer_model.state_dict(), f'model_cv{i}.pth')
    
    test_loader = get_test_loader(test)
    
    final_pred_voting = soft_voting(os.listdir('model'), test_loader)
    
if __name__ == '__main__':
    freeze_support()
    main()
