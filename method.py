import torch
from utils import AsymmetricLoss
from config import CFG

def train(model, optimizer, train_loader, val_loader, device):
    model.to(device)
    criterion =  AsymmetricLoss().to(device)
    best_val_loss = float('inf')
    best_model = None

    for epoch in range(1, CFG['EPOCHS']+1):
        model.train()
        train_loss = []
        for imgs, labels in tqdm(iter(train_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(imgs)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        _val_loss, _val_precision = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val precision : [{_val_precision:.5f}]')

        if best_val_loss > _val_loss:
            best_val_loss = _val_loss
            best_model = model

    return best_model

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score

def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in tqdm(iter(val_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)

            probs = model(imgs)
            probs = torch.sigmoid(probs)
            loss = criterion(probs, labels)

            probs_numpy = probs.cpu().detach().numpy()
            labels_numpy = labels.cpu().detach().numpy()

            val_loss.append(loss.item())
            all_probs.append(probs_numpy)
            all_labels.append(labels_numpy)

        _val_loss = np.mean(val_loss)

    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    ap_scores = []
    for i in range(all_probs.shape[1]):
        ap = average_precision_score(all_labels[:, i], all_probs[:, i])
        ap_scores.append(ap)

    mean_ap = np.mean(ap_scores)

    return _val_loss, mean_ap
