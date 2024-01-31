from Model import Efficientnet
import torch
from tqdm import tqdm
import numpy as np

def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for imgs in tqdm(iter(test_loader)):
            imgs = imgs.float().to(device)

            probs = model(imgs)
            probs = torch.sigmoid(probs)

            probs  = probs.cpu().detach().numpy()
            predictions += probs.tolist()
    return predictions

def soft_voting(file_list, test_loader, device):
    final_pred = None
    
    for i in file_list:
        model = Efficientnet()
        model.load_state_dict(torch.load(f"model/{i}"))
        predictions = inference(model, test_loader, device)

        if final_pred is None:
            final_pred = np.array(predictions)
        else:
            final_pred += np.array(predictions)

    final_pred_soft_voting = final_pred * 0.2
    
    return final_pred_soft_voting
