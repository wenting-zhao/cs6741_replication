import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle
import numpy as np
from utils import metrics
from train import pad_to_longest

def test_epoch(model,test_data,opt):
    model.eval()
    all_predictions = torch.zeros(len(test_data.X))
    all_targets = torch.zeros(len(test_data.X))

    srcs = test_data.X
    tgts = test_data.y
    loss_total = 0.
    N = len(test_data.X)
    bsize = opt.batch_size 
    batches = list(range(0, N, bsize))

    for i in tqdm(batches, mininterval=0.5, desc="Testing", leave=False):
        src = srcs[i:i+bsize]
        tgt = tgts[i:i+bsize]
        src, lens, masks = pad_to_longest(src)
        src = torch.LongTensor(src).cuda()
        lens = torch.LongTensor(lens)
        masks = torch.BoolTensor(masks).cuda()
        tgt = torch.Tensor(tgt).cuda()


        with torch.no_grad():
            preds, attns = model(src, lens, masks)
            preds = preds.squeeze()

            loss = F.binary_cross_entropy_with_logits(preds, tgt)
            loss_total += loss.item()
            
            ## Updates ##
            start_idx, end_idx = i,i+bsize
            all_predictions[start_idx:end_idx] = F.sigmoid(preds).cpu().data
            all_targets[start_idx:end_idx] = tgt.cpu().data

    
    return all_predictions, all_targets, loss_total

