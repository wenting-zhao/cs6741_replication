from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle
import numpy as np
from utils import metrics
from sklearn.metrics import f1_score

def get_sorting_index_with_noise_from_lengths(lengths, noise_frac) :
    if noise_frac > 0 :
        noisy_lengths = [x + np.random.randint(np.floor(-x*noise_frac), np.ceil(x*noise_frac)) for x in lengths]
    else :
        noisy_lengths = lengths
    return np.argsort(noisy_lengths)

def pad_to_longest(data):
    lengths, expanded, masks = [], [], []
    maxlen = max([len(x) for x in data])
    for x in data:
        rem = maxlen - len(x)
        tmpp = [0]*rem
        expanded.append([*x, *tmpp])
        lengths.append(len(x))
        tmp = [1]
        tmp1 = [0]*(len(x)-2)
        tmp2 = [1]*(rem+1)
        masks.append([*tmp, *tmp1, *tmp2])
    return expanded, lengths, masks

def train_epoch(model,train_data,optimizer,attn_optimizer,epoch,opt):
#def train_epoch(model,train_data,optimizer,epoch,opt):
    model.train()

    all_predictions = torch.zeros(len(train_data.X))
    all_targets = torch.zeros(len(train_data.X))

    
    data_in = train_data.X
    target_in = train_data.y
    sorting_idx = get_sorting_index_with_noise_from_lengths([len(x) for x in data_in], noise_frac=0.1)
    data = [data_in[i] for i in sorting_idx]
    target = [target_in[i] for i in sorting_idx]

    bsize = opt.batch_size
    N = len(data)
    loss_total = 0
    loss_orig_total = 0
    tvd_loss_total = 0
    kl_loss_total = 0

    batches = list(range(0, N, bsize))
    batches = shuffle(batches)

    for i in tqdm(batches, mininterval=0.5,desc='(Training)', leave=False):
        src = data[i:i+bsize]
        tgt = target[i:i+bsize]
        src, lens, masks = pad_to_longest(src)
        src = torch.LongTensor(src).cuda()
        lens = torch.LongTensor(lens)
        masks = torch.BoolTensor(masks).cuda()
        tgt = torch.Tensor(tgt).cuda()


        preds, attns = model(src, lens, masks)
        tgt = tgt.unsqueeze(-1)

        loss_orig = F.binary_cross_entropy_with_logits(preds, tgt, reduction='none')
        weight = tgt * opt.pos_weight + (1 - tgt)
        loss = (loss_orig * weight).mean(1).sum()

        optimizer[0].zero_grad()
        optimizer[1].zero_grad()
        attn_optimizer.zero_grad()
        #optimizer.zero_grad()
        loss.backward()
        #print(model.decoder.attention.attn1.weight.grad)
        #print(model.decoder.attention.attn1.weight.grad)
        #print(model.decoder.linear_1.weight.grad)
        optimizer[0].step()
        optimizer[1].step()
        attn_optimizer.step()
        #optimizer.step()
        loss_total += loss.item()
        
        ## Updates ##
        start_idx, end_idx = i,i+bsize
        all_predictions[start_idx:end_idx] = F.sigmoid(preds).squeeze().cpu().data
        all_targets[start_idx:end_idx] = tgt.squeeze().cpu().data

    return all_predictions, all_targets, loss_total
