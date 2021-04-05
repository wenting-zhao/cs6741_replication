import argparse, warnings 
import torch
import numpy as np
from config_args import get_args, config_args
#from runner import run_model
from Trainers.DatasetBC import datasets
from models.Models import Model
from runner import run_model
from itertools import chain
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
opt = get_args(parser)
config_args(opt)
print(opt)
torch.manual_seed(opt.seed)
np.random.seed(opt.seed)

def main(opt):
    #========= Loading Dataset =========#
    dataset = datasets[opt.dataset](opt)

    if opt.output_dir is not None :
        dataset.output_dir = opt.output_dir

    if opt.adversarial :
        exp_name = '+'.join((opt.encoder, 'adversarial'))
    else :
        exp_name = '+'.join((opt.encoder, opt.attention))
    #========= Preparing Model =========#

    model = Model(dataset.vec.vocab_size, dataset.vec.word_dim, opt.hidden_size, dataset.output_size, dataset.vec.embeddings, opt.encoder, opt.attention, opt.use_attention)
    model = model.cuda()
    print(model)

    #params = chain(model.encoder.parameters(), model.decoder.linear_1.parameters())
    #optimizer = torch.optim.Adam(model.parameters(),lr=0.01, weight_decay=1e-5, amsgrad=True)
    optimizer = torch.optim.Adam(model.encoder.parameters(),lr=opt.lr, weight_decay=1e-5, amsgrad=True)
    optimizer2 = torch.optim.Adam(model.decoder.linear_1.parameters(),lr=opt.lr, weight_decay=1e-5, amsgrad=True)
    scheduler = torch.torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step_size, gamma=opt.lr_decay,last_epoch=-1)
    attn_params = model.decoder.attention.parameters()
    attn_optimizer = torch.optim.Adam(attn_params,lr=opt.lr, weight_decay=0, amsgrad=True)

    #========= Preparing Criterion =========#
    y = np.array(dataset.train_data.y)
    opt.pos_weight = (len(y) / sum(y) - 1)


    try:
        run_model(model,dataset.train_data,dataset.test_data,(optimizer,optimizer2),attn_optimizer,scheduler,opt)
        #run_model(model,dataset.train_data,dataset.test_data,optimizer,attn_optimizer,scheduler,opt)
        #run_model(model,dataset.train_data,dataset.test_data,optimizer,scheduler,opt)
    except KeyboardInterrupt:
        print('-' * 89+'\nManual Exit')
        exit()

if __name__ == '__main__':
    main(opt)
