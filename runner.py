import warnings
import time
from tqdm import tqdm
from train import train_epoch
from test import test_epoch
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore")

def run_model(model,train_data,test_data,optimizer,attn_optimizer,scheduler,opt):
#def run_model(model,train_data,test_data,optimizer,scheduler,opt):
    
    valid_losses = []

    losses = []

    for epoch_i in range(opt.epoch):
        print('================= Epoch', epoch_i+1, '=================')
        if scheduler and opt.lr_decay > 0: scheduler.step()
        ################################## TRAIN ###################################
        start = time.time()
        all_predictions,all_targets,train_loss=train_epoch(model,train_data,optimizer,attn_optimizer,(epoch_i+1),opt)
        #all_predictions,all_targets,train_loss=train_epoch(model,train_data,optimizer,(epoch_i+1),opt)
        elapsed = ((time.time()-start)/60)
        print('\n(Training) elapse: {elapse:3.3f} min'.format(elapse=elapsed))
        train_loss = train_loss/len(train_data.X)
        print('B : '+str(train_loss))
        all_predictions[all_predictions >= 0.5] = 1
        all_predictions[all_predictions < 0.5] = 0
        res = classification_report(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions)
        print(res)
        print(f1)

        ################################## TEST ###################################
        start = time.time()
        all_predictions, all_targets, test_loss = test_epoch(model,test_data,opt)
        elapsed = ((time.time()-start)/60)
        print('\n(Testing) elapse: {elapse:3.3f} min'.format(elapse=elapsed))
        test_loss = test_loss/len(test_data.X)
        print('B : '+str(test_loss))
        all_predictions[all_predictions >= 0.5] = 1
        all_predictions[all_predictions < 0.5] = 0
        res = classification_report(all_targets, all_predictions)
        print(res)
        f1 = f1_score(all_targets, all_predictions)
        print(f1)

