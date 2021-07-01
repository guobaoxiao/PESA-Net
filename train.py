import numpy as np
import torch
import torch.optim as optim
import sys
from tqdm import trange
import os
from logger import Logger
from test import valid
from loss import MatchLoss
from utils import tocuda


def train_step(step, optimizer, model, match_loss, data):
    model.train()

    res_logits, res_e_hat = model(data)   
    loss = 0
    loss_val = []
    for i in range(len(res_logits)):
        loss_i, geo_loss, cla_loss, l2_loss, _, _ = match_loss.run(step, data, res_logits[i], res_e_hat[i])
        loss += loss_i
        loss_val += [geo_loss, cla_loss, l2_loss]
    optimizer.zero_grad()
    loss.backward()

    # for name, param in model.named_parameters():
    #     if torch.any(torch.isnan(param.grad)):
    #         print('skip because nan')
    #         return loss_val

    optimizer.step()
    return loss_val,loss


def train(model, train_loader, valid_loader, config):
    model.cuda()   # initialize model???
    optimizer = optim.Adam(model.parameters(), lr=config.train_lr, weight_decay = config.weight_decay)
    match_loss = MatchLoss(config) # loss

    checkpoint_path = os.path.join(config.log_path, 'checkpoint.pth') # whether checkpoint
    print('checkpoint_path:',checkpoint_path)
    config.resume = os.path.isfile(checkpoint_path) # reture F or Y
    # restart?? if there is checkpiont.pth then continue
    if config.resume:
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(checkpoint_path)
        best_acc = checkpoint['best_acc']  # very good
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # logger_train = Logger(os.path.join(config.log_path, 'log_train.txt'), title='oan', resume=True)
        logger_valid = Logger(os.path.join(config.log_path, 'log_valid.txt'), title='oan', resume=True)
        FPR_valid = Logger(os.path.join(config.log_path, 'FPR_valid.txt'), title='oan', resume=True)
    else:
        best_acc = -1
        start_epoch = 0
        # logger_train = Logger(os.path.join(config.log_path, 'log_train.txt'), title='oan')
        # logger_train.set_names(['Learning Rate'] + ['Geo Loss', 'Classfi Loss', 'L2 Loss']*(config.iter_num+1))
        logger_valid = Logger(os.path.join(config.log_path, 'log_valid.txt'), title='oan')
        logger_valid.set_names(['Valid Acc'] + ['Geo Loss', 'Clasfi Loss', 'L2 Loss'])
        FPR_valid = Logger(os.path.join(config.log_path, 'FPR_valid.txt'), title='oan')
        FPR_valid.set_names(['F', 'P', 'R'])


    train_loader_iter = iter(train_loader)
    # print('train_loader_iter:',train_loader_iter)
    # print('start_epoch', 'config.train_iter',start_epoch, config.train_iter)
    for step in range(start_epoch, config.train_iter):       
        try:
            train_data = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            train_data = next(train_loader_iter)
        train_data = tocuda(train_data) # convert tensor data in dictionary to cuda when it is a tensor

        # run training
        cur_lr = optimizer.param_groups[0]['lr']

        
        loss_vals,loss = train_step(step, optimizer, model, match_loss, train_data)  

        # logger_train.append([cur_lr] + loss_vals)

       
        #print('step:', step, 'loss:', loss)
        b_save = ((step + 1) % config.save_intv) == 0    # config.save_intv =1000 1000
        b_validate = ((step + 1) % config.val_intv) == 0 # config.save_intv =1000
        if b_validate:
            
            va_res, geo_loss, cla_loss, l2_loss,P,R,F= valid(valid_loader, model, step, config)
            logger_valid.append([va_res, geo_loss, cla_loss, l2_loss])
            FPR_valid.append([F, P, R])
            print('step:', step,'best_acc:',best_acc,'va_res:',va_res,"F = {}".format(F),"P = {}".format(P),"R = {}".format(R))
            if va_res > best_acc:
                print("Saving best model with va_res = {}".format(va_res))
                best_acc = va_res
                torch.save({
                'epoch': step + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
                }, os.path.join(config.log_path, 'model_best.pth'))

        if b_save:
            torch.save({
            'epoch': step + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
            }, checkpoint_path)




