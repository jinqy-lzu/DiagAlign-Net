
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader,ConcatDataset, random_split
import pandas
import argparse
import os
import random
# import tensorboard_logger as tb_logger
import torch.backends.cudnn as cudnn
# 定义模型
import torch.nn.functional as F
from model import xxxFusionNet, l1_reg_all, geneNet, visionNet, vision_textFusionNet,vision_geneFusion
from dataset import geneHistopDataloader 
from datetime import datetime
from optim import build_optimizer
from schedule import build_scheduler
from model_ema import ModelEMA
from misc import CheckpointManager, init_logger
from utils import CrossEntropySurvLoss,NLLSurvLoss,CoxSurvLoss
from sksurv.metrics import concordance_index_censored
import logging
import time
import math
from report_feature import TextFeature
# from tensorboardX import SummaryWriter
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm
from sksurv.util import Surv
from iAUC import metric_calculator

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count      


def train_baseline(model, model_ema, trainDataloader, criterion, optimizer, text_encoder, device,scheduler,args,epoch,logger):
    train_loss = AverageMeter()
    # criterion = nn.CrossEntropyLoss()
    model.train()
    start = time.time() 
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_sample_haz = []
    all_sample_id = []
    all_S = []
    #l1正则化
    reg_fn = l1_reg_all
    for id, data in enumerate(trainDataloader):
        gene, batch_y, img, batch_diag,event_time,c, sample_id = data #event_time生存时间,c 生存状态

        diag_token = text_encoder.token(batch_diag)
        diag_token = diag_token.cuda()
        text_feat = text_encoder(diag_token)
        text_feat = text_feat.float()


        gene = gene.to(torch.float)
        event_time = event_time.to(torch.float)
        c = c.to(torch.float)
        gene = gene.cuda()
        batch_y = batch_y.cuda()
        img = img.cuda()
        event_time = event_time.cuda()
        c = c.cuda()
        label = batch_y
        optimizer.zero_grad()
        model_loss = 0.0
        if args.mode == "TransMIL":
            hazards, S, Y_hat  = model(img)
        elif args.mode == "MLPOmics":
            hazards, S, Y_hat  = model(gene)
        elif args.mode == "MLPWSI":
            hazards, S, Y_hat  = model(img)
        elif args.mode == "SNNOmics":
            hazards, S, Y_hat  = model(gene)                                                                           
        elif args.mode == "MCAT":
            hazards, S, Y_hat  = model(img, gene)
        elif args.mode == "MMPrognosis":
            img = img.permute(0,2,1)
            hazards, S, Y_hat  = model([gene,img])
        elif args.mode == "Multimodn":
            img = img.permute(0,2,1)
            model_loss, hazards, S, Y_hat = model.forward([gene,img], F.one_hot(label, num_classes=4))
        elif args.mode == "Healnet":
            img = img.permute(0,2,1)
            gene = gene.unsqueeze(1)
            hazards, S, Y_hat  = model([gene,img])
        elif args.mode == "SurvPGC":
            gene = gene.unsqueeze(1)
            text_feat = text_feat.unsqueeze(1)
            hazards, S, Y_hat = model(img, gene,text_feat )
        else:
            raise ValueError("Invalid mode")
        loss = criterion(hazards=hazards, S=S, Y=label, c=c)
        risk_scores = -torch.sum(S, dim=1).detach().cpu().numpy()
        censorships = c.detach().cpu().numpy()
        event_times = event_time.detach().cpu().numpy()
        all_risk_scores.append(risk_scores)
        all_censorships.append(censorships)
        all_event_times.append(event_times)
        all_sample_haz.append(hazards.detach().cpu().numpy())
        all_S.append(S.detach().cpu().numpy())
        all_sample_id.append(sample_id)

        loss_reg = reg_fn(model) * args.lambda_reg
        loss = loss + loss_reg + model_loss

        loss.backward()
        optimizer.step()

        train_loss.update(loss,label.size(0))
        end = time.time()
        if args.sched != "step":
            scheduler.step()
        if model_ema is not None:
            model_ema.update(model)
    all_censorships = np.concatenate(all_censorships)
    all_event_times = np.concatenate(all_event_times)
    all_risk_scores = np.concatenate(all_risk_scores)
    all_sample_haz = np.concatenate(all_sample_haz)
    all_sample_id = np.concatenate(all_sample_id)
    all_S = np.concatenate(all_S)

    all_c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    survival_train  = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    c_index, c_index2,  iauc, iauc_list = metric_calculator(all_survival_months = all_event_times, survival_train = survival_train, all_risk_scores = all_risk_scores, 
                        all_censorships = all_censorships, all_event_times = all_event_times, risk_by_bin = all_S)
    logger.info('Train: {} | ' 'C-index: {:.4f} | '  'iAUC: {:.8F} |' 'Loss: {:.8f} | '
                'LR: {:.3e} | ' 'Time:({:.2f}s) '
                        .format(
                            epoch,
                            all_c_index,
                            iauc,
                            train_loss.avg,
                            optimizer.param_groups[0]['lr'],
                            end-start,
                            ))      
    return {'top1': all_c_index,"train_loss":train_loss.avg},all_censorships,all_event_times,all_risk_scores,all_sample_haz,all_sample_id



def test_baseline(model, testDataloader, criterion, device,optimizer, text_encoder, epoch, logsuffix, args,logger):
    model.eval()
    # criterion = nn.CrossEntropyLoss()
    test_loss = AverageMeter()
    err = AverageMeter()
    all_labels = []
    sur_OS = []
    OS_time = []
    
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_hazards = []
    all_sample_id = []
    all_S = []
    with torch.no_grad():
        for id, data in enumerate(testDataloader):
            gene,batch_y,img, batch_diag,event_time, c, sample_id  = data

            diag_token = text_encoder.token(batch_diag)
            diag_token = diag_token.cuda()
            text_feat = text_encoder(diag_token)
            text_feat = text_feat.float()
            sur_OS.append(c)
            OS_time.append(event_time)
            gene = gene.to(torch.float)
            event_time = event_time.to(torch.float)
            c = c.to(torch.float)
            gene = gene.cuda()
            label = batch_y.cuda()
            img = img.cuda()
            event_time = event_time.cuda()
            c = c.cuda()

            if args.mode == "TransMIL":
                hazards, S, Y_hat  = model(img)
            elif args.mode == "MLPOmics":
                hazards, S, Y_hat  = model(gene)
            elif args.mode == "MLPWSI":
                hazards, S, Y_hat  = model(img)
            elif args.mode == "SNNOmics":
                hazards, S, Y_hat  = model(gene)
            elif args.mode == "MCAT":
                hazards, S, Y_hat  = model(img, gene)
            elif args.mode == "MMPrognosis":
                img = img.permute(0,2,1)
                hazards, S, Y_hat  = model([gene,img])
            elif args.mode == "Multimodn":
                img = img.permute(0,2,1)
                model_loss, hazards, S, Y_hat = model.forward([gene,img], F.one_hot(label, num_classes=4))
            elif args.mode == "Healnet":
                img = img.permute(0,2,1)
                gene = gene.unsqueeze(1)
                hazards, S, Y_hat = model([gene,img])
            elif args.mode == "SurvPGC":
                gene = gene.unsqueeze(1)
                text_feat = text_feat.unsqueeze(1)
                hazards, S, Y_hat = model(img, gene,text_feat)
            else:
                raise ValueError("Invalid mode")
            
            loss = criterion(hazards=hazards, S=S, Y=label, c=c)
            risk_scores = -torch.sum(S, dim=1).detach().cpu().numpy()
            censorships = c.detach().cpu().numpy()
            event_times = event_time.detach().cpu().numpy()

            all_risk_scores.append(risk_scores)
            all_censorships.append(censorships)
            all_event_times.append(event_times)
            all_hazards.append(hazards.detach().cpu().numpy())
            all_S.append(S.detach().cpu().numpy())
            all_sample_id.append(sample_id)
            test_loss.update(loss,label.size(0))
            all_labels.extend(batch_y.tolist())

            
    all_censorships = np.concatenate(all_censorships)
    all_event_times = np.concatenate(all_event_times)
    all_risk_scores = np.concatenate(all_risk_scores)
    all_hazards = np.concatenate(all_hazards)
    all_sample_id = np.concatenate(all_sample_id)
    all_S = np.concatenate(all_S)

    all_c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    # all_c_index = concordance_index_censored((all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    all_labels_tensor = torch.tensor(all_labels)
    # 使用 torch.unique 获取所有唯一标签以及它们的计数
    unique_labels, counts = torch.unique(all_labels_tensor, return_counts=True)

    survival_train  = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    c_index, c_index2,  iauc, iauc_list = metric_calculator(all_survival_months = all_event_times, survival_train = survival_train, all_risk_scores = all_risk_scores, 
                        all_censorships = all_censorships, all_event_times = all_event_times, risk_by_bin = all_S)


    print(f"test data Unique labels: {unique_labels}")
    print(f"test data Counts: {counts}")
    logger.info('{}: {} | ' 'C-index: {:.4f} | ' 'iAUC: {:.8f} | ' 'Loss: {:.8f} | ' ' ERR:{} |'
                'LR: {:.3e} '.format(
                            logsuffix,
                            epoch,
                            all_c_index,
                            iauc,
                            test_loss.avg,
                            err.avg,
                            optimizer.param_groups[0]['lr']
                            ))
    return {"test_loss": test_loss.avg,"top1": all_c_index},all_censorships,all_event_times,all_risk_scores,all_hazards,all_sample_id
