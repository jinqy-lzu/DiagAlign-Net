import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader,ConcatDataset, random_split
import pandas
import argparse

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
# from report_feature import TextFeature
from report_feature_conch import TextFeature
# from tensorboardX import SummaryWriter
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm

from args import parse_args
from train_baseline import train_baseline, test_baseline
from sksurv.util import Surv
from iAUC import metric_calculator

from z_baseline_method.model_MLPOmics import MLPOmics
from z_baseline_method.model_MLPWSI import MLPWSI
from z_baseline_method.model_SNNOmics import SNNOmics
from z_baseline_method.model_TransMIL import TransMIL

from z_baseline_method.model_MCAT import MCAT
from z_baseline_method.model_Multimodn import MLPEncoder, PatchEncoder, ClassDecoder,MultiModNModule
from z_baseline_method.model_MMPrognosis import MMPrognosis
from z_baseline_method.model_Healnet import HealNet
from z_baseline_method.model_SurvPGC import SurvPGC_F

from measure import get_params,get_flops


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch in args.schedule:
        args.lr = args.lr * args.lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

def accuracy(output, target, topk=1):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = topk
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # for k in topk:
        correct_k = correct[:topk].float().sum()
        return correct_k.mul_(1.0 / batch_size)

def calculate_error(Y_hat, Y):
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

    return error
      
def unsqueeze_label(label, class_num,device):  
    batch_size = label.shape[0]
    label_tensor = torch.zeros(batch_size, class_num)
    for i in range(batch_size):
        index = label[i].item()
        label_tensor[int(i), int(index)] = 1
    return label_tensor.cuda(device)
        
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


def train(model, model_ema, trainDataloader, criterion, optimizer, text_encoder, device,scheduler,args,epoch):
    train_loss = AverageMeter()

    model.train()
    start = time.time() 
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_sample_haz = []
    all_sample_id = []
    all_S = []

    reg_fn = l1_reg_all
    for id, data in enumerate(trainDataloader):
        gene, batch_y, img, batch_diag,event_time,c, sample_id = data 

        diag_token = text_encoder.token(batch_diag)

        gene = gene.to(torch.float)
        event_time = event_time.to(torch.float)
        c = c.to(torch.float)
        gene = gene.cuda()
        batch_y = batch_y.cuda()
        img = img.cuda()
        event_time = event_time.cuda()
        c = c.cuda()
        diag_token = diag_token.cuda()

        text_feat = text_encoder(diag_token)#8 768
        text_feat = text_feat.float()
        label = batch_y
        optimizer.zero_grad()
        hazards, S, Y_hat  = model(img, gene, text_feat)
        loss = criterion(hazards=hazards, S=S, Y=label, c=c)
        risk_scores = -torch.sum(S, dim=1).detach().cpu().numpy()
        censorships = c.detach().cpu().numpy()
        event_times = event_time.detach().cpu().numpy()
        all_risk_scores.append(risk_scores)
        all_censorships.append(censorships)
        all_event_times.append(event_times)
        all_sample_haz.append(hazards.detach().cpu().numpy())
        all_sample_id.append(sample_id)
        all_S.append(S.detach().cpu().numpy())

        loss_reg = reg_fn(model) * args.lambda_reg
        loss = loss + loss_reg

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

def test(model, testDataloader, criterion, device,optimizer, text_encoder, epoch, logsuffix, args):
    model.eval()
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
            batch_x,batch_y,batch_img, batch_diag,event_time, c, sample_id  = data
            sur_OS.append(c)
            OS_time.append(event_time)

            batch_x = batch_x.to(torch.float)
            event_time = event_time.to(torch.float)
            c = c.to(torch.float)
            batch_x = batch_x.cuda()
            label = batch_y.cuda()
            batch_img = batch_img.cuda()
            event_time = event_time.cuda()
            c = c.cuda()
            diag_token = text_encoder.token(batch_diag)
            diag_token = diag_token.cuda()
            text_feat = text_encoder(diag_token)
            text_feat = text_feat.float()

            hazards, S, Y_hat = model(batch_img, batch_x, text_feat)
            
            loss = criterion(hazards=hazards, S=S, Y=label, c=c)
            risk_scores = -torch.sum(S, dim=1).detach().cpu().numpy()
            censorships = c.detach().cpu().numpy()
            event_times = event_time.detach().cpu().numpy()

            all_risk_scores.append(risk_scores)
            all_censorships.append(censorships)
            all_event_times.append(event_times)
            all_hazards.append(hazards.detach().cpu().numpy())
            all_sample_id.append(sample_id)
            test_loss.update(loss,label.size(0))
            all_labels.extend(batch_y.tolist())
            all_S.append(S.detach().cpu().numpy())
            err.update(calculate_error(Y_hat, label), label.size(0))
            
    all_censorships = np.concatenate(all_censorships)
    all_event_times = np.concatenate(all_event_times)
    all_risk_scores = np.concatenate(all_risk_scores)
    all_hazards = np.concatenate(all_hazards)
    all_sample_id = np.concatenate(all_sample_id)
    all_S = np.concatenate(all_S)

    all_c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    all_labels_tensor = torch.tensor(all_labels)
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

def seed_torch(seed=1029):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main():
    args, args_text = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = int(args.gpu_id)
    lucky_number = 1024 
    seed_torch(lucky_number)
    args.best_model_path = args.best_model_path + '/' + args.mode + '/'
    args.sur_ret_dir = args.best_model_path

    init_logger(args)
    # save args
    logger.info(args)

    time_str = f'log_{time.strftime("%Y%m%d_%H%M%S", time.localtime())}'
    text_encoder = TextFeature()
    trainDataloader, testDataloader,gene_size, img_feat_size = geneHistopDataloader(geneFile=args.gene_dir, selectData=None, imgDir=args.img_dir,
                                           diagnostic_reports = args.diagnostic_reports, token = text_encoder,batchSize=args.batch_size, train_ratio=args.train_ration)
    image_shape = 0
    gene_shape = 0
    text_shape = 0
    label_size = 0
    for id, data in enumerate(trainDataloader):
        gene, batch_y, img, batch_diag,event_time,c, sample_id = data 
        diag_token = text_encoder.token(batch_diag)
        diag_token = diag_token.cuda()
        text_feat = text_encoder(diag_token)
        gene = gene.to(torch.float)
        if args.mode == "Multimodn":
            img = img.permute(0,2,1)
        elif args.mode == "Healnet":
            img = img.permute(0,2,1)
            gene = gene.unsqueeze(1)
        image_shape = img.shape[1:]
        gene_shape = gene.shape[1:]
        text_shape = text_feat.shape[1:]
        label = F.one_hot(batch_y, num_classes=4)
        label_size = label.shape[1:]
        break


    l = gene_size[0]
    hw, ch = img_feat_size

    if args.mode == "gene":
        model = geneNet(gene_dim=l,vision_dim=ch, text_dim=1, head = 8, classes=args.classes)
    elif args.mode == "vision":
        model = visionNet(gene_dim=l,vision_dim=ch, text_dim=1, head = 8, classes=args.classes)
    elif args.mode == "vision_gene":
        model = vision_geneFusion(gene_dim=l,vision_dim=ch, text_dim=1, head = 8, classes=args.classes)
    elif args.mode == "vision_text":
        model = vision_textFusionNet(gene_dim=l,vision_dim=ch, text_dim=1, head = 8, classes=args.classes)
    elif args.mode == "MLPOmics":
        model_dict = {"input_dim": l, "projection_dim": 64, "dropout": 0.2}
        model = MLPOmics(**model_dict)
        model.cuda()
        logger.info(
        f'Model created, params: {get_params(model) / 1e6:.3f} M, '
        f'FLOPs: {get_flops(model, input_shape=((gene_shape),)) / 1e9:.3f} G')
    
    elif args.mode == "MLPWSI":
        model_dict = {"wsi_embedding_dim": ch, "dropout":0.2, "device": "cuda"}
        model = MLPWSI(**model_dict)
        model.cuda()
        logger.info(
        f'Model created, params: {get_params(model) / 1e6:.3f} M, '
        f'FLOPs: {get_flops(model, input_shape=((image_shape),)) / 1e9:.3f} G')
    elif args.mode == "SNNOmics":
        model_dict = {"omic_input_dim": l}
        model = SNNOmics(**model_dict)
        model.cuda()
        logger.info(
        f'Model created, params: {get_params(model) / 1e6:.3f} M, '
        f'FLOPs: {get_flops(model, input_shape=((gene_shape),)) / 1e9:.3f} G')
    elif args.mode == "TransMIL":
        model_dict = {"input_dim": ch, 'n_classes': args.classes}
        model = TransMIL(**model_dict)
        model.cuda()
        logger.info(
        f'Model created, params: {get_params(model) / 1e6:.3f} M, '
        f'FLOPs: {get_flops(model, input_shape=((image_shape),)) / 1e9:.3f} G')
    elif args.mode == "MCAT":
        model = MCAT(n_classes=args.classes,omic_shape=(l,),wsi_shape=(ch,hw))
        model.cuda()
        logger.info(
        f'Model created, params: {get_params(model) / 1e6:.3f} M, '
        f'FLOPs: {get_flops(model, input_shape=((image_shape),(gene_shape),)) / 1e9:.3f} G')

    elif args.mode == "MMPrognosis":
        model = MMPrognosis(sources=["omic", "slides"],
                                output_dims=args.classes,
                                batch_size=args.batch_size
                                )
    elif args.mode == "Multimodn":
        l_d = 512
        tab_features = l
        patch_dims = hw
        encoders = [MLPEncoder(state_size=l_d, hidden_layers=[1024, 256, 128, 64], n_features=tab_features),
                    PatchEncoder(state_size=l_d, hidden_layers=[512, 256, 128, 64], n_features=patch_dims)]
        decoders = [ClassDecoder(state_size=l_d, n_classes=args.classes, activation=torch.sigmoid)]
        model = MultiModNModule(state_size=l_d,
                encoders=encoders,
                decoders=decoders)
        model.cuda()
    elif args.mode == "Healnet":
        input_channels = [l, ch]
        input_axes = [1, 1]
        modalities = 2
        model = HealNet(n_modalities=modalities,
                channel_dims=input_channels, 
                num_spatial_axes=input_axes,
                out_dims=args.classes)
        model.cuda()
    elif args.mode == "SurvPGC":
        model = SurvPGC_F(wsi_embedding_dim = ch, clinic_embedding_dim = text_shape[0],gene_embedding_dim = l )
        model.cuda()
    else:
        model = xxxFusionNet(gene_dim=l,vision_dim=ch, text_dim=1, head = 8, classes=args.classes)
        model.cuda()

    logger.info(model)
    optimizer = build_optimizer(args.opt,
                                model,
                                args.lr,
                                eps=args.opt_eps,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                filter_bias_and_bn=not args.opt_no_filter,
                                nesterov=not args.sgd_no_nesterov,
                                sort_params=args.dyrep)
    if args.model_ema:
        model_ema = ModelEMA(model, decay=args.model_ema_decay)
    else:
        model_ema = None
    ckpt_manager = CheckpointManager(model,
                                     optimizer,
                                     ema_model=model_ema,
                                     save_dir=args.best_model_path
                                     )
    loss_fn = None
    if args.task_type == 'survival':
        if args.bag_loss == 'ce_surv':
            loss_fn = CrossEntropySurvLoss(alpha=args.alpha_surv)
        elif args.bag_loss == 'nll_surv':
            loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
        elif args.bag_loss == 'cox_surv':
            loss_fn = CoxSurvLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    criterion = loss_fn
    cudnn.benchmark = True

    if args.sched != "step":
        steps_per_epoch = len(trainDataloader)
        warmup_steps = args.warmup_epochs * steps_per_epoch
        decay_steps = args.decay_epochs * steps_per_epoch
        total_steps = args.epochs * steps_per_epoch
        scheduler = build_scheduler(args.sched,
                                    optimizer,
                                    warmup_steps,
                                    args.warmup_lr,
                                    decay_steps,
                                    args.decay_rate,
                                    total_steps,
                                    steps_per_epoch=steps_per_epoch,
                                    decay_by_epoch=args.decay_by_epoch,
                                    min_lr=args.min_lr)
    else:
        scheduler=None
    

    best_acc = 0
    score_auc = []
    label_auc = []
    for epoch in range(args.epochs):
        if args.sched == "step":          
            adjust_learning_rate(optimizer, epoch, args)
        if (args.mode == "Multimodn" or args.mode == "TransMIL" or 
            args.mode == "MLPOmics" or args.mode == "MLPWSI" or 
            args.mode == "SNNOmics" or args.mode == "MCAT" or 
            args.mode == "MMPrognosis" or args.mode == "Healnet" or args.mode == "SurvPGC"):
            metrics,train_sur_os,train_sur_time,train_sur_risk,train_surv_hazards,train_sample_id = train_baseline(model, model_ema, trainDataloader, criterion, optimizer, text_encoder,device, scheduler, args, epoch,logger)
        else:
            metrics,train_sur_os,train_sur_time,train_sur_risk,train_surv_hazards,train_sample_id = train(model, model_ema, trainDataloader, criterion, optimizer, text_encoder,device, scheduler, args, epoch)
        
        if (args.mode == "Multimodn" or args.mode == "TransMIL" or 
            args.mode == "MLPOmics" or args.mode == "MLPWSI" or 
            args.mode == "SNNOmics" or args.mode == "MCAT" or 
            args.mode == "MMPrognosis" or args.mode == "Healnet" or args.mode == "SurvPGC"):
            metrics, sur_os,sur_time,sur_risk,surv_hazards,sample_id = test_baseline(model, testDataloader, criterion, device,optimizer, text_encoder,epoch, "Test:", args,logger)
        else:
            metrics, sur_os,sur_time,sur_risk,surv_hazards,sample_id = test(model, testDataloader, criterion, device,optimizer, text_encoder,epoch, "Test:", args)
        if model_ema is not None:
           test(model_ema.module, testDataloader, criterion, device, optimizer, epoch,"EMA:",args)
        ckpts = ckpt_manager.update(epoch, metrics)
        if best_acc <= metrics['top1']:
            best_acc = metrics['top1']
            col_name1 = []
            survival_df = pandas.DataFrame({"ID": sample_id,"OS":sur_os,"sur_time":sur_time,"risk":sur_risk})
            for i in range(0, surv_hazards.shape[1]):
                col_name1.append(f"{i}_hazard")
            surv_hazards = pandas.DataFrame(surv_hazards, columns=col_name1)
            survival_df = pandas.concat([survival_df, surv_hazards], axis=1)
            file = f'test_{time_str}_survival_risk.csv'
            survival_df.to_csv(os.path.join(args.sur_ret_dir, file))

            col_name2 = []
            survival_df_train = pandas.DataFrame({"ID":train_sample_id,"OS":train_sur_os,"sur_time":train_sur_time,"risk":train_sur_risk})
            for i in range(0, train_surv_hazards.shape[1]):
                col_name2.append(f"{i}_hazard")
            train_surv_hazards = pandas.DataFrame(train_surv_hazards, columns=col_name2)
            survival_df = pandas.concat([survival_df, train_surv_hazards], axis=1)
            file = f'train_{time_str}_survival_risk.csv'
            survival_df_train.to_csv(os.path.join(args.sur_ret_dir, file))
            logger.info("save predict risk!!!!!")  

if __name__ == '__main__':
    main()