from collections import OrderedDict
from os.path import join
import pdb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# from model_utils import *


def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    import torch.nn as nn

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False))

"""

Implement a self normalizing network to handle tabular omics data 

Klambauer, Günter, et al. "Self-normalizing neural networks." Advances in neural information processing systems 30 (2017).

"""


##########################
#### Genomic FC Model ####
##########################
class SNNOmics(nn.Module):
    def __init__(self, omic_input_dim: int, model_size_omic: str='small', n_classes: int=4):
        super(SNNOmics, self).__init__()
        self.n_classes = n_classes
        self.size_dict_omic = {'small': [256, 256], 'big': [1024, 1024, 1024, 256]}
        
        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        fc_omic = [SNN_Block(dim1=omic_input_dim, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
        self.fc_omic = nn.Sequential(*fc_omic)
        self.classifier = nn.Linear(hidden[-1], n_classes)
        init_max_weights(self)


    def forward(self, omics):
        x = omics
        h_omic = self.fc_omic(x)
        predict  = self.classifier(h_omic) # logits needs to be a [B x 4] vector   

        Y_hat = torch.topk(predict, 1, dim=1)[1]
        Y_prob = F.softmax(predict, dim=1)
        hazards = torch.sigmoid(predict)
        S = torch.cumprod(1 - hazards, dim=1)
        risk_scores = -torch.sum(S, dim=1)
        return hazards, S, Y_hat

        # assert len(predict.shape) == 2 and predict.shape[1] == self.n_classes
        # if return_feats:
        #     return h_omic, predict
        # return predict

    def relocate(self):
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if torch.cuda.device_count() > 1:
                device_ids = list(range(torch.cuda.device_count()))
                self.fc_omic = nn.DataParallel(self.fc_omic, device_ids=device_ids).to('cuda:0')
            else:
                self.fc_omic = self.fc_omic.to(device)


            self.classifier = self.classifier.to(device)



def init_max_weights(module):
    r"""
    Initialize Weights function.

    args:
        modules (torch.nn.Module): Initalize weight using normal distribution
    """
    import math
    import torch.nn as nn
    
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()