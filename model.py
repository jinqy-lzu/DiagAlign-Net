import torch
import torch.nn as nn
import torch.nn.functional as F
from vit import ViT
from feature_fusion import FeatureAdapter, FusionText, FusionGene, FusionVision
from gene_feature import StatisSpitialFusion
from vision_feature import VisionText, Vision
import math
def l1_reg_all(model, reg_type=None):
    l1_reg = None

    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            
            l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1) 
    return l1_reg

def l1_reg_modules(model, reg_type=None):
    l1_reg = 0
    l1_reg =  l1_reg_all(model.enhence_module) + l1_reg_all(model.gene_embed) + l1_reg_all(model.fusion_module)
    return l1_reg



class FusionNet(nn.Module):
    def __init__(self, gene_dim, vision_dim, text_dim, head, classes):
        super().__init__()
        self.classes = classes
        self.gene_channel = 128
        self.vision_channel = vision_dim
        self.text_channel = text_dim
        self.text_embed_dim = 1536
        self.gene_embed_dim = 1536
        self.gene_drop = 0.2
        self.lstm_layer = 3
        self.adapt_out = self.gene_channel
        self.head = head
        self.fusion_channel = 128
        self.fusion_drop = 0.3
        self.group_size = 4 
        
        self.vision_embed_dim = 10000 // (self.group_size * self.group_size)
        self.fusion_length = 2
        self.meanpool = nn.AdaptiveAvgPool1d(self.fusion_length)
        self.linear_adapt = False
        
        ####################################
        # The code is being organized
        ####################################
        
        self.classifier = nn.Linear(self.fusion_channel * 3 * self.fusion_length, self.classes)
        
    def forward(self, vision, gene, text,f_fusion = False):

        ####################################
        # The code is being organized
        ####################################

        if f_fusion:
            return fusion, hazards, S, Y_hat

        return hazards, S, Y_hat

    def captum(self, vision, gene, text):
        batch_size, seq_len = text.shape
        if seq_len > self.text_embed_dim:
            pool = nn.AdaptiveAvgPool1d(self.text_embed_dim)
            text = pool(text)  # [batch, feature_dim, text_embed_dim]
        elif seq_len < self.text_embed_dim:
            pad_len = self.text_embed_dim - seq_len
            padding = torch.zeros(batch_size, pad_len, device=text.device, dtype=text.dtype)
            text = torch.cat([text, padding], dim=1)
        if vision.dim() == 3:
            b,l,c = vision.shape
            vision = vision.unsqueeze(1)
            h = w = int(math.sqrt(l))
            vision = vision.permute(0,3,1,2)
            vision = vision.view(b, c, h, -1)
        if text.dim() == 2:
            text = text.unsqueeze(-1)
        if self.linear_adapt:
            text = self.text_adapt(text)
        text_feature = self.adapt_layer(text)
        gene_feature = self.gene_embed(gene)
        gene_feature = self.adapt_layer(gene_feature)

        vision_feature = self.VisionText(vision, text_feature)
        
        t_fusion = self.TMF(text_feature, vision_feature, gene_feature)
        g_fusion = self.GMF(text_feature, vision_feature, gene_feature)
        v_fusion = self.VMF(text_feature, vision_feature, gene_feature)


        t_fusion = self.meanpool(t_fusion.permute(0,2,1))
        g_fusion = self.meanpool(g_fusion.permute(0,2,1))
        v_fusion = self.meanpool(v_fusion.permute(0,2,1))

        fusion = torch.cat([t_fusion, g_fusion, v_fusion], dim=1)
        fusion = torch.flatten(fusion, start_dim=1)

        predict = self.classifier(fusion)
        Y_hat = torch.topk(predict, 1, dim=1)[1]
        Y_prob = F.softmax(predict, dim=1)
        hazards = torch.sigmoid(predict)
        S = torch.cumprod(1 - hazards, dim=1)
        risk_scores = -torch.sum(S, dim=1)
        return risk_scores


class geneNet(nn.Module):
    def __init__(self, gene_dim, vision_dim, text_dim, head, classes):
        super().__init__()
        self.classes = classes
        self.gene_channel = 256
        self.gene_embed_dim = 1536
        self.gene_drop = 0.2
        self.lstm_layer = 3
        self.adapt_out = self.gene_channel
        self.fusion_length = 2
        self.meanpool = nn.AdaptiveAvgPool1d(self.fusion_length)
        self.gene_embed = StatisSpitialFusion(input_size=gene_dim, 
                                              hidden_size=self.gene_channel,
                                              embed_size = self.gene_embed_dim, 
                                              drop_out=self.gene_drop,
                                              num_layers=self.lstm_layer)
        self.adapt_layer = FeatureAdapter(channel_in = self.gene_channel, 
                                            channel_out = self.adapt_out, 
                                            embed_dim = self.gene_embed_dim, 
                                            dropout = 0.2)
        self.classifier = nn.Linear(self.adapt_out  * self.fusion_length, self.classes)
        
    def forward(self, gene, f_fusion=False):
        gene_feature = self.gene_embed(gene)
        gene_feature = self.adapt_layer(gene_feature)
        feature = self.meanpool(gene_feature.permute(0,2,1))
        feature = torch.flatten(feature, start_dim=1)
        predict = self.classifier(feature)
        Y_hat = torch.topk(predict, 1, dim=1)[1]
        Y_prob = F.softmax(predict, dim=1)
        hazards = torch.sigmoid(predict)
        S = torch.cumprod(1 - hazards, dim=1)
        risk_scores = -torch.sum(S, dim=1)
        if f_fusion:
            return feature, hazards, S, Y_hat
        return hazards, S, Y_hat
    
    def captum(self, gene):
        gene_feature = self.gene_embed(gene)
        gene_feature = self.adapt_layer(gene_feature)
        feature = self.meanpool(gene_feature.permute(0,2,1))
        feature = torch.flatten(feature, start_dim=1)
        predict = self.classifier(feature)
        Y_hat = torch.topk(predict, 1, dim=1)[1]
        Y_prob = F.softmax(predict, dim=1)
        hazards = torch.sigmoid(predict)
        S = torch.cumprod(1 - hazards, dim=1)
        risk_scores = -torch.sum(S, dim=1)
        return risk_scores
    


class visionNet(nn.Module):
    def __init__(self, gene_dim, vision_dim, text_dim, head, classes):
        super().__init__()
        self.classes = classes
        self.gene_channel = 256#256 128
        self.vision_channel = vision_dim
        self.head = head
        self.group_size = 4
        
        self.vision_embed_dim = 10000 // (self.group_size * self.group_size)
        self.fusion_length = 2
        self.meanpool = nn.AdaptiveAvgPool1d(self.fusion_length)
        self.adapt_out = 256
        self.Vision = Vision(vision_dim = self.vision_channel , vision_embed_dim = self.vision_embed_dim,
                                    out_dim = self.adapt_out , head = self.head, group_size = self.group_size, 
                                    vision_dropout = 0.3)
        self.classifier = nn.Linear(self.adapt_out * self.fusion_length, self.classes)
        
    def forward(self, vision):

        if vision.dim() == 3:
            b,l,c = vision.shape
            vision = vision.unsqueeze(1)
            h = w = int(math.sqrt(l))
            vision = vision.permute(0,3,1,2)
            vision = vision.view(b, c, h, -1)

        vision_feature = self.Vision(vision)
        v_fusion = self.meanpool(vision_feature.permute(0,2,1))

        fusion = torch.flatten(v_fusion, start_dim=1)
        predict = self.classifier(fusion)
        Y_hat = torch.topk(predict, 1, dim=1)[1]
        Y_prob = F.softmax(predict, dim=1)
        hazards = torch.sigmoid(predict)
        S = torch.cumprod(1 - hazards, dim=1)
        risk_scores = -torch.sum(S, dim=1)
        return hazards, S, Y_hat



    def __init__(self, gene_dim, vision_dim, text_dim, head, classes):
        super().__init__()
        self.classes = classes
        self.gene_channel = 256#256 128
        self.vision_channel = vision_dim
        self.text_channel = text_dim
        self.gene_embed_dim = 1536#768
        self.gene_drop = 0.2
        self.lstm_layer = 3
        self.adapt_out = self.gene_channel
        self.head = head
        self.fusion_channel = 128
        self.fusion_drop = 0.3#0.3
        self.group_size = 4
        
        self.vision_embed_dim = 10000 // (self.group_size * self.group_size)
        self.fusion_length = 2
        self.meanpool = nn.AdaptiveAvgPool1d(self.fusion_length)

        self.gene_embed = StatisSpitialFusion(input_size=gene_dim, 
                                              hidden_size=self.gene_channel,
                                              embed_size = self.gene_embed_dim, 
                                              drop_out=self.gene_drop,
                                              num_layers=self.lstm_layer)
        self.adapt_layer = FeatureAdapter(channel_in = self.gene_channel, 
                                            channel_out = self.adapt_out, 
                                            embed_dim = self.gene_embed_dim, 
                                            dropout = 0.2)

        self.VisionText = VisionText(vision_dim = self.vision_channel , vision_embed_dim = self.vision_embed_dim,
                                    text_dim = self.adapt_out, text_embed_dim = self.gene_embed_dim,
                                    out_dim = self.adapt_out , head = self.head, group_size = self.group_size, vision_dropout = 0.3)
        self.classifier = nn.Linear(self.adapt_out *self.fusion_length, self.classes)
        
    def forward(self, vision, gene, f_fusion=False):
        if vision.dim() == 3:
            b,l,c = vision.shape
            vision = vision.unsqueeze(1)
            h = w = int(math.sqrt(l))
            vision = vision.permute(0,3,1,2)
            vision = vision.view(b, c, h, -1)

        gene_feature = self.gene_embed(gene)
        gene_feature = self.adapt_layer(gene_feature)

        fusion_feature = self.VisionText(vision, gene_feature)
        fusion_feature = self.meanpool(fusion_feature.permute(0,2,1))
        fusion_feature = torch.flatten(fusion_feature, start_dim=1)

        predict = self.classifier(fusion_feature)
        Y_hat = torch.topk(predict, 1, dim=1)[1]
        Y_prob = F.softmax(predict, dim=1)
        hazards = torch.sigmoid(predict)
        S = torch.cumprod(1 - hazards, dim=1)
        risk_scores = -torch.sum(S, dim=1)
        if f_fusion:
            return fusion_feature,hazards, S, Y_hat
        return hazards, S, Y_hat
        # # risk_scores = risk_scores.view(-1, 1)
        # return risk_scores
    def captum(self, vision, gene):
        if vision.dim() == 3:
            b,l,c = vision.shape
            vision = vision.unsqueeze(1)
            h = w = int(math.sqrt(l))
            vision = vision.permute(0,3,1,2)
            vision = vision.view(b, c, h, -1)

        gene_feature = self.gene_embed(gene)
        gene_feature = self.adapt_layer(gene_feature)

        fusion_feature = self.VisionText(vision, gene_feature)
        fusion_feature = self.meanpool(fusion_feature.permute(0,2,1))
        fusion_feature = torch.flatten(fusion_feature, start_dim=1)

        predict = self.classifier(fusion_feature)
        Y_hat = torch.topk(predict, 1, dim=1)[1]
        Y_prob = F.softmax(predict, dim=1)
        hazards = torch.sigmoid(predict)
        S = torch.cumprod(1 - hazards, dim=1)
        risk_scores = -torch.sum(S, dim=1)
        return risk_scores