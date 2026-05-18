import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from diff_attention import DiffAttention


class CrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dims, v_dims, drop_out, num_heads = 8):#(C_S, C_T, C_T, C_T,cross_head
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dims // num_heads#
        self.v_dim = v_dims // num_heads#输出张量
        
        self.proj_q1 = nn.Sequential(nn.Linear(in_dim1, self.k_dim * num_heads))
        self.proj_k2 = nn.Sequential(nn.Linear(in_dim2, self.k_dim * num_heads))
        
        self.proj_v2 = nn.Sequential(nn.Linear(in_dim2, self.v_dim * num_heads)) 
        
        self.proj_o = nn.Sequential(nn.Linear(self.v_dim*num_heads, v_dims))
        self.attn_dropout = Dropout(0.25)
        self.proj_dropout = Dropout(drop_out)
        # self.relu = nn.Sigmoid()
    def forward(self, x1, x2, mask=None):

        batch_size, seq_len1, in_dim1 = x1.size()
        _, seq_len2, _ = x2.size()
        q1 = self.proj_q1(x1)
        q1 = q1.view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)#8 4 4097 768
        k2 = self.proj_k2(x2).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)#8 4 768 4097
        v2 = self.proj_v2(x2).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)#8 4 4097 768
        attention = torch.matmul(q1, k2) / self.k_dim ** 0.5#8 4 4096 4096

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(attention, dim=1)
        attention = self.attn_dropout(attention)
        output = torch.matmul(attention, v2).permute(0, 3, 2, 1).contiguous().view(batch_size, -1, self.v_dim*self.num_heads)#  64 4 49 768
        output = self.proj_o(output)
        output = self.proj_dropout(output)
        output = output.view(batch_size, -1, self.v_dim*self.num_heads)
        
        return output, attention



class FeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, relu_dropout = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.dropout = nn.Dropout(relu_dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class FeatureAdapter(nn.Module):
    def __init__(self, channel_in, channel_out, embed_dim, dropout):
        super(FeatureAdapter, self).__init__()
        ####################################
        # The code is being organized
        ####################################

    def forward(self, x):
        ####################################
        # The code is being organized
        ####################################




class FusionText(nn.Module):
    def __init__(self, vision_dim, vision_embed_dim, gene_dim, gene_embed_dim, text_dim, text_embed_dim, out_dim, dropout, ffn_embed_dim):
        super(FusionText, self).__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.gene_dim = gene_dim
        self.out_dim = out_dim
        ####################################
        # The code is being organized
        ####################################


    def forward(self, text_feature,vision_feature, gene_feature ):
        ####################################
        # The code is being organized
        ####################################

class FusionGene(nn.Module):
    def __init__(self, vision_dim, vision_embed_dim, gene_dim, gene_embed_dim, text_dim, text_embed_dim, out_dim, dropout, ffn_embed_dim):
        super(FusionGene, self).__init__()
        ####################################
        # The code is being organized
        ####################################
    def forward(self, text_feature,vision_feature, gene_feature ):

        ####################################
        # The code is being organized
        ####################################

class FusionVision(nn.Module):
    def __init__(self,vision_dim, vision_embed_dim, gene_dim, gene_embed_dim, text_dim, text_embed_dim, out_dim, dropout, ffn_embed_dim):
        super(FusionVision, self).__init__()
        ####################################
        # The code is being organized
        ####################################
    def forward(self, text_feature,vision_feature, gene_feature ):
        ####################################
        # The code is being organized
        ####################################