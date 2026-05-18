
import torch
import numpy as np 
# from x_transformers import CrossAttender
from torch import nn
from einops import reduce
# from x_transformers import Encoder
from torch.nn import ReLU
import pdb
import math
import pandas as pd
from math import ceil
from torch import nn, einsum
from einops import rearrange, reduce
import torch.nn.functional as F


def exists(val):
    return val is not None



class FeedForward(nn.Module):
    def __init__(self, dim, mult=1, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(self.norm(x))


class MMAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        residual = True,
        residual_conv_kernel = 33,
        eps = 1e-8,
        dropout = 0.,
        num_pathways = 281,
    ):
        super().__init__()
        self.num_pathways = num_pathways
        self.eps = eps
        inner_dim = heads * dim_head

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)

    def forward(self, x, mask=None, return_attn=False):
        b, n, _, h, m, eps = *x.shape, self.heads, self.num_pathways, self.eps

        # derive query, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # set masked positions to 0 in queries, keys, values
        if mask != None:
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        # regular transformer scaling
        q = q * self.scale

        # extract the pathway/histology queries and keys
        q_pathways = q[:, :, :self.num_pathways, :]  # bs x head x num_pathways x dim
        k_pathways = k[:, :, :self.num_pathways, :]

        q_histology = q[:, :, self.num_pathways:, :]  # bs x head x num_patches x dim
        k_histology = k[:, :, self.num_pathways:, :]
        
        # similarities
        einops_eq = '... i d, ... j d -> ... i j'
        cross_attn_histology = einsum(einops_eq, q_histology, k_pathways)
        attn_pathways = einsum(einops_eq, q_pathways, k_pathways)
        cross_attn_pathways = einsum(einops_eq, q_pathways, k_histology)
        
        # softmax
        pre_softmax_cross_attn_histology = cross_attn_histology
        cross_attn_histology = cross_attn_histology.softmax(dim=-1)
        attn_pathways_histology = torch.cat((attn_pathways, cross_attn_pathways), dim=-1).softmax(dim=-1)

        # compute output 
        out_pathways =  attn_pathways_histology @ v
        out_histology = cross_attn_histology @ v[:, :, :self.num_pathways]

        out = torch.cat((out_pathways, out_histology), dim=2)
        
        # add depth-wise conv residual of values
        if self.residual:
            out += self.res_conv(v)

        # merge and combine heads
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)

        if return_attn:  
            # return three matrices
            return out, attn_pathways.squeeze().detach().cpu(), cross_attn_pathways.squeeze().detach().cpu(), pre_softmax_cross_attn_histology.squeeze().detach().cpu()

        return out


class MMAttentionLayer(nn.Module):
    """
    Applies layer norm --> attention
    """

    def __init__(
        self,
        norm_layer=nn.LayerNorm,
        dim=512,
        dim_head=64,
        heads=6,
        residual=True,
        dropout=0.,
        num_pathways = 281,
    ):

        super().__init__()
        self.norm = norm_layer(dim)
        self.num_pathways = num_pathways
        self.attn = MMAttention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            residual=residual,
            dropout=dropout,
            num_pathways=num_pathways
        )

    def forward(self, x=None, mask=None, return_attention=False):

        if return_attention:
            x, attn_pathways, cross_attn_pathways, cross_attn_histology = self.attn(x=self.norm(x), mask=mask, return_attn=True)
            return x, attn_pathways, cross_attn_pathways, cross_attn_histology
        else:
            x = self.attn(x=self.norm(x), mask=mask)

        return x


class SurvPGC_F(nn.Module):
    def __init__(
        self,
        wsi_embedding_dim=1024,
        clinic_embedding_dim=512,
        gene_embedding_dim=768,
        dropout=0.1,
        num_classes=4,
        wsi_projection_dim=256,
        clinic_projection_dim=256,
        gene_projection_dim=256,
        ):
        super(SurvPGC_F, self).__init__()

        #---> general props
        self.num_gene = 4
        self.num_clinic = 6
        self.dropout = dropout

        #---> omics props
        self.gene_embedding_dim = gene_embedding_dim
        self.gene_projection_dim = gene_projection_dim

        self.gene_projection_net = nn.Sequential(
            nn.Linear(self.gene_embedding_dim, self.gene_projection_dim),
        )

        #---> wsi props
        self.wsi_embedding_dim = wsi_embedding_dim 
        self.wsi_projection_dim = wsi_projection_dim

        self.wsi_projection_net = nn.Sequential(
            nn.Linear(self.wsi_embedding_dim, self.wsi_projection_dim),
        )

        #---> clinic props
        self.clinic_embedding_dim = clinic_embedding_dim
        self.clinic_projection_dim = clinic_projection_dim

        self.clinic_projection_net = nn.Sequential(
            nn.Linear(self.clinic_embedding_dim, self.clinic_projection_dim),
        )

        #---> cross attention props
        self.identity = nn.Identity() # use this layer to calculate ig
        self.cross_attender1 = MMAttentionLayer(
            dim=self.wsi_projection_dim,
            dim_head=self.wsi_projection_dim // 2,
            heads=1,
            residual=False,
            dropout=0.1,
            num_pathways=self.num_gene
        )
        self.cross_attender2 = MMAttentionLayer(
            dim=self.wsi_projection_dim,
            dim_head=self.wsi_projection_dim // 2,
            heads=1,
            residual=False,
            dropout=0.1,
            num_pathways=self.num_clinic
        )

        #---> logits props 
        self.num_classes = num_classes
        self.feed_forward = FeedForward(self.wsi_projection_dim // 2, dropout=dropout)
        self.layer_norm = nn.LayerNorm(self.wsi_projection_dim // 2)

        # when both top and bottom blocks 
        self.to_logits = nn.Sequential(
                nn.Linear(self.wsi_projection_dim*2, int(self.wsi_projection_dim/4)),
                nn.ReLU(),
                nn.Linear(int(self.wsi_projection_dim/4), self.num_classes)
            )

    
    def forward(self,x_path, x_omic, x_clinic):

        wsi = x_path
        x_omic = x_omic
        x_clinic = x_clinic
        mask = None
        return_attn = False
        # return_attn = kwargs["return_attn"]
        
        #---> project omic to smaller dimension
        omic_embed = self.gene_projection_net(x_omic)

        #---> project wsi to smaller dimension (same as pathway dimension)
        wsi_embed = self.wsi_projection_net(wsi)

        # ---> project clinic to smaller dimension (same as pathway dimension)
        clinic_embed = self.clinic_projection_net(x_clinic)

        tokens1 = torch.cat([omic_embed, wsi_embed], dim=1)
        tokens1 = self.identity(tokens1)
        tokens2 = torch.cat([clinic_embed, wsi_embed], dim=1)
        tokens2 = self.identity(tokens2)
        
        if return_attn:
            mm_embed1, attn_pathways, cross_attn_genepath, cross_attn_pathgene = self.cross_attender1(x=tokens1, mask=mask if mask is not None else None, return_attention=True)
            mm_embed2, attn_clinic, cross_attn_clipath, cross_attn_pathcli = self.cross_attender2(x=tokens2, mask=mask if mask is not None else None, return_attention=True)
        else:
            mm_embed1 = self.cross_attender1(x=tokens1, mask=mask if mask is not None else None, return_attention=False)
            mm_embed2 = self.cross_attender2(x=tokens2, mask=mask if mask is not None else None, return_attention=False)


            #---> feedforward and layer norm
        mm_embed1 = self.feed_forward(mm_embed1)
        mm_embed1 = self.layer_norm(mm_embed1)
        mm_embed2 = self.feed_forward(mm_embed2)
        mm_embed2 = self.layer_norm(mm_embed2)
        
        #---> aggregate 
        # modality specific mean 
        paths_postSA_embed1 = mm_embed1[:, :self.num_gene, :]
        paths_postSA_embed1 = torch.mean(paths_postSA_embed1, dim=1)

        wsi_postSA_embed1 = mm_embed1[:, self.num_gene:, :]
        wsi_postSA_embed1 = torch.mean(wsi_postSA_embed1, dim=1)

        paths_postSA_embed2 = mm_embed2[:, :self.num_clinic, :]
        paths_postSA_embed2 = torch.mean(paths_postSA_embed2, dim=1)

        wsi_postSA_embed2 = mm_embed2[:, self.num_clinic:, :]
        wsi_postSA_embed2 = torch.mean(wsi_postSA_embed2, dim=1)

        tensor_gp = torch.cat([paths_postSA_embed1, wsi_postSA_embed1], dim=1)
        tensor_cp = torch.cat([paths_postSA_embed2, wsi_postSA_embed2], dim=1)

        # when both top and bottom block
        embedding = torch.cat([paths_postSA_embed1, wsi_postSA_embed1, paths_postSA_embed2, wsi_postSA_embed2], dim=1) #---> both branches
        # embedding = paths_postSA_embed #---> top bloc only
        # embedding = wsi_postSA_embed #---> bottom bloc only
        embedding = self.identity(embedding)

        # embedding = torch.mean(mm_embed, dim=1)
        #---> get logits
        predict = self.to_logits(embedding)

        Y_hat = torch.topk(predict, 1, dim=1)[1]
        Y_prob = F.softmax(predict, dim=1)
        hazards = torch.sigmoid(predict)
        S = torch.cumprod(1 - hazards, dim=1)
        risk_scores = -torch.sum(S, dim=1)
        return hazards, S, Y_hat

        # if kwargs["bag_loss"] == 'nll_diff_surv':
        #     if kwargs["return_attn"] == True:
        #         return tensor_cp, tensor_gp, logits, attn_pathways, cross_attn_genepath, cross_attn_pathgene, attn_clinic, cross_attn_clipath, cross_attn_pathcli
        #     else:
        #         return tensor_cp, tensor_gp, logits
        # else:
        #     if kwargs["return_attn"] == True:
        #         return logits, attn_pathways, cross_attn_genepath, cross_attn_pathgene, attn_clinic, cross_attn_clipath, cross_attn_pathcli
        #     else:
        #         return logits

        
    def captum(self, omic, wsi, clinic):
        
        #---> unpack inputs
        mask = None
        return_attn = False

        #---> get pathway embeddings 
        omic_embed = self.gene_projection_net(omic)

        #---> project wsi to smaller dimension (same as pathway dimension)
        wsi_embed = self.wsi_projection_net(wsi)

        clinic_embed = self.clinic_projection_net(clinic)

        tokens1 = torch.cat([omic_embed, wsi_embed], dim=1)
        tokens1 = self.identity(tokens1)
        tokens2 = torch.cat([clinic_embed, wsi_embed], dim=1)
        tokens2 = self.identity(tokens2)

        if return_attn:
            mm_embed1, attn_pathways, cross_attn_pathhisto, cross_attn_histopath = self.cross_attender1(x=tokens1, mask=mask if mask is not None else None, return_attention=True)
            mm_embed2, attn_clinic, cross_attn_clihisto, cross_attn_histocli = self.cross_attender2(x=tokens2, mask=mask if mask is not None else None, return_attention=True)
        else:
            mm_embed1 = self.cross_attender1(x=tokens1, mask=mask if mask is not None else None, return_attention=False)
            mm_embed2 = self.cross_attender2(x=tokens2, mask=mask if mask is not None else None, return_attention=False)

        #---> feedforward and layer norm 
        mm_embed1 = self.feed_forward(mm_embed1)
        mm_embed1 = self.layer_norm(mm_embed1)
        mm_embed2 = self.feed_forward(mm_embed2)
        mm_embed2 = self.layer_norm(mm_embed2)
        
        #---> aggregate 
        # modality specific mean 
        paths_postSA_embed1 = mm_embed1[:, :self.num_gene, :]
        paths_postSA_embed1 = torch.mean(paths_postSA_embed1, dim=1)

        wsi_postSA_embed1 = mm_embed1[:, self.num_gene:, :]
        wsi_postSA_embed1 = torch.mean(wsi_postSA_embed1, dim=1)

        paths_postSA_embed2 = mm_embed2[:, :self.num_clinic, :]
        paths_postSA_embed2 = torch.mean(paths_postSA_embed2, dim=1)

        wsi_postSA_embed2 = mm_embed2[:, self.num_clinic:, :]
        wsi_postSA_embed2 = torch.mean(wsi_postSA_embed2, dim=1)


        embedding = torch.cat([paths_postSA_embed1, wsi_postSA_embed1, paths_postSA_embed2, wsi_postSA_embed2], dim=1)
        embedding = self.identity(embedding)

        #---> get logits
        logits = self.to_logits(embedding)

        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1)

        if return_attn:
            return risk, attn_pathways, cross_attn_pathhisto, cross_attn_histopath, attn_clinic, cross_attn_clihisto, cross_attn_histocli
        else:
            return risk