
import torch
import torch.nn as nn
import torch.nn.functional as F
from feature_fusion import CrossAttention
from transformers import CLIPTokenizer, CLIPTextModel
from Long_CLIP.model import longclip
# class TextFeature(nn.Module):
#     def __init__(self, model = "clip-vit-large-patch14"):
#         super(TextFeature, self).__init__()
        
#         local_path = f"/home/user1/bingli/4_gene_his_text/{model}"
#         self.tokenizer = CLIPTokenizer.from_pretrained(local_path)
#         self.text_model = CLIPTextModel.from_pretrained(local_path)
#         self.maxtoken_len = 77
#         self.text_model.cuda()
#     def token(self, x):
#         # token = self.tokenizer(x, padding=True, return_tensors="pt")
#         token = self.tokenizer(x, padding=True, truncation=True, max_length=77, return_tensors="pt")
#         return token

#     def forward(self, x):
        
#         with torch.no_grad():
#             text_outputs = self.text_model(**x)
#             text_features = text_outputs.last_hidden_state.mean(dim=1)  # 平均池化得到全局特征
#             return text_features
                
class TextFeature(nn.Module):
    def __init__(self, model = "long-clip"):
        super(TextFeature, self).__init__()

        self.model, preprocess = longclip.load("/home/user1/bingli/4_gene_his_text/survival/Long_CLIP/checkpoints/longclip-L.pt")
        self.model.cuda()
        self.max_length = 77*4-60
    def token(self, x):
        token = longclip.tokenize(x, truncate = True)
        return token

    def forward(self, x):
         with torch.no_grad():
            batch_size, seq_length = x.shape
            if seq_length > self.max_length:
                # 对批次中的每个样本进行处理
                feature_list = []
                for i in range(batch_size):
                    sample_tokens = x[i:i+1, :]  # 保持批次维度
                    
                    if sample_tokens.size(1) > self.max_length:
                        # 分割长序列
                        chunk_features = []
                        for j in range(0, sample_tokens.size(1), self.max_length):
                            chunk = sample_tokens[:, j:j+self.max_length]
                            # 确保chunk长度正确，不足的进行padding
                            if chunk.size(1) < self.max_length:
                                padding_size = self.max_length - chunk.size(1)
                                chunk = torch.cat([chunk, torch.zeros(chunk.size(0), padding_size, dtype=chunk.dtype, device=chunk.device)], dim=1)
                            
                            chunk_feature = self.model.encode_text(chunk)
                            chunk_features.append(chunk_feature)
                        features = torch.cat(chunk_features, dim=1)#按编码长度方向进行拼接
                    feature_list.append(features)  
                text_features = torch.cat(feature_list, dim=0)#按batch_size进行拼接
            else:
                # 如果长度未超过限制，直接编码
                text_features = self.model.encode_text(x)

            return text_features
                
if __name__ == '__main__':
    TextFeature()

