
import torch
import torch.nn as nn
import torch.nn.functional as F
from feature_fusion import CrossAttention
class StatisSpitialFusion(nn.Module):
    def __init__(self, input_size, hidden_size, embed_size, drop_out, num_layers=1):
        super(StatisSpitialFusion, self).__init__()
        ####################################
        # The code is being organized
        ####################################
        self.initialize_weights()
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        ####################################
        # The code is being organized
        ####################################
        return out
        


