import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader,ConcatDataset, random_split
import pandas
from torchvision import transforms, datasets
from PIL import Image
import pandas as pd
import random 
class GeneHistopDataset(Dataset):
    def __init__(self, gene_expression_file, select_data ,image_dir, diagnostic_reports, token, transform=None):
        self.gene_data = pd.read_csv(gene_expression_file)
        self.image_dir = image_dir
        self.diagnostic_reports_data = pd.read_csv(diagnostic_reports)
        self.transform = transform
        self.token = token
        self.text_len = self.diagnostic_reports_data.iloc[:, 2].astype(str).str.len().max()
        self.gene = None
        self.img = None
        self.label = None
        self.event_time=None
        self.survival_status = None
        self.sample_id = None
        self.diagnostic = None
        #筛选需要的基因
        # self.gene_data = self.gene_data[select_data]
        # 记录有效样本的索引
        self.valid_indices = []
        for idx in range(len(self.gene_data)):
            sample_name = self.gene_data.iloc[idx, 1]
            image_path = os.path.join(self.image_dir, f"{sample_name}.pt")
            if os.path.exists(image_path):
                if (self.diagnostic_reports_data['sampleName'] == sample_name).any():
                    self.valid_indices.append(idx)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # 使用 valid_indices 获取有效样本的真实索引
        real_idx = self.valid_indices[idx]
        # 获取样本名和基因表达数据
        sample_id = self.gene_data.iloc[real_idx, 0]
        self.sample_id = sample_id
        sample_name = self.gene_data.iloc[real_idx, 1]
        gene_expression = self.gene_data.iloc[real_idx, 6:-1].values.astype(float)
        time = self.gene_data.iloc[real_idx, 2] 
        self.event_time = torch.tensor(time ,dtype=torch.float32)
        self.survival_status = torch.tensor(self.gene_data.iloc[real_idx, 3])
        #反转生存状态标签0-死，1活
        self.survival_status = 1-self.survival_status
        label = self.gene_data.iloc[real_idx, 4]

        # 读取对应的病理图像特征
        image_feat_path = os.path.join(self.image_dir, f"{sample_name}.pt")
        image_feat  = torch.load(image_feat_path,weights_only=True)
        dig_sample_name = f"{sample_name}_{random.randint(0, 4)}"#随机选择相同样本对应的5份诊断报告中的一份
        matching_rows = self.diagnostic_reports_data[
                            self.diagnostic_reports_data.iloc[:, 0] == dig_sample_name
                        ]

        text = matching_rows.iloc[0, 2]
        if len(text) < self.text_len:
            text = text.ljust(self.text_len)  # 右侧补空格
        else:
            text = text[:self.text_len]  # 截取前256个字符
        # self.diagnostic = self.token.token(text)
        self.diagnostic = text
        self.gene = torch.tensor(gene_expression, dtype=torch.float32)
        self.label = torch.tensor(label)
        self.img =  image_feat
        return self.gene, self.label, self.img, self.diagnostic,self.event_time, self.survival_status,self.sample_id
    def get_size(self):
        real_idx = self.valid_indices[0]
        sample_name = self.gene_data.iloc[real_idx, 1]
        gene_expression = self.gene_data.iloc[real_idx, 6:-1].values.astype(float)
        image_path = os.path.join(self.image_dir, f"{sample_name}.pt")
        image  = torch.load(image_path, weights_only=True)
        return gene_expression.shape, image.shape

def geneHistopDataloader(geneFile, selectData, imgDir, diagnostic_reports, token, batchSize = 8,train_ratio = 0.8):
    #transformer中奖图片resize为1024x1024
    transformTrain = transforms.Compose(
            [transforms.Resize((1024, 1024)),
             transforms.RandomCrop(1024, padding=4), 
             transforms.RandomHorizontalFlip(), transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),])
    transformTest = transforms.Compose(
            [   transforms.Resize((1024, 1024)),
                transforms.ToTensor(), 
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),])

    Dataset = GeneHistopDataset(geneFile, selectData, imgDir,diagnostic_reports, token, transformTrain)
    total_size = len(Dataset)
    print("Total data sample : " + str(total_size))
    train_size = int(total_size * train_ratio)
    test_size = total_size - train_size  # 剩余部分作为测试集
    # 使用 random_split 进行分割

    # torch.manual_seed(75)#Brain

    # torch.manual_seed(42)
    
    # torch.manual_seed(56)
    # torch.manual_seed(43)
    torch.manual_seed(48)
    # torch.manual_seed(56)
    train_dataset, test_dataset = random_split(Dataset, [train_size, test_size])
    trainDataloader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, num_workers=1)
    testDataloader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False, num_workers=1)
    gene_size, img_feat_size = Dataset.get_size()
    return trainDataloader, testDataloader, gene_size, img_feat_size
