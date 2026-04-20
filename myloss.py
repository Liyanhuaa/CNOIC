from util import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from cluster import gbcluster
from dataloader import*
from sklearn.metrics import pairwise_distances_argmin_min
from init_parameter import *
def euclidean_metric(a, b):
    n = a.shape[0]  # 获取张量a的第一个维度大小，通常表示样本数
    m = b.shape[0]  # 获取张量b的第一个维度大小，通常表示样本数
    a = a.unsqueeze(1).expand(n, m, -1)
    # 将张量a的维度扩展，使其具有三个维度。首先，使用unsqueeze(1)将维度扩展到第二个维度上，然后使用expand来将其复制成(n, m, -1)的形状，其中-1表示该维度的大小由张量自动计算
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = ((a - b) ** 2).sum(dim=2)  #我自己计算的这里表示的是距离，去掉了负号
    return logits


class MarginLoss(nn.Module):
    def __init__(self, num_class, size_average=True):
        super(MarginLoss, self).__init__()
        self.num_class = num_class
        self.size_average = size_average

    def forward(self, classes, labels):
        labels = F.one_hot(labels, self.num_class).float()
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2
        loss = labels * left + 0.5 * (1 - labels) * right
        loss = loss.sum(dim=-1)
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class clusterLoss(nn.Module):

    def __init__(self, args,data):  #num_labels表示大类的个数，feat_dim输出特征的维度，k表示每个大类中聚类后子类的个数

        super(clusterLoss, self).__init__()  # 调用了父类 nn.Module 的构造函数，确保损失函数的正确初始化
        self.num_labels = data.num_labels  #类的数量
        self.feat_dim = args.feat_dim  # 表示特征维度的数量

#返回各个子质心和子质心对应的标签
        self.gbcluster=gbcluster(args,data)
        self.gb_centroids = None #粒球质心
        self.gb_radii = None  #粒球半径
        self.gb_labels = None  #粒球标签
        self.gb_result = None
        self.gb_purity=None


    def forward(self, args, features, labels, noise_labels,index,select=True):
        if select==True:
            gb_for_test_centers, gb_for_test_radius, gb_for_test_label= self.gbcluster.forward(args, features, labels,noise_labels, index,select)
            return gb_for_test_centers, gb_for_test_radius, gb_for_test_label

        else:
            clean_ind_indice, clean_ind_label, clean_ind_center, ood_center ,ind_all_indice,ood_all_indice,gb_for_train_centers, gb_for_train_radius, gb_for_train_label,original_examples= self.gbcluster.forward(args, features, labels,noise_labels, index,select)
            flattened_indices =[int(item) for sublist in clean_ind_indice for item in sublist]
            expanded_labels = [int(clean_ind_label[i]) for i, sublist in enumerate(clean_ind_indice) for _ in sublist]



            return flattened_indices,expanded_labels,clean_ind_center,clean_ind_label,ood_center,ind_all_indice,ood_all_indice,gb_for_train_centers, gb_for_train_radius, gb_for_train_label,original_examples

        self.gb_centroids = torch.tensor(gb_centroids).float().to(features.device)
        self.gb_radii = torch.tensor(gb_radii).float().to(features.device)

        self.gb_labels = torch.tensor(gb_labels).long().to(features.device)
        #self.gb_result = torch.tensor(gb_result).long().to(features.device)
        self.gb_purity = torch.tensor(gb_purity).long().to(features.device)

        # 按照标签排序
        sorted_indices = torch.argsort(self.gb_labels)
        self.gb_centroids = self.gb_centroids[sorted_indices]
        self.gb_radii = self.gb_radii[sorted_indices]

        self.gb_labels = self.gb_labels[sorted_indices]
        self.gb_result = gb_result[sorted_indices]   ##########
        self.gb_purity = self.gb_purity[sorted_indices]

        # 计算样本到质心的欧氏距离

        # 计算分类损失
        loss = self.compute_classification_loss(features, labels, self.gb_centroids, self.gb_labels)


        return self.gb_centroids, self.gb_radii, self.gb_labels, loss





    def compute_classification_loss(self, features, labels, centroids, centroid_labels):
        if features.size(0) == 0:
            return torch.tensor(0.0).to(features.device)

        # 计算样本到质心的欧氏距离
        logits = torch.cdist(features, centroids, p=2)
        distances = torch.full((features.shape[0], self.num_labels), float('inf')).to(logits.device)

        # 更新每个样本对应类别的最小距离
        for label in range(self.num_labels):
            class_mask = (centroid_labels == label)
            if class_mask.any():
                class_distances = logits[:, class_mask]
                distances[:, label] = class_distances.min(dim=1)[0]

        # 归一化距离
        distances = F.normalize(distances, p=1, dim=1)
        # 计算概率
        probabilities = F.softmax(-distances, dim=1)
        # 提取每个样本真实标签对应的概率
        true_probabilities = probabilities[torch.arange(probabilities.size(0)), labels]
        # 计算分类损失
        loss = -torch.log(true_probabilities).mean()
        return loss


