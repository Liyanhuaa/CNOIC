from util import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from cluster import gbcluster
from dataloader import *
from sklearn.metrics import pairwise_distances_argmin_min
from init_parameter import *

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = ((a - b) ** 2).sum(dim=2)
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

    def __init__(self, args,data):
        super(clusterLoss, self).__init__()
        self.num_labels = data.num_labels
        self.feat_dim = args.feat_dim
        self.gbcluster = gbcluster(args,data)
        self.gb_centroids = None
        self.gb_radii = None
        self.gb_labels = None
        self.gb_result = None
        self.gb_purity = None

    def forward(self, args, features, labels, noise_labels,index,select=True):
        if select==True:
            gb_for_test_centers, gb_for_test_radius, gb_for_test_label = self.gbcluster.forward(args, features, labels,noise_labels, index,select)
            return gb_for_test_centers, gb_for_test_radius, gb_for_test_label

        else:
            clean_ind_indice, clean_ind_label, clean_ind_center, ood_center ,ind_all_indice,ood_all_indice,gb_for_train_centers, gb_for_train_radius, gb_for_train_label,original_examples\
                = self.gbcluster.forward(args, features, labels,noise_labels, index,select)
            flattened_indices = [int(item) for sublist in clean_ind_indice for item in sublist]
            expanded_labels = [int(clean_ind_label[i]) for i, sublist in enumerate(clean_ind_indice) for _ in sublist]

            return flattened_indices,expanded_labels,clean_ind_center,clean_ind_label,ood_center,ind_all_indice,ood_all_indice,gb_for_train_centers, gb_for_train_radius, gb_for_train_label,original_examples