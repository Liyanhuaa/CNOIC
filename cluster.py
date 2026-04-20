from util import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
import math
import numpy as np
from init_parameter import *
from dataloader import *
import torch.distributed as dist
import ast
import time
import pandas as pd
import random
import scipy.io
import matplotlib.pyplot as plt
from sklearn.cluster import k_means
from scipy import stats
from dataloader import *
from init_parameter import *
import cluster2 as new_GBNR

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class gbcluster(nn.Module):

    def __init__(self,args,data):
        super(gbcluster, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, args, features, labels,noise_labels,index,select):
        if select == False:
            a_purity = args.purity_train
        else:
            a_purity = args.purity_get_ball

        noise_label_features = torch.cat((noise_labels.reshape(-1, 1), features), dim=1)
        index_label_features = torch.cat((index.reshape(-1, 1), noise_label_features), dim=1)
        out = torch.cat((labels.reshape(-1, 1),index_label_features), dim=1)
        pur_tensor = torch.Tensor([[a_purity]] * out.size(0))
        out = torch.cat((pur_tensor.to(self.device), out), dim=1)
        examples,original_examples= GBNR.apply(args,out.to(self.device),select)
        len_known_label = np.unique(noise_labels.cpu().numpy())
        clean_ind_indice=[]
        clean_ind_label=[]
        clean_ind_center=[]
        ind_all_indice=[]
        ood_all_indice=[]
        ood_center = []
        gb_for_train_centers=[]
        gb_for_train_label = []
        gb_for_train_radius = []

        gb_for_test_centers=[]
        gb_for_test_label = []
        gb_for_test_radius = []

        if  select==True:

            for example in examples:
                gb_for_test_centers.append(example.centers)
                gb_for_test_label.append(example.label)
                gb_for_test_radius.append(example.radius)
            return gb_for_test_centers, gb_for_test_radius, gb_for_test_label

        else:
            p_noise_ind = args.p_noise_ind
            p_noise_ood = args.p_noise_ood
            n_noise = args.n_noise
            for example in examples:
                if example.purity > p_noise_ind  and example.numbers > n_noise :
                    clean_ind_indice.append(example.clean_ind_indices)
                    ind_all_indice.append(example.all_indice)
                    clean_ind_label.append(example.label)
                    clean_ind_center.append(example.centers)

                    gb_for_train_centers.append(example.centers)
                    gb_for_train_label.append(example.label)
                    gb_for_train_radius.append(example.radius)

                if example.purity < p_noise_ood and example.numbers <n_noise:
                    ood_center.append(example.centers)
                    ood_all_indice.append(example.all_indice)

            print("clean_ind_center 数量:", len(clean_ind_center))
            print("ood_center 数量:", len(ood_center))

            return clean_ind_indice, clean_ind_label, clean_ind_center, ood_center, ind_all_indice, ood_all_indice,gb_for_train_centers, gb_for_train_radius, gb_for_train_label,original_examples

def calculate_distances(center, p):
    return ((center - p) ** 2).sum(axis=0) ** 0.5

class GBNR(torch.autograd.Function):
    @staticmethod
    def forward(self,args, input_,select):

        self.batch_size = input_.size(0)
        input_main = input_[:, 2:]
        self.input = input_[:, 4:]
        self.res = input_[:, 1:2]
        self.index = input_[:, 2:3]
        self.noise_labels = input_[:, 3:4]
        pur = input_[:, 0].cpu().numpy().tolist()[0]

        self.flag = 0
        examples ,original_examples= new_GBNR.main(args,input_main,select)
        print("example.shape",len(examples))
        return  examples,original_examples

    @staticmethod
    def backward(self, output_grad, input, index, id, _):
        result = np.zeros([self.batch_size, 154], dtype='float64')

        for i in range(output_grad.size(0)):
            for a in self.balls[i]:
                input_np = np.array(self.input)
                a_np = np.array(a[1:])

                if input_np.shape[1:] == a_np.shape:
                    mask = (input_np == a_np).all(axis=1)
                    if mask.any():
                        result[mask, 4:] = output_grad[i, :].cpu().numpy()

        return torch.Tensor(result)