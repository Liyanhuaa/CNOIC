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
    random.seed(seed)  # Set Python random seed
    np.random.seed(seed)  # Set numpy random seed
    torch.manual_seed(seed)  # Set PyTorch seed for CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # Set PyTorch seed for all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable benchmark

class gbcluster(nn.Module):

    def __init__(self,args,data):
        super(gbcluster, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, args, features, labels,noise_labels,index,select):
        if select == False:
            a_purity = args.purity_train
        else:
            a_purity = args.purity_get_ball
        #index1=torch.arange(len(labels)).to('cuda:0')   #生成1到样本长度的粒球 并打乱顺序
        noise_label_features = torch.cat((noise_labels.reshape(-1, 1), features), dim=1)  #
        index_label_features = torch.cat((index.reshape(-1, 1), noise_label_features), dim=1)  #
        out = torch.cat((labels.reshape(-1, 1),index_label_features), dim=1)
        pur_tensor = torch.Tensor([[a_purity]] * out.size(0))
        out = torch.cat((pur_tensor.to(self.device), out), dim=1)
        examples,original_examples= GBNR.apply(args,out.to(self.device),select)
        len_known_label = np.unique(noise_labels.cpu().numpy())
        clean_ind_indice=[]#这个索引是符合距离小于半径的样本的索引
        clean_ind_label=[]
        clean_ind_center=[]
        ind_all_indice=[] #包含的ind粒球的所有样本的索引
        ood_all_indice=[]#被认为是ood的粒球中包含的所有样本的索引
        ood_center = []
        gb_for_train_centers=[]
        gb_for_train_label = []
        gb_for_train_radius = []


        gb_for_test_centers=[]
        gb_for_test_label = []
        gb_for_test_radius = []

        if  select==True: #说明是最后阶段挑选粒球，这个是上一级已经挑选过了，这一集直接拿出去就行了

            for example in examples:
                gb_for_test_centers.append(example.centers)
                gb_for_test_label.append(example.label)
                gb_for_test_radius.append(example.radius)
            return gb_for_test_centers, gb_for_test_radius, gb_for_test_label


        else:  #说明是在训练过程中，这一步是进行噪声分类
            p_noise_ind = args.p_noise_ind#0.8
            p_noise_ood = args.p_noise_ood#0.5
            n_noise = args.n_noise #5
            for example in examples:
                if example.purity > p_noise_ind  and example.numbers > n_noise :
                    clean_ind_indice.append(example.clean_ind_indices)
                    ind_all_indice.append(example.all_indice)
                    clean_ind_label.append(example.label)
                    clean_ind_center.append(example.centers)

                    gb_for_train_centers.append(example.centers)
                    gb_for_train_label.append(example.label)
                    gb_for_train_radius.append(example.radius)


                if example.purity < p_noise_ood and example.numbers <n_noise: #and len(example.len_label)>len(len_known_label)-2:  #其他数据集的都是0.5和5  改成0.6 和20   在0.25 0.3的情况下改成0.7和20
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
        # 在forward中，需要定义MyReLU这个运算的forward计算过程
        # 同时可以保存任何在后向传播中需要使用的变量值
        self.batch_size = input_.size(0)
        input_main = input_[:, 2:]  # noise_label+64 [bs,65]
        self.input = input_[:, 4:]  # backward中使用，只有向量
        self.res = input_[:, 1:2]  # 样本原标签
        self.index = input_[:, 2:3]  # 样本原始索引
        self.noise_labels = input_[:, 3:4]
        pur = input_[:, 0].cpu().numpy().tolist()[0]  # 从第0维取出纯度

        self.flag = 0
        examples ,original_examples= new_GBNR.main(args,input_main,select)  # 加了pur
        print("example.shape",len(examples))
        return  examples,original_examples

    @staticmethod
    def backward(self, output_grad, input, index, id, _):
        # 初始化结果数组
        result = np.zeros([self.batch_size, 154], dtype='float64')  # +4

        for i in range(output_grad.size(0)):
            # 将 balls[i] 转换为 numpy 数组，处理过程中形状可能不一致
            for a in self.balls[i]:
                input_np = np.array(self.input)
                a_np = np.array(a[1:])

                # 确保 a_np 和 input_np 的形状一致
                if input_np.shape[1:] == a_np.shape:
                    mask = (input_np == a_np).all(axis=1)
                    if mask.any():
                        result[mask, 4:] = output_grad[i, :].cpu().numpy()

        return torch.Tensor(result)


