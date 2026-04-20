import ast
import math
import time
import pandas as pd
import random
import numpy
import scipy.io
import matplotlib.pyplot as plt
from sklearn.cluster import k_means
# from config import args
import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import itertools
# import resnet


# 1.输入数据data
# 2.打印绘制原始数据
# 3.判断粒球的纯度
# 4.纯度不满足要求，k-means划分粒球
# 5.绘制每个粒球的数据点
# 6.计算粒球均值，得到粒球中心和半径，绘制粒球


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import itertools

def set_seed(seed):
    random.seed(seed)  # Set Python random seed
    np.random.seed(seed)  # Set numpy random seed
    torch.manual_seed(seed)  # Set PyTorch seed for CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # Set PyTorch seed for all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable benchmark
class Inputexamples(object):
    """A single set of features of data."""

    def __init__(self, numbers,label,centers,radius,purity,len_label,result,clean_ind_indices,all_indice):
        self.numbers = numbers
        self.label=label
        self.centers = centers
        self.radius = radius
        self.purity = purity
        self.len_label = len_label
        self.result = result
        self.clean_ind_indices = clean_ind_indices
        self.all_indice = all_indice

def calculate_center(data):
    return np.mean(data, axis=0)

def calculate_radius(data, center):
    return np.max(np.sqrt(np.sum((data - center) ** 2, axis=1)))





def get_label_and_purity(gb):


    len_label = numpy.unique(gb[:, 1], axis=0)

    if len(len_label) == 1:
        purity = 1.0
        label = len_label[0]
    else:

        num = gb.shape[0]  # 球内样本数
        gb_label_temp = {}
        for label in len_label.tolist():

            gb_label_temp[sum(gb[:, 1] == label)] = label

        try:
            max_label = max(gb_label_temp.keys())
        except:
            print("+++++++++++++++++++++++++++++++")
            print(gb_label_temp.keys())
            print(gb)
            print("******************************")
            exit()
        purity = max_label / num if num else 1.0

        label = gb_label_temp[max_label]
    # print(label)
    # 标签、纯度
    return label, purity

def get_label_and_purity2(gb):


    len_label = numpy.unique(gb[:, 1], axis=0)


    if len(len_label) == 1:
        purity = 1.0
        label = len_label[0]
    else:

        num = gb.shape[0]
        gb_label_temp = {}
        for label in len_label.tolist():

            gb_label_temp[sum(gb[:, 1] == label)] = label

        try:
            max_label = max(gb_label_temp.keys())
        except:
            print("+++++++++++++++++++++++++++++++")
            print(gb_label_temp.keys())
            print(gb)
            print("******************************")
            exit()
        purity = max_label / num if num else 1.0

        label = gb_label_temp[max_label]

    return label, purity,len_label

# 返回粒球中心和半径
def calculate_center_and_radius(gb):
    data_no_label = gb[:, 2:]
    sample_indices = gb[:, 0]
    center = data_no_label.mean(axis=0)
    distances = np.linalg.norm(data_no_label - center, axis=1)
    radius = numpy.mean((((data_no_label - center) ** 2).sum(axis=1) ** 0.5))
    valid_indices = sample_indices[distances < radius]
    valid_indices=valid_indices.tolist()
    return center, radius,valid_indices,sample_indices



def splits(args, gb_dict, select):
    gb_dict_pending = gb_dict.copy()
    gb_dict_final = {}

    while len(gb_dict_pending) > 0:
        key, value = gb_dict_pending.popitem()
        gb = value[0]

        if gb.shape[0] == 0:
            continue
        if isinstance(gb, torch.Tensor):
            gb = gb.cpu().numpy()
        value[0] = gb

        distances = value[1]
        if isinstance(distances, np.ndarray):
            distances = distances.tolist()
        value[1] = distances

        label, p = get_label_and_purity(gb)
        number_of_samples_in_ball = len(gb)

        if select == False:
            a_purity = args.purity_train
            a_min_ball = args.min_ball_train
        else:
            a_purity = args.purity_get_ball
            a_min_ball = args.min_ball_get_ball

        if p < a_purity and number_of_samples_in_ball > a_min_ball:
            gb_dict_new = splits_ball(args, {key: value})
            gb_dict_pending.update(gb_dict_new)
        else:
            gb_dict_final[key] = value

    return gb_dict_final


def calculate_distances(data, p):
    if isinstance(data, torch.Tensor) and isinstance(p, torch.Tensor):
        dis = (data - p).clone().detach() ** 2
        dis = dis.cpu().numpy()
    else:
        dis = (data - p) ** 2
    dis_top10 = np.sort(dis)[-10:]

    return 0.6 * np.sqrt(dis).sum() + 0.4 * np.sqrt(dis_top10).sum()

def splits_ball(args,gb_dict):
    center = []
    distances_other_class = []
    balls = []
    gb_dis_class = []
    center_other_class = []
    ball_list = {}
    distances_other_temp = []
    centers_dict = []
    gbs_dict = []
    distances_dict = []

    gb_dict_temp = gb_dict.popitem()
    for center_split in gb_dict_temp[0].split('_'):
        try:
            center.append(float(eval(center_split.strip())))
        except:
            center.append(float(center_split.strip()))
    center = np.array(center)
    centers_dict.append(center)
    gb = gb_dict_temp[1][0]
    distances = gb_dict_temp[1][1]


    len_label = numpy.unique(gb[:, 1], axis=0)
    for label in len_label.tolist():
        gb_dis_temp = []
        for i in range(0, len(distances)):
            if gb[i, 1] == label:
                gb_dis_temp.append(distances[i])
        if len(gb_dis_temp) > 0:
            gb_dis_class.append(gb_dis_temp)

    if len(len_label)==1:
        for i in range(0, len(gb_dis_class)):
            # 随机异类点
            set_seed(args.seed)
            ran = random.randint(0, len(gb_dis_class[i]) - 1)
            center_other_temp = gb[distances.index(gb_dis_class[i][ran])]
            center_other_class.append(center_other_temp)
    else:

        for i in range(0, len(gb_dis_class)):

            set_seed(args.seed)
            ran = random.randint(0, len(gb_dis_class[i]) - 1)
            center_other_temp = gb[distances.index(gb_dis_class[i][ran])]

            if center[1] != center_other_temp[1]:
                center_other_class.append(center_other_temp)

    centers_dict.extend(center_other_class)


    distances_other_class.append(distances)
    for center_other in center_other_class:
        balls = []
        distances_other = []
        for feature in gb:
            distances_other.append(calculate_distances(feature[2:], center_other[2:]))

        distances_other_temp.append(distances_other)  # 临时存放到每个新中心的距离
        distances_other_class.append(distances_other)

    for i in range(len(distances)):
        distances_temp = []
        distances_temp.append(distances[i])
        for distances_other in distances_other_temp:
            distances_temp.append(distances_other[i])

        classification = distances_temp.index(min(distances_temp))  # 0:老中心；1,2...：新中心
        balls.append(classification)

    balls_array = np.array(balls)


    for i in range(0, len(centers_dict)):
        gbs_dict.append(gb[balls_array == i, :])

    i = 0
    for j in range(len(centers_dict)):
        distances_dict.append([])

    for label in balls:
        distances_dict[label].append(distances_other_class[label][i])
        i += 1

    for i in range(len(centers_dict)):
        gb_dict_key = str(centers_dict[i][0])
        for j in range(1, len(centers_dict[i])):
            gb_dict_key += '_' + str(centers_dict[i][j])
        gb_dict_value = [gbs_dict[i], distances_dict[i]]  # 粒球 + 到中心的距离
        ball_list[gb_dict_key] = gb_dict_value

    return ball_list


def main(args,data,select):  # +pur

    set_seed(args.seed)
    center_init = data[random.randint(0, len(data) - 1), :]
    distance_init = np.array([calculate_distances(feature[2:], center_init[2:]) for feature in data])

    gb_dict = {}
    gb_dict_key = str(center_init.tolist()[1])
    for j in range(2, len(center_init)):
        gb_dict_key += '_' + str(center_init.tolist()[j])

    gb_dict_value = [data, distance_init]
    gb_dict[gb_dict_key] = gb_dict_value
    gb_dict = splits(args,gb_dict,select)
    examples=[]
    lenss_ball=[]
    p1_ball=[]
    label_ball=[]

    if select==False:
        len_ball=2
        b_purity=0.1
    else:
        len_ball=args.min_ball_select_ball
        b_purity=args.purity_select_ball


    for i in gb_dict.keys():
        gb = gb_dict[i][0]
        if len(gb_dict[i][0]) > len_ball:
            lab, p = get_label_and_purity(gb)
            lenss_ball.append(len(gb_dict[i][-1]))
            p1_ball.append(p)
            label_ball.append(lab)
            if p > b_purity:

                center, radius1, valid_indices,all_indices=calculate_center_and_radius(gb_dict[i][0])
                a = list(center)
                lab, p,len_label = get_label_and_purity2(gb_dict[i][0])
                len_label = len_label.tolist()
                examples.append(
                    Inputexamples(numbers=len(gb_dict[i][-1]),
                                  label=lab,
                                  centers=a,
                                  radius=radius1,
                                  purity=p,
                                  len_label=len_label,
                                  result=gb_dict[i][0],
                                  clean_ind_indices=valid_indices,
                                  all_indice=all_indices))
    original_examples=examples
    return examples,original_examples







