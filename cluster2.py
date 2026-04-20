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





# 判断粒球的标签和纯度
def get_label_and_purity(gb):
    # 分离不同标签数据

    len_label = numpy.unique(gb[:, 1], axis=0)
    # print("len_label\n", len_label)  # 球内所有样本标签（不含重复)

    if len(len_label) == 1:  # 若球内只有一类标签样本，则纯度为1，将该标签定为球的标签
        purity = 1.0
        label = len_label[0]
    else:
        # 矩阵的行数
        num = gb.shape[0]  # 球内样本数
        gb_label_temp = {}
        for label in len_label.tolist():
            # 分离不同标签数据
            gb_label_temp[sum(gb[:, 1] == label)] = label
        # print("分离\n", gb_label_temp)  # dic{该标签对应样本数：标签类别}
        # 粒球中最多的一类数据占整个的比例
        try:
            max_label = max(gb_label_temp.keys())
        except:
            print("+++++++++++++++++++++++++++++++")
            print(gb_label_temp.keys())
            print(gb)
            print("******************************")
            exit()
        purity = max_label / num if num else 1.0  # pur为球内同一标签对应最多样本的类的样本数/球内总样本数

        label = gb_label_temp[max_label]  # 对应样本最多的一类定为球标签
    # print(label)
    # 标签、纯度
    return label, purity

def get_label_and_purity2(gb):
    # 分离不同标签数据

    len_label = numpy.unique(gb[:, 1], axis=0)
    # print("len_label\n", len_label)  # 球内所有样本标签（不含重复)

    if len(len_label) == 1:  # 若球内只有一类标签样本，则纯度为1，将该标签定为球的标签
        purity = 1.0
        label = len_label[0]
    else:
        # 矩阵的行数
        num = gb.shape[0]  # 球内样本数
        gb_label_temp = {}
        for label in len_label.tolist():
            # 分离不同标签数据
            gb_label_temp[sum(gb[:, 1] == label)] = label
        # print("分离\n", gb_label_temp)  # dic{该标签对应样本数：标签类别}
        # 粒球中最多的一类数据占整个的比例
        try:
            max_label = max(gb_label_temp.keys())
        except:
            print("+++++++++++++++++++++++++++++++")
            print(gb_label_temp.keys())
            print(gb)
            print("******************************")
            exit()
        purity = max_label / num if num else 1.0  # pur为球内同一标签对应最多样本的类的样本数/球内总样本数

        label = gb_label_temp[max_label]  # 对应样本最多的一类定为球标签
    # print(label)
    # 标签、纯度
    return label, purity,len_label

# 返回粒球中心和半径
def calculate_center_and_radius(gb):
    data_no_label = gb[:, 2:]  # 第3列往后的所有数据
    sample_indices = gb[:, 0]
    # print("data no label\n",data_no_label)
    center = data_no_label.mean(axis=0)  # 同一列在所有行之间求平均
    # print("center:\n", center)
    distances = np.linalg.norm(data_no_label - center, axis=1)
    radius = numpy.mean((((data_no_label - center) ** 2).sum(axis=1) ** 0.5))
    valid_indices = sample_indices[distances < radius]
    valid_indices=valid_indices.tolist()
    return center, radius,valid_indices,sample_indices

def max_calculate_center_and_radius(gb):
    data_no_label = gb[:, 1:]  # 第2列往后的所有数据
    sample_indices=gb[:, 0]
    # print("data no label\n",data_no_label)
    center = data_no_label.mean(axis=0)  # 同一列在所有行之间求平均
    distances = np.linalg.norm(data_no_label - center, axis=1)
    radius = numpy.max((((data_no_label - center) ** 2).sum(axis=1) ** 0.5))
    valid_indices = sample_indices[distances < radius]
    return center, radius,valid_indices


def splitsyuanlia(args,gb_dict,select):
    len_of_ball = len(gb_dict)
    i = 0
    keys = list(gb_dict.keys())
    while True:
        key = keys[i]
        gb_dict_single = {key: gb_dict[key]}
        gb = gb_dict_single[key][0]
        if gb.shape[0] == 0:
            print(gb.shape)
            i+=1
        else:
            if isinstance(gb, torch.Tensor):
                gb = gb.cpu().numpy()
            gb_dict_single[key][0] = gb

            distances = gb_dict_single[key][1]
            if isinstance(distances, np.ndarray):
                distances = distances.tolist()
            gb_dict_single[key][1] = distances

            label, p = get_label_and_purity(gb)
            number_of_samples_in_ball = len(gb_dict_single[key][0])
            if select==False:
                a_purity = args.purity_train
                a_min_ball = args.min_ball_train
            else:
                a_purity = args.purity_get_ball
                a_min_ball = args.min_ball_get_ball
            if (p < a_purity and number_of_samples_in_ball > a_min_ball) or number_of_samples_in_ball>args.max_ball:
                gb_dict_new = splits_ball(args,gb_dict_single).copy()
                if len(gb_dict_new) > 1:
                    gb_dict.pop(key)
                    gb_dict.update(gb_dict_new)
                    keys.remove(key)
                    keys.extend(gb_dict_new.keys())
                    len_of_ball += len(gb_dict_new) - 1
                else:
                    if len(gb_dict_new) == 0 or key not in gb_dict_new:
                        print(f"Warning: splits_ball returned an empty or invalid result for key {key}")
                    i += 1
            else:
                i += 1
            if i >= len_of_ball:
                break
    return gb_dict


# def splits(args, gb_dict, select):
#     i = 0
#     keys = list(gb_dict.keys())  # 获取所有键的列表
#     while i < len(keys):  # 使用 i 来控制遍历，确保不会超出索引范围
#         key = keys[i]  # 当前处理的键
#         gb_dict_single = {key: gb_dict[key]}  # 提取当前键值对
#         gb = gb_dict_single[key][0]
#
#         if gb.shape[0] == 0:  # 如果粒球为空，跳过并继续处理下一个键
#             print(gb.shape)
#             i += 1
#             continue
#         else:
#             if isinstance(gb, torch.Tensor):
#                 gb = gb.cpu().numpy()  # 如果是Tensor类型，转为numpy数组
#             gb_dict_single[key][0] = gb  # 更新粒球的数据
#
#             distances = gb_dict_single[key][1]
#             if isinstance(distances, np.ndarray):
#                 distances = distances.tolist()  # 如果距离是NumPy数组，转为列表
#             gb_dict_single[key][1] = distances  # 更新距离
#
#             label, p = get_label_and_purity(gb)  # 获取粒球的标签和纯度
#             number_of_samples_in_ball = len(gb_dict_single[key][0])  # 获取粒球中样本的数量
#
#             # 根据select参数选择不同的纯度和最小样本数阈值
#             if select == False:
#                 a_purity = args.purity_train
#                 a_min_ball = args.min_ball_train
#             else:
#                 a_purity = args.purity_get_ball
#                 a_min_ball = args.min_ball_get_ball
#
#             # 如果满足条件，进行粒球的分割
#             if p < a_purity and number_of_samples_in_ball > a_min_ball:
#                 gb_dict_new = splits_ball(args, gb_dict_single).copy()  # 调用分割函数
#                 if len(gb_dict_new) > 1:
#                     gb_dict.pop(key)  # 删除旧的粒球
#                     gb_dict.update(gb_dict_new)  # 更新粒球字典
#                     keys.remove(key)  # 移除已处理的键
#                     keys.extend(gb_dict_new.keys())  # 添加新分割出的键
#                 else:
#                     if len(gb_dict_new) == 0 or key not in gb_dict_new:
#                         print(f"Warning: splits_ball returned an empty or invalid result for key {key}")
#                 # 分割处理完后，继续处理下一个键
#                 i += 1
#             else:
#                 i += 1  # 如果不满足分割条件，继续处理下一个键
#
#     return gb_dict

def splits(args, gb_dict, select):
    gb_dict_pending = gb_dict.copy()  # 待处理
    gb_dict_final = {}  # 最终合格的粒球

    while len(gb_dict_pending) > 0:
        key, value = gb_dict_pending.popitem()  # 取出一个待判断的粒球
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

        # 阈值判断
        if select == False:
            a_purity = args.purity_train
            a_min_ball = args.min_ball_train
        else:
            a_purity = args.purity_get_ball
            a_min_ball = args.min_ball_get_ball

        # 判断是否要继续分裂
        if p < a_purity and number_of_samples_in_ball > a_min_ball:
            gb_dict_new = splits_ball(args, {key: value})  # 分裂为多个粒球
            gb_dict_pending.update(gb_dict_new)  # 新生成的粒球继续等待判断
        else:
            gb_dict_final[key] = value  # 满足条件，放入最终字典

    return gb_dict_final

# 计算距离
def calculate_distances(data, p):  #原始的距离计算公式
    if isinstance(data, torch.Tensor) and isinstance(p, torch.Tensor):
        dis = (data - p).clone().detach() ** 2
        dis = dis.cpu().numpy()
    else:
        dis = (data - p) ** 2
    dis_top10 = np.sort(dis)[-10:]

    return 0.6 * np.sqrt(dis).sum() + 0.4 * np.sqrt(dis_top10).sum()
def calculate_distances2(data, p):
    if isinstance(data, torch.Tensor) and isinstance(p, torch.Tensor):
        dis = (data - p).clone().detach() ** 2
        dis = dis.cpu().numpy()
    else:
        dis = (data - p) ** 2


    return np.sqrt(dis.sum())

"""def calculate_distances(data, p):
    if p.ndim == 1:
        p = p.reshape(1, -1)
    squared_diff = (data - p) ** 2
    squared_distances = squared_diff.sum(axis=1)
    distances = np.sqrt(squared_distances)
    return distances
"""
def splits_ball(args,gb_dict):
    # {center: [gb, distances]}
    center = []
    distances_other_class = []  # 粒球到异类点的距离
    balls = []  # 聚类后的label
    gb_dis_class = []  # 不同标签数据的距离
    center_other_class = []  # 与当前粒球标签不同的类
    center_distances = []  # 新距离
    ball_list = {}  # 最后要返回的字典，键：中心点，值：粒球 + 到中心的距离
    distances_other = []
    distances_other_temp = []

    centers_dict = []  # 中心list
    gbs_dict = []  # 粒球数据list
    distances_dict = []  # 距离list

    # 取出字典中的数据:center,gb,distances
    # 取字典数据，包括键值
    gb_dict_temp = gb_dict.popitem()
    for center_split in gb_dict_temp[0].split('_'):  # 将字典中原来是粒球质心分割，表示成正常的数值形式
        try:
            center.append(float(eval(center_split.strip())))
        except:
            center.append(float(center_split.strip()))
    center = np.array(center)  # 转为array
    centers_dict.append(center)  # 老中心加入中心list
    gb = gb_dict_temp[1][0] # 取出粒球数据
    distances = gb_dict_temp[1][1]  # 取出到老中心的距离


    # 分离不同标签数据的距离
    len_label = numpy.unique(gb[:, 1], axis=0)
    # print(len_label)
    for label in len_label.tolist():
        # 分离不同标签距离
        gb_dis_temp = []
        for i in range(0, len(distances)):
            if gb[i, 1] == label:
                gb_dis_temp.append(distances[i])
        if len(gb_dis_temp) > 0:
            gb_dis_class.append(gb_dis_temp)  # gb_dis_class 存了4个list，每个list是不同标签的样本到质心的距离

    if len(len_label)==1:
        for i in range(0, len(gb_dis_class)):
            # 随机异类点
            set_seed(args.seed)
            ran = random.randint(0, len(gb_dis_class[i]) - 1)
            center_other_temp = gb[distances.index(gb_dis_class[i][ran])]  # 随机点对应的标签和向量 # 随机点对应的标签和向量
            center_other_class.append(center_other_temp)
    else:
    # 取新中心
        for i in range(0, len(gb_dis_class)):

            # 随机异类点
            set_seed(args.seed)
            ran = random.randint(0, len(gb_dis_class[i]) - 1)
            center_other_temp = gb[distances.index(gb_dis_class[i][ran])]  # 随机点对应的标签和向量 # 随机点对应的标签和向量

            if center[1] != center_other_temp[1]:  # 判断新的粒球中心样本和旧类是否是相同的标签
                center_other_class.append(center_other_temp)

    centers_dict.extend(center_other_class)  # 在旧中心的基础上扩展了新的中心 一个旧的中心加上3个新的中心样本（中心样本不代表就是中心）


    distances_other_class.append(distances)  # 所有样本到原来质心的距离
    # 计算到每个新中心的距离
    for center_other in center_other_class:
        balls = []  # 聚类后的label
        distances_other = []
        for feature in gb:
            # 欧拉距离
            distances_other.append(calculate_distances(feature[2:], center_other[2:]))
        # 新中心list

        distances_other_temp.append(distances_other)  # 临时存放到每个新中心的距离
        distances_other_class.append(distances_other)


    # 某一个数据到原中心和新中心的距离，取最小以分类
    for i in range(len(distances)):
        distances_temp = []
        distances_temp.append(distances[i])
        for distances_other in distances_other_temp:
            distances_temp.append(distances_other[i])

        classification = distances_temp.index(min(distances_temp))  # 0:老中心；1,2...：新中心
        balls.append(classification)
    # 聚类情况
    balls_array = np.array(balls)


    # 根据聚类情况，分配数据
    for i in range(0, len(centers_dict)):
        gbs_dict.append(gb[balls_array == i, :])


    # 分配新距离
    i = 0
    for j in range(len(centers_dict)):
        distances_dict.append([])

    for label in balls:
        distances_dict[label].append(distances_other_class[label][i])
        i += 1


    # 打包成字典
    for i in range(len(centers_dict)):
        gb_dict_key = str(centers_dict[i][0])
        for j in range(1, len(centers_dict[i])):
            gb_dict_key += '_' + str(centers_dict[i][j])
        gb_dict_value = [gbs_dict[i], distances_dict[i]]  # 粒球 + 到中心的距离
        ball_list[gb_dict_key] = gb_dict_value

    return ball_list


def main(args,data,select):  # +pur

    # 初始随机中心
    set_seed(args.seed)
    center_init = data[random.randint(0, len(data) - 1), :]  # 任选一行，也就是某个样本作为初始中心
    distance_init = np.array([calculate_distances(feature[2:], center_init[2:]) for feature in data]) #计算所有样本距离初始中心（后64维向量）的欧拉距离：差的平方和的开方

    # 封装成字典
    gb_dict = {}
    gb_dict_key = str(center_init.tolist()[1])  # 质心的标签+向量
    for j in range(2, len(center_init)):
        gb_dict_key += '_' + str(center_init.tolist()[j])

    gb_dict_value = [data, distance_init]  # 所有样本[bs,65]+该样本到中心的距离[1,bs]
    gb_dict[gb_dict_key] = gb_dict_value  # gb_dict 是一个字典，字典里面包含了其对应的 粒球（质心+标签）+粒球中的样本（标签+向量）+样本到质心的距离。


    # 分类划分
    gb_dict = splits(args,gb_dict,select)


    examples=[]
    centers = []
    numbers = []
    radius = []
    len_label=[]
    max_radius = []
    lenss_ball=[]
    p1_ball=[]
    label_ball=[]
    purity=[]
    clean_ind_indices=[]
    index = []
    result = []


    if select==False:
        len_ball=2
        b_purity=0.1
    else:
        len_ball=args.min_ball_select_ball
        b_purity=args.purity_select_ball



    for i in gb_dict.keys():  # 遍历每个球
        gb = gb_dict[i][0]
        if len(gb_dict[i][0]) > len_ball:  # 过滤掉包含样本数量小于5的粒球
            lab, p = get_label_and_purity(gb)
            lenss_ball.append(len(gb_dict[i][-1]))
            p1_ball.append(p)
            label_ball.append(lab)
            if p > b_purity:

                center, radius1, valid_indices,all_indices=calculate_center_and_radius(gb_dict[i][0])
                a = list(center)  # a 每个球center [1,64]
                lab, p,len_label = get_label_and_purity2(gb_dict[i][0])  # 获取每个球标签、纯度
                #a.insert(0, lab)  # 下标为0的位置（首位）插入球标签

                centers.append(a)  # 球的标签和质心
                radius.append(radius1)  # 球的半径
                purity.append(p)
                len_label = len_label.tolist()
                len_label.append(len_label)
                result.append(gb_dict[i][0])  # 球的样本
                clean_ind_indices.append(valid_indices)

                index1 = []
                for j in gb_dict[i][0]:  # 粒球中样本的标签和向量
                    index1.append(j[0])  # 最后一个元素的长度，也就是粒球中样本的个数# 原来括号中是-1，我改成了0
                numbers.append(len(gb_dict[i][-1]))  # 球内部样本数
                index.append(index1)
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





"""numbers : 每个球内部样本数
    result: [ball_samples,ball_label+64维 样本向量]
    centers: [ball_numbers,ball_label+64维 球中心向量]
    radius: [ball_numbers,每个球半径]
"""



