from model import *
from dataloader import *
from util import *
from pytorch_pretrained_bert.optimization import BertAdam
from myloss import *
import matplotlib.pyplot as plt
from torch.optim import AdamW
from transformers import get_scheduler
# from transformers import AdamW, get_scheduler
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib import colors as mcolors
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from utils import util
def set_seed(seed):
    random.seed(seed)  # Set Python random seed
    np.random.seed(seed)  # Set numpy random seed
    torch.manual_seed(seed)  # Set PyTorch seed for CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # Set PyTorch seed for all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable benchmark
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
class PretrainModelManager:

    def __init__(self, args, data):
        # 定义和载入模型
        self.model = BertForModel.from_pretrained(args.bert_model, cache_dir="", num_labels=data.num_labels)
        # freezed 11层参数，12层参与训练
        if args.freeze_bert_parameters:
            for name, param in self.model.bert.named_parameters():
                param.requires_grad = False
                if "encoder.layer.11" in name or "pooler" in name:
                    param.requires_grad = True
        # gpu 运算
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        # 总的计算次数（backward）
        self.num_train_optimization_steps = len(data.train_dataloader) * args.num_train_epochs

        self.optimizer = self.get_optimizer(args)
        self.optimizer2 = self.get_optimizer2(args)

        self.best_eval_score = 0
        self.best_evaluation_score = 0 #后续自己的训练过程中的最优分数
        self.best_original_examples=None
        self.clusterLoss = clusterLoss(args, data)
        self.gb_centroids = None
        self.gb_radii = None
        self.gb_labels = None
    def euclidean_metric(self,a, b):
        n = a.shape[0]
        m = b.shape[0]
        a = a.unsqueeze(1).expand(n, m, -1)
        b = b.unsqueeze(0).expand(n, m, -1)
        return torch.sqrt(torch.pow(a - b, 2).sum(2))
    def open_classify(self, data, features, gb_centroids, gb_radii, gb_labels):
        # 计算输入特征与所有质心之间的欧氏距离
        logits = self.euclidean_metric(features, gb_centroids)
        _, preds = logits.min(dim=1)  # 获取最小距离的索引，对应最近的质心
        print("Features shape:", features.shape)
        print("Indexed centroids shape:", gb_centroids[preds].shape)

        # 计算每个输入特征与其预测类别质心之间的实际欧氏距离
        euc_dis = torch.norm(features - gb_centroids[preds], dim=1)

        # 初始化分类结果为未知类别
        final_preds = torch.tensor([data.unseen_token_id] * features.shape[0], device=features.device)

        # 对每个样本进行分类
        for i in range(features.shape[0]):
            #if euc_dis[i] < gb_radii[preds[i]]:
            final_preds[i] = gb_labels[preds[i]]
            # else:
            #     final_preds[i] = data.unseen_token_id  # 如果距离大于半径，标记为未知类别

        return final_preds

#这个是验证函数  这个函数用来做消融实验  基于交叉熵损失的分类过程
    def eval(self, args, data):  # 用来评估深度学习模型

        self.model.eval()  # 评估模式，只进行前向传播，用于预测，不进行梯度计算
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)  # 储存真实标签
        total_logits = torch.empty((0, data.num_labels)).to(self.device)  # 储存预测输出

        for batch in tqdm(data.eval_dataloader, desc="Iteration"):  # 循环，遍历了数据集中的每个批次
            batch = tuple(t.to(self.device) for t in batch)  # 将数据批次中的所有张量 都移动到模型所在的设备
            input_ids, input_mask, segment_ids, label_ids, label_noiseids, index = batch
            with torch.set_grad_enabled(False):  # 它将梯度计算关闭，意味着在这个上下文中，不会进行梯度计算
                _, logits = self.model(input_ids, segment_ids, input_mask,
                                       mode='eval')  # 将数据批次中的所有张量（input_ids、input_mask、segment_ids、label_ids）都移动到模型所在的设备
                total_labels = torch.cat((total_labels, label_ids))  # 当前批次的真实标签 label_ids 连接到先前的批次数据上，以逐步积累所有的标签
                total_logits = torch.cat((total_logits, logits))  # 将当前批次模型的预测 logits 连接到先前的批次数据上，以逐步积累所有的预测

        total_probs, total_preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)  # 根据输入的logit输出   total_probs 包含了每个样本的最大概率值，第二个张量 total_preds 包含了对应的预测类别索引
        y_pred = total_preds.cpu().numpy()  # 将预测值 total_preds  转换为 NumPy 数组，并将其移到 CPU 上，以便进行后续的计算。
        y_true = total_labels.cpu().numpy()  # 将 真实标签 total_labels 转换为 NumPy 数组，并将其移到 CPU 上，以便进行后续的计算。
        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        return acc

    # 利用粒球的方式进行评估的代码，输入每个类别中粒球的质心和半径，以及待测样本，给出样本从预测类别。
    def evaluation(self, args, data, gb_centroids, gb_radii, gb_labels,mode="eval"):
        self.model.eval()

        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_preds = torch.empty(0, dtype=torch.long).to(self.device)
        if mode == 'eval':
            dataloader = data.eval_dataloader
        elif mode == 'test':
            dataloader = data.test_dataloader

        for batch in tqdm(dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, label_noiseids, index = batch
            with torch.set_grad_enabled(False):  # 上下文管理器，用于暂时关闭梯度计算
                pooled_output, _ = self.model(input_ids, segment_ids,
                                              input_mask)  # 得到模型的输出特征表示 pooled_output。通常，这里的 _ 是一个占位符，用于接收不需要的输出
                preds = self.open_classify(data,pooled_output,gb_centroids, gb_radii, gb_labels)  # 将模型的输出特征表示输入到 self.open_classify 方法中，用于进行分类预测

                total_labels = torch.cat((total_labels, label_ids))  # 将当前批次的标签和预测结果追加到总的标签和预测张量中。
                total_preds = torch.cat((total_preds, preds))

        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        acc = round(accuracy_score(y_true, y_pred) * 100, 2)

        return acc

    def compute_cluster_loss(self, clean_ind_label, ind_all_indice, ood_all_indice, accumulated_features,
                             accumulated_indexs):

        ind_centers = []
        for single_indice in ind_all_indice:
            find_indices = torch.tensor(single_indice, dtype=torch.long).to('cuda:0')
            find_positions = torch.nonzero(accumulated_indexs.unsqueeze(1) == find_indices, as_tuple=False)[:, 0]
            feature_for_center = accumulated_features[find_positions].to('cuda:0')
            center = feature_for_center.mean(axis=0)
            ind_centers.append(center)

        ood_centers = []
        for single_indice in ood_all_indice:
            find_indices = torch.tensor(single_indice, dtype=torch.long).to('cuda:0')
            find_positions = torch.nonzero(accumulated_indexs.unsqueeze(1) == find_indices, as_tuple=False)[:, 0]
            feature_for_center = accumulated_features[find_positions].to('cuda:0')
            center = feature_for_center.mean(axis=0)
            ood_centers.append(center)

        ind_centers = torch.stack(ind_centers) if ind_centers else torch.empty(0).to(
            'cuda:0')  # If ood_centers is empty, initia
        ood_centers = torch.stack(ood_centers) if ood_centers else torch.empty(0).to(
            'cuda:0')  # If ood_centers is empty, initialize it as an empty tensor
        if ind_centers.size(0) > 0:
            pairwise_dist_clean = torch.cdist(ind_centers, ind_centers,p=2)  # 欧几里得距离, shape [num_clean_centers, num_clean_centers]

        # 计算 clean_ind_center 和 ood_center 的 pairwise 距离矩阵
        if ood_centers.size(0) and ind_centers.size(0) > 0:  # Only calculate if ood_centers is not empty
            pairwise_dist_ood = torch.cdist(ind_centers, ood_centers, p=2)

        inter_loss = 0.0  # 不同类别的质心远离
        intra_loss = 0.0  # 同类别的质心靠近
        ood_loss = 0.0  # clean_ind_center 和 ood_center 远离

        # 1. 同类别质心靠近
        if ind_centers.size(0) > 0:
            for i in range(ind_centers.size(0)):
                for j in range(ind_centers.size(0)):
                    if i == j:
                        continue  # 跳过自己
                    if clean_ind_label[i] == clean_ind_label[j]:
                        intra_loss += pairwise_dist_clean[i, j] ** 2  # 同类别距离平方和

        # 2. 不同类别质心远离
        if ind_centers.size(0) > 0:
            for i in range(ind_centers.size(0)):
                for j in range(ind_centers.size(0)):
                    if i == j:
                        continue  # 跳过自己
                    if clean_ind_label[i] != clean_ind_label[j]:
                        inter_loss += 1.0 / (pairwise_dist_clean[i, j] ** 2 + 1e-6)  # 不同类别距离倒数和

        # 3. clean_ind_center 和 ood_center 远离
        if ood_centers.size(0) and ind_centers.size(0)> 0:  # Only calculate if ood_centers is not empty
            for i in range(ind_centers.size(0)):
                for j in range(ood_centers.size(0)):
                    ood_loss += 1.0 / (pairwise_dist_ood[i, j] ** 2 + 1e-6)  # clean 和 ood 的距离倒数和

        # 归一化损失
        num_clean_centers = ind_centers.size(0)
        num_ood_centers = ood_centers.size(0)
        if ind_centers.size(0) > 0:
            intra_loss = intra_loss / (num_clean_centers * (num_clean_centers - 0.9))  # 归一化#原本是减去1  改成了减去0.9
            inter_loss = inter_loss / (num_clean_centers * (num_clean_centers - 0.9))  # 归一化
        if ood_centers.size(0)>0 and ind_centers.size(0) > 0:
            ood_loss = ood_loss / (
                    num_clean_centers * num_ood_centers) if num_ood_centers > 0 else 0  # Avoid division by zero
        if ind_centers.size(0) > 0 and ood_centers.size(0)>0 :
            total_loss = intra_loss + inter_loss + ood_loss

        if ind_centers.size(0) ==0 and ood_centers.size(0) > 0:
            total_loss = 0
        if ind_centers.size(0) >0 and ood_centers.size(0) == 0:
            total_loss = intra_loss + inter_loss

        return intra_loss,inter_loss,ood_loss

    def train(self, args, data):
        wait = 0  # 计数器，用于跟踪模型性能没有改善的次数
        wait_ood_train=0
        best_model = None  # 用于存储在训练过程中性能最好的模型
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):  # trange创建进度条
            self.model.train()  # 将模型设置为训练模式。在训练模式下，模型会进行前向传播和反向传播，并根据损失函数更新模型参数
            tr_loss = 0  # 用于跟踪当前时代的训练损失和训练样本的数量
            nb_tr_examples, nb_tr_steps = 0, 0

            if epoch < args.warm_train_epoch:
                for step, batch in enumerate(data.train_dataloader):  # 在每次迭代中，enumerate() 返回两个值，一个是索引（step），另一个是可迭代对象中的元素（batch）
                    batch = tuple(t.to(self.device) for t in
                                  batch)  # 这行代码将数据批次中的所有张量（input_ids、input_mask、segment_ids、label_ids）都移动到模型所在的设备（通常是 GPU）上
                    input_ids, input_mask, segment_ids, label_ids,label_noiseids,index= batch
                    batch_number = len(data.train_dataloader)
                    with ((torch.set_grad_enabled(True))):

                            loss1 = self.model(input_ids, segment_ids, input_mask, label_noiseids,mode="train")  # 这行代码调用模型进行前向传播，并计算损失

                            self.optimizer.zero_grad()  # 将优化器的梯度信息归零，以准备进行下一次反向传播
                            loss1.backward()  # 执行反向传播，并根据损失函数更新模型的参数
                            self.optimizer.step()
                            tr_loss += loss1.item()
                            util.summary_writer.add_scalar("Loss/loss1", loss1.item(), step + epoch * batch_number)
                            nb_tr_examples += input_ids.size(0)
                            nb_tr_steps += 1
                    # 获取特征时不计算梯度
                loss = tr_loss / nb_tr_steps
                print('train_loss', loss)
                #eval_score = self.evaluation(args, data, gb_for_train_centers, gb_for_train_radius,gb_for_train_label)  # 这一行去掉代表没有根据聚类进行表征学习的
                eval_score = self.eval(args, data)
                print('eval_score', eval_score)
                if eval_score > self.best_eval_score:
                    best_model = copy.deepcopy(self.model)  # 如果当前评估分数更好，将当前模型保存为最佳模型
                    wait = 0
                    self.best_eval_score = eval_score
                else:
                    wait += 1
                    if wait >= args.wait_patient:  # 如果 wait 达到了一个指定的阈值（args.wait_patient），则结束训练循环
                        break

                # self.model = best_model  # 将最佳模型设置为训练结束后的模型
                # if args.save_model:  # 检查是否存在一个名为args.save_model的参数，如果指定了保存模型的选项，调用 self.save_model 方法来保存训练后的模型
                #     self.save_model(args)
            else:
                memory_bank = []
                memory_bank_label = []
                memory_bank_noise_label = []
                memory_bank_index = []
                memory_bank_input_ids = []
                memory_bank_input_mask = []
                memory_bank_segment_ids = []


                for step, batch in enumerate(data.train_dataloader):  # 在每次迭代中，enumerate() 返回两个值，一个是索引（step），另一个是可迭代对象中的元素（batch）
                    batch = tuple(t.to(self.device) for t in
                                  batch)  # 这行代码将数据批次中的所有张量（input_ids、input_mask、segment_ids、label_ids）都移动到模型所在的设备（通常是 GPU）上
                    input_ids, input_mask, segment_ids, label_ids, label_noiseids, index = batch
                    batch_number = len(data.train_dataloader)
                    with ((torch.set_grad_enabled(True))):

                        features = self.model(input_ids, segment_ids, input_mask, feature_ext="True")
                        memory_bank.append(features.cpu())  # 将特征移到CPU以节省GPU内存
                        memory_bank_label.append(label_ids.cpu())
                        memory_bank_input_ids.append( input_ids.cpu())
                        memory_bank_input_mask.append(input_mask.cpu())
                        memory_bank_segment_ids.append(segment_ids.cpu())
                        memory_bank_noise_label.append(label_noiseids.cpu())
                        memory_bank_index.append(index.cpu())

                        # if (step + 1) == batch_number:
                        if args.dataset=='snips':
                            bbb=20
                        if args.dataset=='banking':
                            bbb=15
                        if args.dataset == 'stackoverflow':
                            bbb = 10

                        if (step + 1) % bbb==0:

                            accumulated_features = torch.cat(memory_bank, dim=0).to('cuda:0')
                            accumulated_labels = torch.cat(memory_bank_label, dim=0).to('cuda:0')
                            accumulated_noise_labels = torch.cat(memory_bank_noise_label, dim=0).to('cuda:0')
                            accumulated_indexs=torch.cat(memory_bank_index, dim=0).to('cuda:0')
                            memory_bank_input_ids=torch.cat(memory_bank_input_ids, dim=0).to('cuda:0')
                            memory_bank_input_mask=torch.cat(memory_bank_input_mask, dim=0).to('cuda:0')
                            memory_bank_segment_ids= torch.cat(memory_bank_segment_ids, dim=0).to('cuda:0')


                            clean_ind_indices,clean_ind_labels,clean_ind_center,clean_ind_label,ood_center,ind_all_indice,ood_all_indice,gb_for_train_centers, gb_for_train_radius, gb_for_train_label,original_examples= self.clusterLoss.forward(args,accumulated_features,accumulated_labels,accumulated_noise_labels,accumulated_indexs,
                                                                                                               select=False)
                            clean_ind_center = torch.tensor(clean_ind_center)
                            clean_ind_label = torch.tensor(clean_ind_label)
                            ood_center= torch.tensor(ood_center)
                            intra_loss, inter_loss, ood_loss = self.compute_cluster_loss(clean_ind_label, ind_all_indice,ood_all_indice,accumulated_features,accumulated_indexs)



                            selected_indices = torch.tensor(clean_ind_indices, dtype=torch.long).to('cuda:0')
                            selected_positions = torch.nonzero(accumulated_indexs.unsqueeze(1) == selected_indices,
                                                                   as_tuple=False)[:, 0]

                            clean_ind_input_ids = memory_bank_input_ids[selected_positions].to('cuda:0')
                            clean_ind_input_mask = memory_bank_input_mask[selected_positions].to('cuda:0')
                            clean_ind_segment_ids = memory_bank_segment_ids[selected_positions].to('cuda:0')
                            clean_ind_label_ids = torch.tensor(clean_ind_labels).to('cuda:0')

                            #损失函数/来自干净样本 和ind噪声的样本  修改了噪声的标签。
                            feature_clean_ind = self.model(clean_ind_input_ids, clean_ind_segment_ids, clean_ind_input_mask, feature_ext="True")  # 这行代码调用模型进行前向传播，并计算损失

                            clean_ind_center=clean_ind_center.to(feature_clean_ind.device)
                            clean_ind_label=clean_ind_label.to(feature_clean_ind.device)
                            loss_clean_ind = self.model(clean_ind_input_ids, clean_ind_segment_ids, clean_ind_input_mask,
                                              clean_ind_label_ids,
                                              mode="train")  # 这行代码调用模型进行前向传播，并计算损失

                            #loss_clean_ind = self.compute_classification_loss(data,feature_clean_ind,  clean_ind_label_ids, clean_ind_center,  clean_ind_label)
                            total_loss=args.alpha*intra_loss+ args.beta*inter_loss+  args.gamma*ood_loss+  args.delta*loss_clean_ind
                            print(
                                f"[Epoch {epoch}] intra_loss: {intra_loss:.4f}, inter_loss: {inter_loss:.4f}, ood_loss: {ood_loss:.4f}, clean_ce_loss: {loss_clean_ind:.4f}, total_loss: {total_loss:.4f}")
                            self.optimizer2.zero_grad()
                            total_loss.backward()
                            self.optimizer2.step()
                            self.scheduler.step()
                            current_lr = self.optimizer.param_groups[0]['lr']
                            print(f"Learning rate after scheduler step: {current_lr:.8f}")

                            self.optimizer2.zero_grad()# 将优化器的梯度信息归零，以准备进行下一次反向传播
                            util.summary_writer.add_scalar("Loss/total_loss", total_loss.item(), step + epoch * batch_number)
                            nb_tr_examples += input_ids.size(0)

                            gb_for_train_centers = torch.tensor(gb_for_train_centers, dtype=torch.float32).to(self.device)
                            gb_for_train_radius= torch.tensor(gb_for_train_radius, dtype=torch.float32).to(self.device)
                            gb_for_train_label = torch.tensor([int(x) for x in gb_for_train_label], dtype=torch.long).to(self.device)
                            memory_bank = []
                            memory_bank_label = []
                            memory_bank_noise_label = []
                            memory_bank_index = []
                            memory_bank_input_ids = []
                            memory_bank_input_mask = []
                            memory_bank_segment_ids = []
                #self.draw2(args, data)
                print('train_loss', total_loss.item())
                # evaluation_score= self.evaluation(args, data,gb_for_train_centers, gb_for_train_radius, gb_for_train_label)  #这一行去掉代表没有根据聚类进行表征学习的
                evaluation_score = self.eval(args, data)
                print('evaluation_score', evaluation_score)
                if evaluation_score > self.best_evaluation_score:
                    best_model = copy.deepcopy(self.model)  # 如果当前评估分数更好，将当前模型保存为最佳模型
                    wait_ood_train = 0
                    self.best_evaluation_score = evaluation_score
                    #self.best_original_examples=original_examples
                else:
                    wait_ood_train += 1
                    if  wait_ood_train >= args.wait_patient:  # 如果 wait 达到了一个指定的阈值（args.wait_patient），则结束训练循环

                        break
            args.total_epoch_stop = epoch + 1
        self.model = best_model  # 将最佳模型设置为训练结束后的模型
        if args.save_model:  # 检查是否存在一个名为args.save_model的参数，如果指定了保存模型的选项，调用 self.save_model 方法来保存训练后的模型
            self.save_model(args)
        #self.draw(args, data)
        #self.draw2(args, data)
        #return self.best_original_examples
    def compute_classification_loss(self, data,features, labels, centroids, centroid_labels):
        if features.size(0) == 0:
            return torch.tensor(0.0).to(features.device)


        logits = torch.cdist(features, centroids, p=2)
        distances = torch.full((features.shape[0], len(data.known_label_list)), float('inf')).to(logits.device)


        for label in range(len(data.known_label_list)):
            class_mask = (centroid_labels == label)
            if class_mask.any():
                class_distances = logits[:, class_mask]
                distances[:, label] = class_distances.min(dim=1)[0]


        distances = F.normalize(distances, p=1, dim=1)

        probabilities = F.softmax(-distances, dim=1)

        true_probabilities = probabilities[torch.arange(probabilities.size(0)), labels]

        loss = -torch.log(true_probabilities).mean()
        return loss

    def calculate_granular_balls(self, args, data):  # 模型训练

        for epoch in trange(int(1), desc="Epoch"):  # trange创建进度条
            self.model.train()  # 将模型设置为训练模式。在训练模式下，模型会进行前向传播和反向传播，并根据损失函数更新模型参数
            memory_bank = []
            memory_bank_label = []
            memory_bank_noise_label = []
            memory_bank_index = []

            for step, batch in enumerate(data.train_dataloader):  # 在每次迭代中，enumerate() 返回两个值，一个是索引（step），另一个是可迭代对象中的元素（batch）
                batch = tuple(t.to(self.device) for t in
                              batch)  # 这行代码将数据批次中的所有张量（input_ids、input_mask、segment_ids、label_ids）都移动到模型所在的设备（通常是 GPU）上
                input_ids, input_mask, segment_ids, label_ids,label_noiseids,index = batch
                batch_number = len(data.train_dataloader)

                # 获取特征时不计算梯度
                features = self.model(input_ids, segment_ids, input_mask, feature_ext="True")
                memory_bank.append(features.cpu())  # 将特征移到CPU以节省GPU内存
                memory_bank_label.append(label_ids.cpu())
                memory_bank_noise_label.append(label_noiseids.cpu())
                memory_bank_index.append(index.cpu())

                if (step + 1) == batch_number:
                    accumulated_features = torch.cat(memory_bank, dim=0).to('cuda:0')
                    accumulated_labels = torch.cat(memory_bank_label, dim=0).to('cuda:0')
                    accumulated_noise_labels = torch.cat(memory_bank_noise_label, dim=0).to('cuda:0')
                    accumulated_indexs = torch.cat(memory_bank_index, dim=0).to('cuda:0')

                    gb_for_test_centers, gb_for_test_radius, gb_for_test_label= self.clusterLoss.forward(args, accumulated_features, accumulated_labels, accumulated_noise_labels,accumulated_indexs,select=True)


                    gb_for_test_centers = torch.tensor(gb_for_test_centers, dtype=torch.float32).to(self.device)
                    gb_for_test_radius = torch.tensor(gb_for_test_radius, dtype=torch.float32).to(self.device)
                    gb_for_test_label = torch.tensor([int(x) for x in gb_for_test_label], dtype=torch.long).to(
                        self.device)


        return gb_for_test_centers, gb_for_test_radius, gb_for_test_label

    def draw2(self, args, data):
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib.colors as mcolors

        self.model.train()

        memory_bank = []
        memory_bank_label = []
        memory_bank_noise_label = []

        for step, batch in enumerate(data.train_dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, label_noiseids, index = batch

            with torch.no_grad():
                features = self.model(input_ids, segment_ids, input_mask, feature_ext="True")
            memory_bank.append(features.cpu())
            memory_bank_label.append(label_ids.cpu())
            memory_bank_noise_label.append(label_noiseids.cpu())

        features = torch.cat(memory_bank, dim=0).numpy()
        true_labels = torch.cat(memory_bank_label, dim=0).numpy()
        noise_labels = torch.cat(memory_bank_noise_label, dim=0).numpy()

        # 找出 OOD 标签 ID
        ood_label_candidates = set(true_labels) - set(noise_labels)
        assert len(ood_label_candidates) == 1, f"Expected exactly one OOD label, got {ood_label_candidates}"
        ood_label_id = list(ood_label_candidates)[0]

        # TSNE 降维
        tsne = TSNE(n_components=2, random_state=0)
        X_2d = tsne.fit_transform(features)

        # 已知类标签索引
        known_label_ids = sorted(list(set(true_labels) - {ood_label_id}))

        # 自动生成颜色支持任意类数量
        def generate_distinct_colors(n):
            return [mcolors.hsv_to_rgb((i / n, 0.5, 0.9)) for i in range(n)]

        colors_list = generate_distinct_colors(len(known_label_ids))
        label_to_color = {label_id: colors_list[i] for i, label_id in enumerate(known_label_ids)}

        # 绘图
        plt.figure(figsize=(8, 6))
        for i in range(len(true_labels)):
            x, y = X_2d[i]
            t_label = true_labels[i]
            n_label = noise_labels[i]

            if t_label == ood_label_id:
                plt.scatter(x, y, c='gray', marker='x', s=20, alpha=1)
            elif t_label != n_label:
                color = label_to_color.get(t_label, 'black')
                plt.scatter(x, y, c=[color], marker='x', s=20, alpha=1)  # IND噪声
            else:
                color = label_to_color.get(t_label, 'black')
                plt.scatter(x, y, edgecolors=[color], facecolors='none', marker='o', s=5, alpha=0.8)  # Clean样本，空心圆圈

        # 不显示图例
        plt.title(f"TSNE Visualization: {args.dataset}, known_cls_ratio={args.known_cls_ratio}, ind_noise_ratio={args.ind_noise_ratio},ood_type={args.ood_type}")
        plt.xlabel("TSNE-1")
        plt.ylabel("TSNE-2")
        plt.tight_layout()
        plt.show()

    def draw(self, args, data):  # 模型训练

        for epoch in trange(int(1), desc="Epoch"):  # trange创建进度条
            self.model.train()  # 将模型设置为训练模式。在训练模式下，模型会进行前向传播和反向传播，并根据损失函数更新模型参数
            tr_loss = 0  # 用于跟踪当前时代的训练损失和训练样本的数量
            nb_tr_examples, nb_tr_steps = 0, 0
            memory_bank = []
            memory_bank_label = []
            memory_bank_noise_label = []

            for step, batch in enumerate(data.train_dataloader):  # 在每次迭代中，enumerate() 返回两个值，一个是索引（step），另一个是可迭代对象中的元素（batch）
                batch = tuple(t.to(self.device) for t in
                              batch)  # 这行代码将数据批次中的所有张量（input_ids、input_mask、segment_ids、label_ids）都移动到模型所在的设备（通常是 GPU）上
                input_ids, input_mask, segment_ids, label_ids,label_noiseids,index = batch
                batch_number = len(data.train_dataloader)

                # 获取特征时不计算梯度
                features = self.model(input_ids, segment_ids, input_mask, feature_ext="True")
                memory_bank.append(features.cpu())  # 将特征移到CPU以节省GPU内存
                memory_bank_label.append(label_ids.cpu())
                memory_bank_noise_label.append(label_noiseids.cpu())
                ##使用累积的特征进行聚类
                if (step + 1) == batch_number:
                    accumulated_features = torch.cat(memory_bank, dim=0).to('cuda:0')
                    accumulated_labels = torch.cat(memory_bank_label, dim=0).to('cuda:0')
                    accumulated_features2 = accumulated_features.cpu().detach().numpy()
                    accumulated_noise_labels = torch.cat(memory_bank_noise_label, dim=0).to('cuda:0')
                    accumulated_labels2 = accumulated_labels.cpu().numpy()

                    # 使用 TSNE 进行降维
                    tsne = TSNE(n_components=2, random_state=0)
                    X_2d = tsne.fit_transform(accumulated_features2)

                    # 创建一个颜色数组
                    colors = np.array([0.5] * len(accumulated_labels2))  # 默认所有颜色为灰色

                    # 更新颜色数组：对于噪声标签与真实标签不同的样本，将颜色设置为灰色
                    for i in range(len(accumulated_labels2)):
                        if accumulated_labels2[i] != accumulated_noise_labels[i]:
                            colors[i] = 0.5  # 灰色表示噪声标签与真实标签不一致
                        else:
                            colors[i] = accumulated_labels2[i]  # 使用标签的颜色

                    # 可视化结果
                    plt.figure(figsize=(8, 6))
                    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, cmap='plasma', marker='.',
                                          s=10)  # 调整点的大小为10

                    # 自定义颜色条范围
                    norm = mcolors.Normalize(vmin=min(accumulated_labels2),
                                             vmax=max(accumulated_labels2))  # 根据数据的标签范围设置颜色条
                    cbar = plt.colorbar(scatter, ticks=[min(accumulated_labels2), max(accumulated_labels2)])  # 设置颜色条的范围
                    cbar.set_label('Labels')  # 设置颜色条标签
                    cbar.set_ticks([min(accumulated_labels2), max(accumulated_labels2)])  # 设置颜色条显示的刻度

                    plt.show()
    def get_optimizer(self, args):

        param_optimizer = list(self.model.named_parameters())  # 获取参数的名称和值的元组列表
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']  # 不对偏置（bias）、LayerNormalization 层的偏置和权重应用权重衰减

        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            # 需要应用权重衰减
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            # 不需要应用权重衰减
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,  # 通过 BertAdam 类创建了一个优化器对象 optimizer
                             lr=args.lr,
                             warmup=args.warmup_proportion,  # 预热比例
                             t_total=self.num_train_optimization_steps)  # 总训练步数
        return optimizer



    def get_optimizer2(self, args):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        # 使用 AdamW 优化器
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr2)

        # 使用 linear 学习率调度器
        self.scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=int(args.warmup_proportion * self.num_train_optimization_steps),
            num_training_steps=self.num_train_optimization_steps
        )
        return optimizer

    def save_model(self, args):

        if not os.path.exists(args.pretrain_dir):  # 如果目录 args.pretrain_dir 不存在，就执行后续的代码
            os.makedirs(args.pretrain_dir)  # 调用 os.makedirs(args.pretrain_dir)，代码会检查指定路径的目录是否存在。如果不存在，它将创建该目录以及必要的父级目录
        self.save_model = self.model.module if hasattr(self.model,
                                                       'module') else self.model  # hasattr(self.model, 'module')  用于检查对象 self.model 是否有一个名为 'module' 的属性。

        model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME)
        model_config_file = os.path.join(args.pretrain_dir, CONFIG_NAME)
        torch.save(self.save_model.state_dict(), model_file)  # 将 self.save_model 的状态字典（即模型的权重参数）保存到 model_file
        with open(model_config_file, "w") as f:  # 使用 open 函数以写入模式打开 model_config_file 文件
            f.write(self.save_model.config.to_json_string())  # 使用 f.write 方法将模型的配置信息以 JSON 字符串的形式写入该文件


def calculate_distances(a, b):
    distances = torch.sqrt(torch.sum((a[:, None, :] - b[None, :, :]) ** 2, dim=2))
    return distances

