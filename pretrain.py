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

        self.num_train_optimization_steps = len(data.train_dataloader) * args.num_train_epochs

        self.optimizer = self.get_optimizer(args)
        self.optimizer2 = self.get_optimizer2(args)

        self.best_eval_score = 0
        self.best_evaluation_score = 0
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

        logits = self.euclidean_metric(features, gb_centroids)
        _, preds = logits.min(dim=1)
        print("Features shape:", features.shape)
        print("Indexed centroids shape:", gb_centroids[preds].shape)


        final_preds = torch.tensor([data.unseen_token_id] * features.shape[0], device=features.device)
        for i in range(features.shape[0]):
            final_preds[i] = gb_labels[preds[i]]
        return final_preds


    def eval(self, args, data):

        self.model.eval()
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, data.num_labels)).to(self.device)

        for batch in tqdm(data.eval_dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, label_noiseids, index = batch
            with torch.set_grad_enabled(False):
                _, logits = self.model(input_ids, segment_ids, input_mask,
                                       mode='eval')  #
                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, logits))

        total_probs, total_preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        return acc


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
            with torch.set_grad_enabled(False):
                pooled_output, _ = self.model(input_ids, segment_ids,
                                              input_mask)
                preds = self.open_classify(data,pooled_output,gb_centroids, gb_radii, gb_labels)

                total_labels = torch.cat((total_labels, label_ids))
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
            'cuda:0')
        ood_centers = torch.stack(ood_centers) if ood_centers else torch.empty(0).to(
            'cuda:0')
        if ind_centers.size(0) > 0:
            pairwise_dist_clean = torch.cdist(ind_centers, ind_centers,p=2)


        if ood_centers.size(0) and ind_centers.size(0) > 0:
            pairwise_dist_ood = torch.cdist(ind_centers, ood_centers, p=2)

        inter_loss = 0.0
        intra_loss = 0.0
        ood_loss = 0.0

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
        wait = 0
        wait_ood_train=0
        best_model = None
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            if epoch < args.warm_train_epoch:
                for step, batch in enumerate(data.train_dataloader):
                    batch = tuple(t.to(self.device) for t in
                                  batch)
                    input_ids, input_mask, segment_ids, label_ids,label_noiseids,index= batch
                    batch_number = len(data.train_dataloader)
                    with ((torch.set_grad_enabled(True))):

                            loss1 = self.model(input_ids, segment_ids, input_mask, label_noiseids,mode="train")

                            self.optimizer.zero_grad()
                            loss1.backward()
                            self.optimizer.step()
                            tr_loss += loss1.item()
                            util.summary_writer.add_scalar("Loss/loss1", loss1.item(), step + epoch * batch_number)
                            nb_tr_examples += input_ids.size(0)
                            nb_tr_steps += 1

                loss = tr_loss / nb_tr_steps
                print('train_loss', loss)

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
                            clean_ind_label = torch.tensor(clean_ind_label)
                            intra_loss, inter_loss, ood_loss = self.compute_cluster_loss(clean_ind_label, ind_all_indice,ood_all_indice,accumulated_features,accumulated_indexs)



                            selected_indices = torch.tensor(clean_ind_indices, dtype=torch.long).to('cuda:0')
                            selected_positions = torch.nonzero(accumulated_indexs.unsqueeze(1) == selected_indices,
                                                                   as_tuple=False)[:, 0]

                            clean_ind_input_ids = memory_bank_input_ids[selected_positions].to('cuda:0')
                            clean_ind_input_mask = memory_bank_input_mask[selected_positions].to('cuda:0')
                            clean_ind_segment_ids = memory_bank_segment_ids[selected_positions].to('cuda:0')
                            clean_ind_label_ids = torch.tensor(clean_ind_labels).to('cuda:0')





                            loss_clean_ind = self.model(clean_ind_input_ids, clean_ind_segment_ids, clean_ind_input_mask,
                                              clean_ind_label_ids,
                                              mode="train")


                            total_loss=args.alpha*intra_loss+ args.beta*inter_loss+  args.gamma*ood_loss+  args.delta*loss_clean_ind
                            print(
                                f"[Epoch {epoch}] intra_loss: {intra_loss:.4f}, inter_loss: {inter_loss:.4f}, ood_loss: {ood_loss:.4f}, clean_ce_loss: {loss_clean_ind:.4f}, total_loss: {total_loss:.4f}")
                            self.optimizer2.zero_grad()
                            total_loss.backward()
                            self.optimizer2.step()
                            self.scheduler.step()
                            current_lr = self.optimizer.param_groups[0]['lr']
                            print(f"Learning rate after scheduler step: {current_lr:.8f}")

                            self.optimizer2.zero_grad()
                            util.summary_writer.add_scalar("Loss/total_loss", total_loss.item(), step + epoch * batch_number)
                            nb_tr_examples += input_ids.size(0)


                            memory_bank = []
                            memory_bank_label = []
                            memory_bank_noise_label = []
                            memory_bank_index = []
                            memory_bank_input_ids = []
                            memory_bank_input_mask = []
                            memory_bank_segment_ids = []

                print('train_loss', total_loss.item())

                evaluation_score = self.eval(args, data)
                print('evaluation_score', evaluation_score)
                if evaluation_score > self.best_evaluation_score:
                    best_model = copy.deepcopy(self.model)
                    wait_ood_train = 0
                    self.best_evaluation_score = evaluation_score

                else:
                    wait_ood_train += 1
                    if  wait_ood_train >= args.wait_patient:

                        break
            args.total_epoch_stop = epoch + 1
        self.model = best_model
        if args.save_model:
            self.save_model(args)

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

    def calculate_granular_balls(self, args, data):

        for epoch in trange(int(1), desc="Epoch"):
            self.model.train()
            memory_bank = []
            memory_bank_label = []
            memory_bank_noise_label = []
            memory_bank_index = []

            for step, batch in enumerate(data.train_dataloader):
                batch = tuple(t.to(self.device) for t in
                              batch)
                input_ids, input_mask, segment_ids, label_ids,label_noiseids,index = batch
                batch_number = len(data.train_dataloader)


                features = self.model(input_ids, segment_ids, input_mask, feature_ext="True")
                memory_bank.append(features.cpu())
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


    def get_optimizer(self, args):

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            # 需要应用权重衰减
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            # 不需要应用权重衰减
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.lr,
                             warmup=args.warmup_proportion,
                             t_total=self.num_train_optimization_steps)
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

        if not os.path.exists(args.pretrain_dir):
            os.makedirs(args.pretrain_dir)
        self.save_model = self.model.module if hasattr(self.model,
                                                       'module') else self.model

        model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME)
        model_config_file = os.path.join(args.pretrain_dir, CONFIG_NAME)
        torch.save(self.save_model.state_dict(), model_file)
        with open(model_config_file, "w") as f:
            f.write(self.save_model.config.to_json_string())

