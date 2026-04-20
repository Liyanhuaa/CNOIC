from init_parameter import *
from dataloader import *
from pretrain import *


import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
def set_seed(seed):
    random.seed(seed)  # Set Python random seed
    np.random.seed(seed)  # Set numpy random seed
    torch.manual_seed(seed)  # Set PyTorch seed for CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # Set PyTorch seed for all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable benchmark

class ModelManager:

    def __init__(self, args, data,pretrained_model=None):

        self.model = pretrained_model
        if self.model is None:
            self.model = BertForModel.from_pretrained(args.bert_model, cache_dir = "", num_labels = data.num_labels)
            self.restore_model(args)

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id     
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.best_eval_score = 0
        self.delta = None
        self.delta_points = []
        self.centroids = None

        self.test_results = None
        self.predictions = None
        self.true_labels = None
        self.bset_gb_for_select_for_test_centers=None
        self.bset_gb_for_select_for_test_radius =None
        self.bset_gb_for_select_for_test_label=None



    def euclidean_metric(self,a, b):
        n = a.shape[0]
        m = b.shape[0]
        a = a.unsqueeze(1).expand(n, m, -1)
        b = b.unsqueeze(0).expand(n, m, -1)
        return torch.sqrt(torch.pow(a - b, 2).sum(2))

    def open_classify(self, data, features, gb_centroids, gb_radii, gb_labels):
        # 计算输入特征与所有质心之间的欧氏距离
        gb_centroids=gb_centroids.to(self.device)
        gb_radii=gb_radii.to(self.device)
        gb_labels=gb_labels.to(self.device)
        features=features.to(self.device)
        logits = self.euclidean_metric(features, gb_centroids)
        _, preds = logits.min(dim=1)  # 获取最小距离的索引，对应最近的质心
        # print("Features shape:", features.shape)
        # print("Indexed centroids shape:", gb_centroids[preds].shape)

        # 计算每个输入特征与其预测类别质心之间的实际欧氏距离
        euc_dis = torch.norm(features - gb_centroids[preds], dim=1)

        # 初始化分类结果为未知类别
        final_preds = torch.tensor([data.unseen_token_id] * features.shape[0], device=features.device)

        # 对每个样本进行分类
        for i in range(features.shape[0]):
            if euc_dis[i] < gb_radii[preds[i]]:
                final_preds[i] = gb_labels[preds[i]]
            else:
                final_preds[i] = data.unseen_token_id  # 如果距离大于半径，标记为未知类别

        return final_preds

    def eval_select(self, args, data, best_original_examples):
        self.model.eval()
        dataloader = data.eval_dataloader

        for purity_limit in (0.5, 0.6, 0.7, 0.8, 0.9):
            for n_limit in (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20):
                print("purity_limit:", purity_limit)
                print("n_limit:", n_limit)

                gb_for_select_for_test_centers = []
                gb_for_select_for_test_label = []
                gb_for_select_for_test_radius = []

                for example in best_original_examples:
                    if example.purity > purity_limit and example.numbers > n_limit:
                        gb_for_select_for_test_centers.append(example.centers)
                        gb_for_select_for_test_label.append(example.label)
                        gb_for_select_for_test_radius.append(example.radius)

                if len(gb_for_select_for_test_centers) == 0:
                    continue

                gb_for_select_for_test_centers = torch.tensor(
                    gb_for_select_for_test_centers, dtype=torch.float32, device=self.device
                )
                gb_for_select_for_test_radius = torch.tensor(
                    gb_for_select_for_test_radius, dtype=torch.float32, device=self.device
                )
                gb_for_select_for_test_label = torch.tensor(
                    [int(x) for x in gb_for_select_for_test_label], dtype=torch.long, device=self.device
                )

                total_labels = []
                total_preds = []

                with torch.inference_mode():
                    for batch in tqdm(dataloader, desc="Iteration"):
                        batch = tuple(t.to(self.device) for t in batch)
                        input_ids, input_mask, segment_ids, orignal_label, label_ids, index = batch

                        pooled_output, _ = self.model(input_ids, segment_ids, input_mask)
                        preds = self.open_classify(
                            data,
                            pooled_output,
                            gb_for_select_for_test_centers,
                            gb_for_select_for_test_radius,
                            gb_for_select_for_test_label
                        )

                        total_labels.append(label_ids.cpu())
                        total_preds.append(preds.cpu())

                total_labels = torch.cat(total_labels, dim=0)
                total_preds = torch.cat(total_preds, dim=0)

                y_pred = total_preds.numpy()
                y_true = total_labels.numpy()

                self.predictions = [data.label_list[idx] for idx in y_pred]
                self.true_labels = [data.label_list[idx] for idx in y_true]

                cm = confusion_matrix(y_true, y_pred)
                eval_score = F_measure(cm)['F1-score']

                if eval_score > self.best_eval_score:
                    self.best_eval_score = eval_score
                    self.bset_gb_for_select_for_test_centers = gb_for_select_for_test_centers
                    self.bset_gb_for_select_for_test_radius = gb_for_select_for_test_radius
                    self.bset_gb_for_select_for_test_label = gb_for_select_for_test_label
                    print("best_purity_limit:", purity_limit)
                    print("best_n_limit:", n_limit)

        return (
            self.bset_gb_for_select_for_test_centers,
            self.bset_gb_for_select_for_test_radius,
            self.bset_gb_for_select_for_test_label
        )

    def evaluation(self, args, data, gb_centroids, gb_radii, gb_labels, mode="eval"):
        self.model.eval()

        if mode == 'eval':
            dataloader = data.eval_dataloader
        elif mode == 'test':
            dataloader = data.test_dataloader
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # 只转一次设备
        gb_centroids = gb_centroids.to(self.device)
        gb_radii = gb_radii.to(self.device)
        gb_labels = gb_labels.to(self.device)

        total_labels = []
        total_preds = []

        with torch.inference_mode():
            for batch in tqdm(dataloader, desc="Iteration"):
                batch = tuple(t.to(self.device, non_blocking=True) for t in batch)
                input_ids, input_mask, segment_ids, orignal_label, label_ids, index = batch

                pooled_output, _ = self.model(input_ids, segment_ids, input_mask)
                preds = self.open_classify(data, pooled_output, gb_centroids, gb_radii, gb_labels)

                total_labels.append(label_ids.cpu())
                total_preds.append(preds.cpu())

                del input_ids, input_mask, segment_ids, orignal_label, label_ids, index
                del pooled_output, preds

        total_labels = torch.cat(total_labels, dim=0)
        total_preds = torch.cat(total_preds, dim=0)

        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()

        self.predictions = [data.label_list[idx] for idx in y_pred]
        self.true_labels = [data.label_list[idx] for idx in y_true]

        cm = confusion_matrix(y_true, y_pred)

        if mode == 'eval':
            eval_score = F_measure(cm)['F1-score']
            return eval_score

        elif mode == 'test':
            results = F_measure(cm)
            acc = round(accuracy_score(y_true, y_pred) * 100, 2)
            results['Accuracy'] = acc
            self.test_results = results
            print('Accuracy:', acc)

    def class_count(self, labels):
        class_data_num = []
        for l in np.unique(labels):
            num = len(labels[labels == l])
            class_data_num.append(num)
        return class_data_num



    def restore_model(self, args):
        output_model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME)
        self.model.load_state_dict(torch.load(output_model_file))

    

    def save_results(self, args):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        var = [args.dataset, args.open_noise_dataset,args.known_cls_ratio, args.ind_noise_ratio,args.ood_noise_ratio
               ,args.seed,args.purity_train,args.purity_get_ball,args.purity_select_ball,
               args.min_ball_train,args.min_ball_get_ball,args.min_ball_select_ball,
               args.p_noise_ind,args.p_noise_ood,args.n_noise,args.warm_train_epoch,args.ood_type,args.train_batch_size,args.eval_batch_size,args.num_train_epochs,args.wait_patient,args.lr,args.lr2,args.alpha,args.beta,args.gamma,args.delta,args.total_epoch_stop,args.total_time_min,args.peak_memory_gb]
        names = ['dataset', 'open_noise_dataset','known_cls_ratio',
                 'ind_noise_ratio','ood_noise_ratio','seed',
                 'purity_train','purity_get_ball','purity_select_ball',
                 'min_ball_train','min_ball_get_ball','min_ball_select_ball',
                 'p_noise_ind','p_noise_ood','n_noise','warm_train_epoch','ood_type','train_batch_size','eval_batch_size','num_train_epochs','wait_patient','lr','lr2','alpha','beta','gamma','delta','total_epoch_stop','total_time_min','peak_memory_gb']
        vars_dict = {k:v for k,v in zip(names, var) }
        results = dict(self.test_results,**vars_dict)
        keys = list(results.keys())
        values = list(results.values())
        file_name = 'results.csv'
        results_path = os.path.join(args.save_results_path, file_name)
        if not os.path.exists(results_path):
            ori = []
            ori.append(values)
            df1 = pd.DataFrame(ori,columns = keys)
            df1.to_csv(results_path,index=False)
        else:
            df1 = pd.read_csv(results_path)
            new = pd.DataFrame(results,index=[1])
            df1 = pd.concat([df1, new], ignore_index=True)
            df1.to_csv(results_path,index=False)
        data_diagram = pd.read_csv(results_path)
        
        print('test_results', data_diagram)



    
    
