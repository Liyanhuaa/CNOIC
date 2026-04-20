import argparse


def init_model():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default='data', type=str,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")

    parser.add_argument("--save_results_path", type=str, default='outputs', help="the path to save results")

    parser.add_argument("--pretrain_dir", default='models', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")


    parser.add_argument("--bert_model",
                        default=r"E:\code_hh\GBNOISE\gbnoise_1_ijcai22\uncased_L-12_H-768_A-12",
                        type=str, help="The path for the pre-trained bert model.")
    # parser.add_argument("--bert_model",
    #                     default="/hy-tmp/gbnoise_tkde/uncased_L-12_H-768_A-12",
    #                     type=str, help="The path for the pre-trained bert model.")

    parser.add_argument("--max_seq_length", default=None, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--feat_dim", default=768, type=int, help="The feature dimension.")

    parser.add_argument("--warmup_proportion", default=0.1, type=float)

    parser.add_argument("--freeze_bert_parameters", action="store_true", help="Freeze the last parameters of BERT")

    parser.add_argument("--save_model", action="store_true", help="save trained-model")

    parser.add_argument("--save_results", action="store_true", help="save test results")

    parser.add_argument("--dataset", default=None, type=str, required=True,
                        help="The name of the dataset to train selected")
    parser.add_argument("--open_noise_dataset", default=None, type=str, required=True,
                        help="The name of the dataset to train selected")



    parser.add_argument("--known_cls_ratio", default=0.85, type=float, required=True,
                        help="The number of known classes")
    parser.add_argument("--ind_noise_ratio", default=0.25, type=float, required=True,
                        help="The ratio of ind noise")
    parser.add_argument("--ood_noise_ratio", default=0.25, type=float, required=True,
                        help="The ratio of ood noise")

    parser.add_argument("--labeled_ratio", default=1.0, type=float,
                        help="The ratio of labeled samples in the training set")

    parser.add_argument("--method", type=str, default=None, help="which method to use")

    parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")

    parser.add_argument("--gpu_id", type=str, default='0', help="Select the GPU id")

    parser.add_argument("--lr", default=2e-5, type=float,
                        help="The learning rate of BERT.")
    parser.add_argument("--lr_boundary", type=float, default=0.05, help="The learning rate of the decision boundary.")

    parser.add_argument("--num_train_epochs", default=100, type=float,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--train_batch_size", default=128, type=int,
                        help="Batch size for training.")

    parser.add_argument("--eval_batch_size", default=128, type=int,
                        help="Batch size for evaluation.")

    parser.add_argument("--wait_patient", default=10, type=int,
                        help="Patient steps for Early Stop.")

    parser.add_argument("--lr2", type=float, default=2e-5, help="The learning rate of the decision boundary.")

    parser.add_argument("--num_subcentroids", default=4, type=int,
                        help="number of subcentroids.") #可能是NN 不需要的
    parser.add_argument("--step", default=5, type=int,
                        help="step of cluster.") #可能是NN 不需要的
    parser.add_argument("--save_index_path", type=str, default='save_indice', help="the path to save results")
    parser.add_argument("--max_ball", default=100000, type=int,
                        help="min_ball of granular ball select ball..")
    parser.add_argument("--output_dir", type=str, default=r"E:\code_hh\GBNOISE\gbnoise_nn2\outputs", help="Directory to save index files.")
   #  parser.add_argument("--output_dir", type=str, default="/hy-tmp/gbnoise_tkde/outputs",
   #                      help="Directory to save index files.")

    parser.add_argument('--alpha', default=1.0, type=float,
                        help="Weight for intra-class (intra) loss: encourages samples of the same class to be close.")

    parser.add_argument('--beta', default=1.0, type=float,
                        help="Weight for inter-class (inter) loss: pushes different classes further apart.")

    parser.add_argument('--gamma', default=1.0, type=float,
                        help="Weight for OOD-center separation (ood) loss: keeps OOD class centers away from IND.")

    parser.add_argument('--delta', default=1.0, type=float,
                        help="Weight for clean IND sample classification loss.")

    parser.add_argument("--purity_train", default=0.9, type=int,
                        help="the purity of granular ball in train.")
    parser.add_argument("--p_noise_ind", default=0.8, type=int,
                        help="granular ball bigger than it is ind ball.")

#从这里之后的参数需要修改

    parser.add_argument("--purity_get_ball", default=0.7, type=float,
                        help="the purity of granular ball in get ball.")
    parser.add_argument("--purity_select_ball", default=0.6, type=float,
                        help="the purity of granular ball in select ball.")



    parser.add_argument("--min_ball_train", default=10, type=int,
                        help="min_ball of granular ball in train.")
    parser.add_argument("--min_ball_get_ball", default=10, type=int,
                        help="min_ball of granular ball in get ball..")
    parser.add_argument("--min_ball_select_ball", default=12, type=int,
                        help="min_ball of granular ball select ball..")


    parser.add_argument("--n_noise", default=5, type=int,
                        help=" granular ball smaller than it is ood ball")#只有一个数据集不是5



    parser.add_argument("--p_noise_ood", default=0.2, type=float,
                        help="granular ball smaller than it is ood ball.")  #不同数据集不一样


    parser.add_argument("--warm_train_epoch", default=10, type=int,
                        help=" the number of warm train epoch")

    parser.add_argument("--ood_type", default='near', type=str,
                        help=" the number of warm train epoch")

    parser.add_argument("--total_epoch_stop", default=0, type=int,
                        help="epoch number when training stops")

    parser.add_argument("--total_time_min", default=0.0, type=float,
                        help="total running time (minutes)")

    parser.add_argument("--peak_memory_gb", default=0.0, type=float,
                        help="peak GPU memory usage (GB)")





    return parser
