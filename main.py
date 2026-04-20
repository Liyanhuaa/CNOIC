from init_parameter import *
from dataloader import *
from pretrain import *
from torch.utils.tensorboard import SummaryWriter
# from tensorflow.keras.callbacks import TensorBoard
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from utils import util
import time
from gb_test import ModelManager
from train_boundary import Manager_boundary

if __name__ == '__main__':
    start_time = time.time()


    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    random.seed(100)
    np.random.seed(100)
    torch.manual_seed(100)
    print('Parameters Initialization...')
    parser = init_model()
    args = parser.parse_args()
    data = Data(args)


    # ind_noise_list = (0.1,0.15,0.2,0.25,0.3,0.35)
    # ood_noise_list = (0.1,0.15,0.2,0.25,0.3,0.35)

    # for args.ind_noise_ratio in ind_noise_list:
    #     for args.ood_noise_ratio in ood_noise_list:

    if args.purity_select_ball>0:
            print('Training begin...')
            manager_p1 = PretrainModelManager(args, data)
            manager_p1.train(args, data)
            print('Training finished!')

    print('Calculate ball begin...')
    gb_centroids, gb_radii, gb_labels=manager_p1.calculate_granular_balls(args, data)
    print('Calculate ball  finished!')


    manager = ModelManager(args, data, manager_p1.model)
    print('Evaluation begin...')
    manager.evaluation(args, data, gb_centroids, gb_radii, gb_labels,mode="test")
    print('Evaluation finished!')

    end_time = time.time()
    total_time_min = (end_time - start_time) / 60

    if torch.cuda.is_available():
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
    else:
        peak_memory_gb = 0.0
    print(f"Total Running Time: {total_time_min:.2f} min")
    print(f"Peak GPU Memory Usage: {peak_memory_gb:.2f} GB")

    args.total_time_min = round(total_time_min, 2)
    args.peak_memory_gb = round(peak_memory_gb, 2)



    manager.save_results(args)
    #
    #         min_ball_list = (9,10,12,14,16,17)
    #         purity_list = (0.4,0.5,0.6, 0.7, 0.8)
    # #
    # #
    # #
    #         for min_ball in min_ball_list:
    #             for purity in purity_list:
    #                 print(f"\n=== Start evaluation with purity={purity}, min_ball={min_ball} ===")
    #
    #                 args.purity_select_ball = purity
    #                 args.min_ball_select_ball = min_ball
    #
    #                 # 开始计时
    #                 start_time = time.time()
    #
    #                 print('Calculate ball begin...')
    #                 gb_centroids, gb_radii, gb_labels = manager_p1.calculate_granular_balls(args, data)
    #                 if len(gb_centroids) == 0:
    #                     continue
    #                 else:
    #                     print('Calculate ball finished!')
    #
    #                     # 重新初始化 ModelManager（如果需要新的实例，否则可以复用）
    #                     manager = ModelManager(args, data, manager_p1.model)
    #
    #                     print('Evaluation begin...')
    #                     manager.evaluation(args, data, gb_centroids, gb_radii, gb_labels, mode="test")
    #                     print('Evaluation finished!')
    #
    #                     end_time = time.time()
    #                     elapsed_time = end_time - start_time
    #                     print(f"Elapsed time for purity={purity}, min_ball={min_ball}: {elapsed_time:.2f} seconds")
    #
    #                     # 显式释放无用变量，防止显存累积
    #                     del gb_centroids, gb_radii, gb_labels
    #                     import gc
    #
    #                     gc.collect()
    #                     import torch
    #
    #                     torch.cuda.empty_cache()