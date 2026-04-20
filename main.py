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
