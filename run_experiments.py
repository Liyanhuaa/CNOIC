import os

PYTHON_PATH = "python"


dataset = ['stackoverflow','banking', 'snips']


ind_list=[0.1,0.2]
type_list = ['near', 'far']


seed_list = [0,1,2,3,4]
known_list = [0.25, 0.5]


for data in dataset:
    for known_cls_ratio in known_list:
        for ind in ind_list:
            for type in type_list:

                    for seed in seed_list:
                        if data == 'stackoverflow':
                            open_noise_dataset = 'snips'
                            # granular ball 参数
                            min_ball_train = 10
                            min_ball_get_ball = 10
                            min_ball_select_ball = 9

                            # noise 参数
                            p_noise_ood = 0.5
                            n_noise = 8
                            # 训练参数
                            warm_train_epoch = 10

                            purity_get_ball = 0.7
                            purity_select_ball = 0.6
                        if data == 'banking':
                            open_noise_dataset = 'stackoverflow'
                            min_ball_train = 8
                            min_ball_get_ball = 8
                            min_ball_select_ball = 4

                            # noise 参数
                            p_noise_ood = 0.4
                            n_noise = 5
                            # 训练参数
                            warm_train_epoch = 20

                            purity_get_ball = 0.8
                            purity_select_ball = 0.6
                        if data == 'snips':
                            open_noise_dataset = 'stackoverflow'
                            min_ball_train = 20
                            min_ball_get_ball = 20
                            min_ball_select_ball = 80


                            p_noise_ood = 0.3
                            n_noise = 10

                            warm_train_epoch = 10

                            purity_get_ball = 0.8
                            purity_select_ball = 0.6


                        cmd = (
                            f"{PYTHON_PATH} main.py "
                            f"--dataset {data} "
                            f"--open_noise_dataset {open_noise_dataset} "
                            f"--known_cls_ratio {known_cls_ratio} "
                            f"--ind_noise_ratio {ind} "
                            f"--ood_noise_ratio {ind} "
                            f"--seed {seed} "
                            f"--freeze_bert_parameters "
                            f"--gpu_id 0 "
                            f"--ood_type {type} "
    

                            f"--min_ball_train {min_ball_train} "
                            f"--min_ball_get_ball {min_ball_get_ball} "
                            f"--min_ball_select_ball {min_ball_select_ball} "
    
                            f"--purity_get_ball {purity_get_ball} "
                            f"--purity_select_ball {purity_select_ball} "

                            f"--n_noise {n_noise} "
                            f"--p_noise_ood {p_noise_ood} "
    
   
                            f"--warm_train_epoch {warm_train_epoch} "
                        )

                        print(f"Running: IND={ind}, OOD={ind}, seed={seed}")
                        os.system(cmd)