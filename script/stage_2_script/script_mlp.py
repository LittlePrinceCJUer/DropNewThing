import os, sys
import matplotlib.pyplot as plt

# 1) locate this script’s folder
file_path = os.path.abspath(__file__)
script_dir = os.path.dirname(file_path)
# 2) go up two levels to the project root
proj_root = os.path.dirname(os.path.dirname(script_dir))
# 3) insert it into sys.path
sys.path.insert(0, proj_root)

from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting import Setting
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('digits', 'MNIST-style digit data')

    method_obj = Method_MLP('multi-layer perceptron', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = proj_root + '/result/stage_2_result/'
    result_obj.result_destination_file_name = 'MLP_prediction_result'

    setting_obj = Setting('MLP_exp', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, None)
    setting_obj.print_setup_summary()
    metrics = setting_obj.load_run_save_evaluate()
    print('************ METRICS ************')
    for name, val in metrics.items():
        print(f'{name:20s}: {val:.4f}')
    print('************ Finish ************')
    # ------------------------------------------------------
    

    # # ---- compare training-loss for different learning rates ----
    # lrs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    # plt.figure(figsize=(10, 8))

    # for lr in lrs:
    #     # instantiate a fresh model for each lr
    #     method_i = Method_MLP('MLP', '2-layer MLP', lr)
    #     setting_i = Setting('MLP_lr_compare', '')
    #     setting_i.prepare(data_obj, method_i, result_obj, None)

    #     # this will load data, train the model, save predictions, and plot losses internally
    #     _ = setting_i.load_run_save_evaluate()

    #     # extract and plot training loss curve
    #     epochs = list(range(len(method_i.train_losses)))
    #     plt.plot(epochs,
    #              method_i.train_losses,
    #              label=f'lr={lr:.0e}')

    # # finalize the comparison plot
    # plt.xlabel('Epoch')
    # plt.ylabel('Training Loss')
    # plt.title('Training Loss vs. Epoch for Different Learning Rates')
    # plt.legend()
    # plt.grid(False)
    # plt.show()