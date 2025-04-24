import os, sys

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
    

    