import os, sys
import matplotlib.pyplot as plt

# 1) locate this script’s folder
file_path = os.path.abspath(__file__)
script_dir = os.path.dirname(file_path)
# 2) go up two levels to the project root
proj_root = os.path.dirname(os.path.dirname(script_dir))
# 3) insert it into sys.path
sys.path.insert(0, proj_root)


import numpy as np
import torch

from local_code.stage_3_code.Dataset_Loader     import Dataset_Loader
from local_code.stage_3_code.Method_MNIST_CNN   import Method_MNIST_CNN
from local_code.stage_3_code.Method_CIFAR10_CNN import Method_CIFAR10_CNN
from local_code.stage_3_code.Method_ORL_CNN     import Method_ORL_CNN
from local_code.stage_3_code.Result_Saver       import Result_Saver
from local_code.stage_3_code.Setting            import Setting
from local_code.stage_3_code.Evaluate_Accuracy  import Evaluate_Accuracy

if __name__ == '__main__':
    # reproducibility
    np.random.seed(1)
    torch.manual_seed(1)

    # define the experiments: (dataset_name, MethodClass, epochs)
    experiments = [
        #('MNIST', Method_MNIST_CNN,   30,  1e-3),
        #('CIFAR', Method_CIFAR10_CNN,  30, 1e-3),
        ('ORL',   Method_ORL_CNN,     30, 1e-3),
    ]

    for name, MethodClass, epochs, ler in experiments:
        print(f'\n=== Running {name} CNN ===')

        # Dataset loader
        data_obj = Dataset_Loader(name, f'{name} dataset')

        # Model
        method_obj = MethodClass(f'{name}_CNN', f'{name} CNN', max_epoch=epochs, learning_rate=ler)

        # Result saver
        result_obj = Result_Saver('saver', '')
        result_obj.result_destination_folder_path = proj_root + '/result/stage_3_result/'
        result_obj.result_destination_file_name = f'{name.lower()}_preds.pkl'

        # Evaluator (used for print_setup_summary)
        evaluate_obj = Evaluate_Accuracy('accuracy', '')

        # Setting orchestrator
        setting_obj = Setting(f'{name}_exp', f'{name} CNN experiment')
        setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)

        # Run
        setting_obj.print_setup_summary()
        metrics = setting_obj.load_run_save_evaluate()

        # Print metrics
        print(f'--- {name} Metrics ---')
        for k, v in metrics.items():
            print(f'{k:15s}: {v:.4f}')