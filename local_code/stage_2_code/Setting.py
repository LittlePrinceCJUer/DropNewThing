import os, sys

# 1) locate this script’s folder
file_path = os.path.abspath(__file__)
script_dir = os.path.dirname(file_path)
# 2) go up two levels to the project root
proj_root = os.path.dirname(os.path.dirname(script_dir))

from local_code.base_class.setting import setting
import torch
import matplotlib.pyplot as plt
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_2_code.Evaluate_Precision import Evaluate_Precision
from local_code.stage_2_code.Evaluate_Recall import Evaluate_Recall
from local_code.stage_2_code.Evaluate_F1Score import Evaluate_F1Score

class Setting(setting):
    def load_run_save_evaluate(self):
        # Windows:
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Mac:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # -- load train split --
        print('Loading train split...')
        self.dataset.dataset_source_folder_path = proj_root + '/data/stage_2_data/'
        self.dataset.dataset_source_file_name = 'train.csv'
        train = self.dataset.load()

        # -- load test split --
        print('Loading test split...')
        self.dataset.dataset_source_file_name = 'test.csv'
        test = self.dataset.load()

        # move all data to device
        for split in (train, test):
            split['X'] = split['X'].to(device)
            split['y'] = split['y'].to(device)

        # attach to method & move model to device
        self.method.data = {'train': train, 'test': test}
        self.method = self.method.to(device)

        # run training & prediction
        result = self.method.run()

        # save raw predictions
        self.result.data = result
        self.result.save()

        # compute metrics
        metrics = {}
        # accuracy
        acc = Evaluate_Accuracy('acc', '')
        acc.data = result
        metrics['accuracy'] = acc.evaluate()
        # precision macro & micro
        for avg in ('macro','micro'):
            p = Evaluate_Precision(f'prec_{avg}', '', average=avg)
            p.data = result
            metrics[f'precision_{avg}'] = p.evaluate()
        # recall
        #for avg in ('macro','micro'):
            r = Evaluate_Recall(f'recall_{avg}', '', average=avg)
            r.data = result
            metrics[f'recall_{avg}'] = r.evaluate()
        # f1
        #for avg in ('macro','micro'):
            f1 = Evaluate_F1Score(f'f1_{avg}', '', average=avg)
            f1.data = result
            metrics[f'f1_{avg}'] = f1.evaluate()

        # plot loss curves
        plt.figure(figsize=(10,8))
        plt.plot(self.method.train_losses, label='Training Loss')
        #plt.plot(self.method.test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        plt.show()

        return metrics
    
    def print_setup_summary(self):
        print('dataset:', self.dataset.dataset_name, ', method:', self.method.method_name,
              ', setting:', self.setting_name, ', result:', self.result.result_name, ', evaluation:', 'Accuracy, Precision, Recall, F1 Score')