import os, sys

# 1) locate this script’s folder
file_path = os.path.abspath(__file__)
script_dir = os.path.dirname(file_path)
# 2) go up two levels to the project root
proj_root = os.path.dirname(os.path.dirname(script_dir))

from local_code.base_class.setting import setting
import torch, matplotlib.pyplot as plt
from local_code.stage_3_code.Evaluate_Accuracy   import Evaluate_Accuracy
from local_code.stage_3_code.Evaluate_Precision  import Evaluate_Precision
from local_code.stage_3_code.Evaluate_Recall     import Evaluate_Recall
from local_code.stage_3_code.Evaluate_F1Score    import Evaluate_F1Score

class Setting(setting):
    def load_run_save_evaluate(self):
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # Nvidia GPU:
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load splits
        self.dataset.dataset_source_folder_path = proj_root + '/data/stage_3_data/'
        self.dataset.dataset_source_file_name   = self.dataset.dataset_name
        loaded = self.dataset.load()
        train  = loaded['train']
        test   = loaded['test']

        # move to device
        for split in (train, test):
            split['X'] = split['X'].to(device)
            split['y'] = split['y'].to(device)

        # attach & run
        self.method.data = {'train': train, 'test': test}
        self.method.to(device)
        result = self.method.run()

        # save raw predictions
        self.result.data = result
        self.result.save()

        # compute metrics
        metrics = {}
        acc = Evaluate_Accuracy('accuracy',''); acc.data = result
        metrics['accuracy'] = acc.evaluate()
        for avg in ('macro','micro'):
            p = Evaluate_Precision(f'precision_{avg}','',average=avg); p.data = result
            r = Evaluate_Recall   (f'recall_{avg}','',   average=avg); r.data = result
            f1= Evaluate_F1Score  (f'f1_{avg}','',        average=avg); f1.data = result
            metrics[f'precision_{avg}'] = p.evaluate()
            metrics[f'recall_{avg}']    = r.evaluate()
            metrics[f'f1_{avg}']        = f1.evaluate()

        # plot loss curves
        plt.figure(figsize=(6,4))
        plt.plot(self.method.train_losses, label='Train Loss')
        #plt.plot(self.method.test_losses,  label='Test Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.title(f'{self.method.method_name} Loss')
        plt.legend(); plt.show()

        return metrics
