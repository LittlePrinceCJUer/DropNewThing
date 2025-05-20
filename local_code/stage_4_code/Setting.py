import os, sys
from local_code.base_class.setting import setting
import torch
import matplotlib.pyplot as plt
from local_code.stage_4_code.Evaluate_Accuracy   import Evaluate_Accuracy
from local_code.stage_4_code.Evaluate_Precision  import Evaluate_Precision
from local_code.stage_4_code.Evaluate_Recall     import Evaluate_Recall
from local_code.stage_4_code.Evaluate_F1Score    import Evaluate_F1Score

# compute project root dynamically
def _proj_root():
    fp = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(os.path.dirname(fp)))

class Setting(setting):
    def load_run_save_evaluate(self):
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        root = _proj_root() + '/data/stage_4_data/'

        self.dataset.dataset_source_folder_path = root
        self.dataset.dataset_source_file_name   = self.dataset.dataset_name
        loaded = self.dataset.load()
        train, test = loaded['train'], loaded['test']

        # to device
        for sp in (train, test):
            sp['X'] = sp['X'].to(device)
            sp['y'] = sp['y'].to(device)

        # run
        self.method.data = {'train':train, 'test':test}
        self.method.to(device)
        result = self.method.run()

        # save
        self.result.data = result
        self.result.save()

        # evaluate
        metrics = {}
        acc = Evaluate_Accuracy('acc','');           acc.data = result; metrics['accuracy'] = acc.evaluate()
        for avg in ('macro','micro'):
            p = Evaluate_Precision(f'prec_{avg}','',average=avg); p.data = result; metrics[f'precision_{avg}'] = p.evaluate()
            r = Evaluate_Recall   (f'recall_{avg}','',average=avg); r.data = result; metrics[f'recall_{avg}']    = r.evaluate()
            f1= Evaluate_F1Score  (f'f1_{avg}','',average=avg);    f1.data = result; metrics[f'f1_{avg}']        = f1.evaluate()

        # plot training loss curve and save to file
        plt.figure(figsize=(6,4))
        plt.plot(self.method.train_losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{self.method.method_name} Training Loss')
        plt.legend()

        # build filename: e.g. BC_10e_1e-3lr_64B.png
        epochs     = self.method.max_epoch
        lr         = self.method.lr
        bs         = self.method.batch_size
        layers     = self.method.num_layers
        fprefix = f"RNN_BC_{epochs}e_{lr}lr_{bs}B_{layers}L"

        # save plot
        fname = fprefix + ".png"
        save_dir = self.result.result_destination_folder_path
        print(f"Saving plot to {save_dir}...")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, fname))
        plt.close()

        # save metrics to txt file
        txt_fname = fprefix + ".txt"
        txt_path  = os.path.join(save_dir, txt_fname)
        with open(txt_path, 'w') as f:
            f.write("=== Metrics ===\n")
            for name, val in metrics.items():
                f.write(f"{name:15s}: {val:.4f}\n")
        print(f"Saved metrics to {txt_path}")

        return metrics