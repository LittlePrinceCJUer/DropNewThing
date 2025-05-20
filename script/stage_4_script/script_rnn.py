import os, sys
# make project root importable
target = os.path.abspath(__file__)
script_dir = os.path.dirname(target)
proj_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, proj_root)

import numpy as np
import torch

from local_code.stage_4_code.Dataset_Loader   import Dataset_Loader
from local_code.stage_4_code.Method_Classifier_RNN    import Method_TextRNN
from local_code.stage_4_code.Result_Saver      import Result_Saver
from local_code.stage_4_code.Setting           import Setting
from local_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy

if __name__ == '__main__':
    np.random.seed(1); torch.manual_seed(1)

    # single experiment on text_classification
    data_obj   = Dataset_Loader('text_classification', 'Sentiment data')

    archs = ['rnn', 'lstm', 'gru']

    for arch in archs:
        method_obj = Method_TextRNN('TextRNN','Binary sentiment RNN',
                                    emb_dim=100, hidden_size=128, num_layers=2,
                                    max_epoch=40, learning_rate=1e-3,
                                    batch_size=128, rnn_arch = arch)
        
        result_obj = Result_Saver('saver','')
        result_obj.result_destination_folder_path = proj_root + '/result/stage_4_result/'
        result_obj.result_destination_file_name   = 'rnn_preds.pkl'

        evaluate_obj = Evaluate_Accuracy('accuracy','')

        setting_obj = Setting('rnn_exp','RNN sentiment classification')
        setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
        setting_obj.print_setup_summary()
        metrics = setting_obj.load_run_save_evaluate()

        print('--- Metrics ---')
        for k,v in metrics.items(): print(f'{k:15s}: {v:.4f}')