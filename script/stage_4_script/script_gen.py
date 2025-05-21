import os, sys
import numpy as np
import torch

# make project root importable
HERE      = os.path.dirname(__file__)
PROJ_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PROJ_ROOT)

from local_code.stage_4_code.Dataset_Loader_Generation import Dataset_Loader_Generation
from local_code.stage_4_code.Method_TextGen_RNN      import Method_TextGen_RNN
from local_code.stage_4_code.Result_Saver            import Result_Saver
from local_code.stage_4_code.Setting_Generation      import Setting_Generation
from local_code.stage_4_code.Evaluate_Accuracy       import Evaluate_Accuracy

if __name__=='__main__':
    np.random.seed(1)
    torch.manual_seed(1)

    archs = ['rnn','lstm','gru']
    for arch in archs:
        print(f"\n=== Generation with {arch.upper()} ===")

        # 1) data loader & build vocab
        data_obj = Dataset_Loader_Generation(
            'text_generation',
            'jokes for generation',
            emb_dim=100
        )
        data_obj.dataset_source_folder_path = os.path.join(
            PROJ_ROOT, 'data','stage_4_data','text_generation'
        )
        data_obj.dataset_source_file_name = 'data'
        _ = data_obj.load()

        # 2) model
        vocab_size = len(data_obj.itos)
        epochs = 200
        method_obj = Method_TextGen_RNN(
            'TextGenRNN','joke generator',
            vocab_size=vocab_size,
            emb_dim=100,
            hidden_size=128,
            num_layers=1,
            rnn_arch=arch,
            max_epoch=epochs,
            learning_rate=1e-3,
            batch_size=64,
            pad_idx=data_obj.pad_idx,
            sos_idx=data_obj.sos_idx,
            eos_idx=data_obj.eos_idx
        )

        # 3) result saver
        result_obj = Result_Saver('saver','')
        result_obj.result_destination_folder_path = os.path.join(
            PROJ_ROOT,'result','stage_4_result'
        )
        result_obj.result_destination_file_name   = f"{arch}_gen_{epochs}E_preds.pkl"

        # 4) dummy evaluator
        evaluate_obj = Evaluate_Accuracy('none','')

        # 5) setting
        setting_obj = Setting_Generation(
            f'gen_{arch}',
            'Generation with '+arch.upper(),
            max_gen=30
        )
        setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
        setting_obj.print_setup_summary()
        setting_obj.load_run_save_evaluate()
