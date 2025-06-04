import os
import sys
import numpy as np
import torch

# Make project root importable
target = os.path.abspath(__file__)
script_dir = os.path.dirname(target)
proj_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, proj_root)

from local_code.stage_5_code.Dataset_Loader import Dataset_Loader
from local_code.stage_5_code.Method_GCN import Method_GCN
from local_code.stage_5_code.Setting_GCN import Setting_GCN
from local_code.stage_5_code.Result_Saver import Result_Saver
from local_code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy

if __name__ == "__main__":
    np.random.seed(1)
    torch.manual_seed(1)

    datasets = [("citeseer", 6)]   #("pubmed",3), , ("cora", 7)
    base_hidden = [16, 3]   #1024, 512, 256, 128, 64, 32, 16

    for name, num_cls in datasets:
        base_hidden[-1] = num_cls  # Set the last hidden dimension to num_classes
        device_str = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"\n=== Running GCN on {name} ({device_str}) ===")

        # 1) We need to peek at input_dim & num_classes. Set the loader's folder first:
        data_obj = Dataset_Loader(seed=1, dName=name, dDescription=f"{name} citation network")
        data_obj.dataset_name = name

        # **IMPORTANT**: point dataset_source_folder_path to ".../data/stage_5_data/<name>"
        data_folder = os.path.join(proj_root, "data", "stage_5_data", name)
        data_obj.dataset_source_folder_path = data_folder

        loaded = data_obj.load()  # now it finds “<folder>/node” and “<folder>/link”
        in_dim = loaded["graph"]["X"].shape[1]
        num_classes = int(loaded["graph"]["y"].max().item() + 1)

        # 2) Loop over depth = 1..5
        for depth in range(2, len(base_hidden) + 1):
            hidden_dims = base_hidden[:depth]

            method_name = f"{name}_GCN_{depth}L"
            method_desc = f"{name} GCN with {depth} layers"
            method_obj = Method_GCN(
                mName=method_name,
                mDescription=method_desc,
                input_dim=in_dim,
                hidden_dims=hidden_dims,
                num_classes=num_classes,
                learning_rate=0.01,
                weight_decay=5e-4,
                dropout=0.5,
                max_epoch=1000,
            )

            # Prepare Result_Saver
            result_obj = Result_Saver("saver", "")
            result_obj.result_name = "saver"
            result_obj.result_destination_folder_path = proj_root + "/result/stage_5_result/"
            os.makedirs(result_obj.result_destination_folder_path, exist_ok=True)
            result_obj.result_destination_file_name = f"{name}_{depth}L_gcn_preds.pkl"

            # Accuracy evaluator (just for summary)
            evaluate_obj = Evaluate_Accuracy("accuracy", "")

            setting_name = f"{name}_GCN_{depth}L_exp"
            setting_desc = f"{name} GCN, depth={depth}"
            setting_obj = Setting_GCN(setting_name, setting_desc)
            setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)

            setting_obj.print_setup_summary()
            metrics = setting_obj.load_run_save_evaluate()

            print(f"--- {name} ({depth}L) Metrics ---")
            for k, v in metrics.items():
                print(f"{k:15s}: {v:.4f}")
