import os
import torch
import matplotlib.pyplot as plt
from local_code.base_class.setting import setting
from local_code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_5_code.Evaluate_Precision import Evaluate_Precision
from local_code.stage_5_code.Evaluate_Recall import Evaluate_Recall
from local_code.stage_5_code.Evaluate_F1Score import Evaluate_F1Score

def _proj_root():
    fp = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(os.path.dirname(fp)))

class Setting_GCN(setting):
    """
    Opens/appends to `train_<dataset>.txt`, logs training + testing. Assumes
    that `Dataset_Loader.load()` returns adjacency as a sparse tensor. On MPS
    we convert it to dense before moving to the device.
    """

    def load_run_save_evaluate(self):
        # 1) Device selection
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # 2) Compute the path to this dataset's folder
        root = _proj_root() + "/data/stage_5_data/"
        dataset_name = self.dataset.dataset_name
        dataset_folder = os.path.join(root, dataset_name)

        # 3) Ensure result folder exists
        save_dir = self.result.result_destination_folder_path
        os.makedirs(save_dir, exist_ok=True)

        # 4) Open (append) logfile
        logfile_path = os.path.join(save_dir, f"train_{dataset_name}.txt")
        log_f = open(logfile_path, "a", encoding="utf8")

        # 5) Write header
        epochs = self.method.max_epoch
        lr = self.method.lr
        layers = self.method.num_layers
        hdims = "_".join(str(d) for d in self.method.hidden_dims)
        header = (
            f"GCN_{dataset_name}_{layers}L_{epochs}e_{lr}lr , "
            f"hidden_dims: {hdims} , "
            f"dataset: {dataset_name} , "
            f"method: {self.method.method_name} , "
            f"setting: {self.setting_name} , "
            f"result: {self.result.result_name} , "
            f"evaluation: accuracy"
        )
        print(header, file=log_f)
        print(header)
        print(f"loading data for {dataset_name}...", file=log_f)
        print(f"loading data for {dataset_name}...")

        # 6) Load data
        self.dataset.dataset_source_folder_path = dataset_folder
        loaded = self.dataset.load()  # now loader reads “<folder>/node” & “<folder>/link”
        graph = loaded["graph"]
        splits = loaded["train_test_val"]

        # 7) Move features/labels to device
        graph["X"] = graph["X"].to(device)
        graph["y"] = graph["y"].to(device)

        # 8) FIX: convert sparse adjacency to dense, then send to MPS/CPU
        #     instead of `graph["A"] = graph["utility"]["A"].to(device)`
        graph["A"] = graph["utility"]["A"].to_dense().to(device)

        # 9) Give Method_GCN the logfile handle
        self.method.logfile_path = logfile_path
        self.method.log_f = log_f

        # 10) Train
        self.method.data = {"graph": graph, "splits": splits}
        self.method.to(device)

        print(f">> {self.method.method_name} training...", file=log_f)
        print(f">> {self.method.method_name} training...")
        result = self.method.run()
        log_f.close()

        # 11) Re‐open logfile to append testing metrics
        log_f = open(logfile_path, "a", encoding="utf8")
        print(f">> {self.method.method_name} testing...", file=log_f)
        print(f">> {self.method.method_name} testing...")

        # 12) Evaluate
        acc = Evaluate_Accuracy("accuracy", "")
        acc.data = result
        print("evaluating accuracy...", file=log_f);       print("evaluating accuracy...")
        acc_val = acc.evaluate()

        p_macro = Evaluate_Precision("precision_macro", "", average="macro")
        p_macro.data = result
        print("evaluating precision (macro)...", file=log_f); print("evaluating precision (macro)...")
        p_ma = p_macro.evaluate()

        r_macro = Evaluate_Recall("recall_macro", "", average="macro")
        r_macro.data = result
        print("evaluating recall (macro)...", file=log_f);    print("evaluating recall (macro)...")
        r_ma = r_macro.evaluate()

        f1_macro = Evaluate_F1Score("f1_macro", "", average="macro")
        f1_macro.data = result
        print("evaluating f1 (macro)...", file=log_f);        print("evaluating f1 (macro)...")
        f1_ma = f1_macro.evaluate()

        p_micro = Evaluate_Precision("precision_micro", "", average="micro")
        p_micro.data = result
        print("evaluating precision (micro)...", file=log_f); print("evaluating precision (micro)...")
        p_mi = p_micro.evaluate()

        r_micro = Evaluate_Recall("recall_micro", "", average="micro")
        r_micro.data = result
        print("evaluating recall (micro)...", file=log_f);    print("evaluating recall (micro)...")
        r_mi = r_micro.evaluate()

        f1_micro = Evaluate_F1Score("f1_micro", "", average="micro")
        f1_micro.data = result
        print("evaluating f1 (micro)...", file=log_f);        print("evaluating f1 (micro)...")
        f1_mi = f1_micro.evaluate()

        # 13) Save predictions
        self.result.data = result
        self.result.save()

        # 14) Append final metrics to logfile
        print("--- Metrics ---", file=log_f); print("--- Metrics ---")
        metrics = {
            "accuracy": acc_val,
            "precision_macro": p_ma,
            "recall_macro": r_ma,
            "f1_macro": f1_ma,
            "precision_micro": p_mi,
            "recall_micro": r_mi,
            "f1_micro": f1_mi,
        }
        for k, v in metrics.items():
            line = f"{k:15s}: {v:.4f}"
            print(line, file=log_f)
            print(line)

        # 15) Plot training curves (loss & optional train‐acc)
        plt.figure(figsize=(8, 6))
        plt.plot(self.method.train_losses, label="Train Loss")
        # if hasattr(self.method, "train_accuracies"):
        #     # if Method_GCN logs training accuracy every N epochs
        #     epochs_shown = [i for i in range(self.method.max_epoch) if i % 10 == 0]
        #     acc_vals = [self.method.train_accuracies[i] for i in epochs_shown]
        #     plt.plot(epochs_shown, acc_vals, label="Train Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(
            f"{self.method.method_name}  "
            #f"{self.method.num_layers}L  "
            f"{self.method.max_epoch}e  "
            f"{self.method.lr}lr  Training Curves"
        )
        plt.legend()

        plot_fname = f"GCN_{dataset_name}_{self.method.num_layers}L_{self.method.max_epoch}e_{self.method.lr}lr.png"
        plt.savefig(os.path.join(save_dir, plot_fname))
        plt.close()

        log_f.close()
        return metrics
