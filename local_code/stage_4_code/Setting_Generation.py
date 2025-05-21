import os, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from local_code.base_class.setting import setting

class Setting_Generation(setting):
    def __init__(self, sName=None, sDescription=None, max_gen=None):
        super().__init__(sName, sDescription)
        # when not None, only generate this many samples
        self.max_gen = max_gen

    def load_run_save_evaluate(self):
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # --- load train split ---
        root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'data','stage_4_data','text_generation'
        )
        self.dataset.dataset_source_folder_path = root
        self.dataset.dataset_source_file_name   = 'data'
        loaded = self.dataset.load()
        train = loaded['train']
        train['X'] = train['X'].to(device)
        train['y'] = train['y'].to(device)

        # --- train ---
        self.method.pretrained_emb = self.dataset.embedding_matrix
        self.method.data = train
        self.method.to(device)
        print(">> TextGen RNN training...")
        self.method.train(train['X'], train['y'])

        # --- save checkpoint & loss curve ---
        gen_dir = os.path.join(self.result.result_destination_folder_path, 'gen')
        os.makedirs(gen_dir, exist_ok=True)
        ckpt = os.path.join(gen_dir, f"{self.method.rnn_arch}_model.pt")
        torch.save(self.method.state_dict(), ckpt)

        plt.figure(figsize=(6,4))
        plt.plot(self.method.train_losses, label='Train Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.title(f'{self.method.method_name} Loss')
        plt.legend()
        png = os.path.join(
            gen_dir,
            f"Gen_{self.method.rnn_arch}_{self.method.max_epoch}e_{self.method.lr}lr_{self.method.batch_size}B.png"
        )
        plt.savefig(png)
        plt.close()

        # --- generation, limited by max_gen ---
        limit = self.max_gen if self.max_gen is not None else len(self.dataset.raw_jokes)
        outputs = []
        for tokens in self.dataset.raw_jokes[:limit]:
            prefix = tokens[:6]
            prefix_ids = [ self.dataset.stoi.get(tok, self.dataset.unk_idx)
                           for tok in prefix ]
            gen_ids = self.method.generate(prefix_ids, max_gen_len=200)
            gen_toks = [ self.dataset.itos[i] for i in gen_ids ]
            outputs.append(' '.join(prefix + gen_toks))

        out_txt = os.path.join(gen_dir, f'{self.method.rnn_arch}_generated_jokes.txt')
        with open(out_txt, 'w', encoding='utf8') as f:
            for line in outputs:
                f.write(line + "\n")

        print(f"Saved {limit} generations to {out_txt}")
        return {}
