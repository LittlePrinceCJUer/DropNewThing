# local_code/stage_4_code/Method_TextGen_RNN.py

from local_code.base_class.method import method
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

class Method_TextGen_RNN(method, nn.Module):
    def __init__(self, mName, mDescription,
                 vocab_size, emb_dim, hidden_size,
                 num_layers=1,
                 rnn_arch='lstm',
                 max_epoch=10,
                 learning_rate=1e-3,
                 batch_size=64,
                 pad_idx=0, sos_idx=2, eos_idx=3):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.rnn_arch    = rnn_arch

        if rnn_arch == 'rnn':
            self.rnn = nn.RNN(emb_dim, hidden_size,
                              num_layers=num_layers,
                              batch_first=True)
        elif rnn_arch == 'lstm':
            self.rnn = nn.LSTM(emb_dim, hidden_size,
                               num_layers=num_layers,
                               batch_first=True)
        elif rnn_arch == 'gru':
            self.rnn = nn.GRU(emb_dim, hidden_size,
                              num_layers=num_layers,
                              batch_first=True)
        else:
            raise ValueError(f"Unknown rnn_arch: {rnn_arch}")

        self.classifier = nn.Linear(hidden_size, vocab_size)
        self.max_epoch  = max_epoch
        self.lr         = learning_rate
        self.batch_size = batch_size

        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.train_losses = []

    def forward(self, x):
        emb = self.embedding(x)     # [B, T, E]
        out, _ = self.rnn(emb)      # [B, T, H]
        logits = self.classifier(out)  # [B, T, V]
        return logits

    def train(self, X_train, y_train, X_test=None, y_test=None):
        device = X_train.device
        # load pretrained embeddings if provided
        if hasattr(self, 'pretrained_emb'):
            self.embedding.weight.data.copy_(self.pretrained_emb.to(device))

        ds = TensorDataset(X_train.cpu(), y_train.cpu())
        loader = DataLoader(ds,
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=4,
                            pin_memory=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        loss_fn   = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

        for epoch in range(self.max_epoch):
            #self.train()
            total_loss = 0.0

            for xb_cpu, yb_cpu in loader:
                xb = xb_cpu.to(device, non_blocking=True)
                yb = yb_cpu.to(device, non_blocking=True)

                out = self.forward(xb)               # [B, T, V]
                # permute to [B, V, T] and compare to [B, T]
                loss = loss_fn(out.permute(0,2,1), yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * xb.size(0)

            avg_loss = total_loss / len(X_train)
            self.train_losses.append(avg_loss)
            if epoch % 5 == 0:
                print(f"[GEN] Epoch {epoch:2d}  Loss {avg_loss:.4f}")

    def generate(self, prefix_ids, max_gen_len=100):
        device = next(self.parameters()).device
        seq = prefix_ids.copy()
        #self.eval()
        with torch.no_grad():
            for _ in range(max_gen_len):
                inp = torch.tensor([seq], device=device)
                out = self.forward(inp)           # [1, T, V]
                nxt = int(out[0, -1].argmax().item())
                if nxt == self.eos_idx:
                    break
                seq.append(nxt)
        return seq[len(prefix_ids):]

    def run(self):
        # not used by Setting_Generation
        return {}
