from local_code.base_class.method import method
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from local_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy

class Method_TextRNN(method, nn.Module):
    def __init__(self, mName, mDescription,
                 emb_dim=100, hidden_size=128, num_layers=2,
                 max_epoch=10, learning_rate=1e-3,
                 batch_size=64):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.rnn = nn.RNN(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.classifier = nn.Linear(hidden_size, 1)

        self.max_epoch  = max_epoch
        self.lr         = learning_rate
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.train_losses = []

    def forward(self, x):
        # x: [batch, seq_len, emb_dim]
        out, _ = self.rnn(x)                # out: [batch, seq_len, hidden]
        h_sum = out.sum(dim=1)              # sum over time: [batch, hidden]
        logits = self.classifier(h_sum)     # [batch, 1]
        return logits.squeeze(1)            # [batch]

    def train(self, X_train, y_train, X_test=None, y_test=None):
        device = X_train.device
        # DataLoader on CPU
        X_cpu, y_cpu = X_train.cpu(), y_train.cpu()
        ds = TensorDataset(X_cpu, y_cpu)
        loader = DataLoader(
            ds, batch_size=self.batch_size,
            shuffle=True, num_workers=8,
            pin_memory=True, persistent_workers=True, prefetch_factor=2
        )

        optim   = torch.optim.Adam(self.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()
        #acc_eval = Evaluate_Accuracy('train-acc','')

        for epoch in range(self.max_epoch):
            #self.train()
            running_loss = 0.0
            for Xb_cpu, yb_cpu in loader:
                Xb = Xb_cpu.to(device, non_blocking=True)
                yb = yb_cpu.to(device, non_blocking=True).float()
                logits = self.forward(Xb)
                loss = loss_fn(logits, yb)
                optim.zero_grad(); loss.backward(); optim.step()
                running_loss += loss.item() * Xb.size(0)

            epoch_loss = running_loss / len(X_train)
            self.train_losses.append(epoch_loss)

            if epoch % 5 == 0:
                total, correct = 0, 0
                with torch.no_grad():
                    for xb_cpu, yb_cpu in loader:
                        xb = xb_cpu.to(device, non_blocking=True)
                        yb = yb_cpu.to(device, non_blocking=True).float()
                        out = self.forward(xb)
                        preds = (torch.sigmoid(out) > 0.5).long()
                        correct += preds.eq(yb.long()).sum().item()
                        total += yb.size(0)
                train_acc = correct/total
                print(f'[RNN] Epoch {epoch:2d}  Loss {epoch_loss:.4f}  Acc {train_acc:.4f}')

    def test(self, X):
        #self.eval()
        with torch.no_grad():
            logits = self.forward(X)
            return (logits > 0).long()

    def run(self):
        tr, te = self.data['train'], self.data['test']
        print('>> TextRNN training...')
        self.train(tr['X'], tr['y'])
        print('>> TextRNN testing...')
        pred = self.test(te['X'])
        return {'pred_y': pred, 'true_y': te['y']}