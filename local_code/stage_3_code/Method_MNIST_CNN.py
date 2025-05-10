from local_code.base_class.method import method
import torch, torch.nn.functional as F
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy

class Method_MNIST_CNN(method, nn.Module):
    def __init__(self, mName, mDescription, max_epoch=60, learning_rate=1e-3, batch_size=64):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # architecture
        self.conv1 = nn.Conv2d(1, 6, 3, 1, 1)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 10, 3, 1)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(10, 24, 3, 1)
        self.maxpool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2 * 2 * 24, 64)
        self.fc2 = nn.Linear(64, 10)

        # training hyperparams
        self.max_epoch  = max_epoch
        self.lr         = learning_rate
        self.batch_size = batch_size

        # to record loss curves
        self.train_losses = []
        #self.test_losses  = []

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = self.maxpool1(x)
        x = F.relu(self.conv2(x)); x = self.maxpool2(x)
        x = F.relu(self.conv3(x)); x = self.maxpool3(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def train(self, X_train, y_train, X_test=None, y_test=None):
        # create DataLoader for mini-batch GD
        train_ds     = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        loss_fn   = nn.CrossEntropyLoss()
        acc_eval  = Evaluate_Accuracy('train-acc', '')

        for epoch in range(self.max_epoch):
            running_loss = 0.0
            for X_b, y_b in train_loader:
                logits = self.forward(X_b)
                loss   = loss_fn(logits, y_b)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * X_b.size(0)

            epoch_loss = running_loss / len(X_train)
            self.train_losses.append(epoch_loss)

            if epoch % 10 == 0:
                preds = self.forward(X_train).argmax(dim=1)
                acc_eval.data = {'true_y': y_train, 'pred_y': preds}
                print(f'[MNIST] Epoch {epoch:3d}  Loss {epoch_loss:.4f}  Acc {acc_eval.evaluate():.4f}')

    def test(self, X):
        with torch.no_grad():
            return self.forward(X).argmax(dim=1)

    def run(self):
        tr, te = self.data['train'], self.data['test']
        print('>> MNIST_CNN training...')
        self.train(tr['X'], tr['y'], te['X'], te['y'])
        print('>> MNIST_CNN testing...')
        pred = self.test(te['X'])
        return {'pred_y': pred, 'true_y': te['y']}
