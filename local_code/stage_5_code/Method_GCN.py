from local_code.base_class.method import method
import torch
import torch.nn.functional as F


class GCNLayer(torch.nn.Module):
    """
    A single GCN layer that learns a weight matrix W of shape [in_dim, out_dim].
    """

    def __init__(self, in_dim, out_dim, dtype=torch.float32):
        super(GCNLayer, self).__init__()
        # We wrap W as an nn.Parameter so that it appears in model.parameters()
        self.W = torch.nn.Parameter(torch.empty(in_dim, out_dim, dtype=dtype))
        torch.nn.init.xavier_uniform_(self.W, gain=1.0)

    def forward(self, A_hat, H):
        """
        A_hat: [N, N] (normalized adjacency with self‐loops)
        H:     [N, in_dim]
        returns: [N, out_dim]
        """
        H = torch.matmul(A_hat, H)      # [N, in_dim]
        H = torch.matmul(H, self.W)     # [N, out_dim]
        return F.relu(H)

    def parameters(self):
        # Only the weight matrix
        return [self.W]


class Method_GCN(method, torch.nn.Module):
    """
    Full‐batch GCN for node classification. Supports 1–5 GCN layers, then a final linear
    “classifier” to num_classes. During training, only the training indices contribute
    to the cross‐entropy loss. All training and (optional) train‐accuracy printouts
    go to `self.log_f` (a file handle) and also to stdout.
    """

    def __init__(
        self,
        mName,
        mDescription,
        input_dim,
        hidden_dims,   # e.g. [256,128,64] or length from 1 to 5
        num_classes,
        learning_rate=0.001,
        weight_decay=5e-4,
        dropout=0.5,
        max_epoch=100,
        dtype=torch.float32,
    ):
        method.__init__(self, mName, mDescription)
        torch.nn.Module.__init__(self)

        # Create GCN layers in a ModuleList
        dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(hidden_dims)):
            layers.append(GCNLayer(dims[i], dims[i + 1], dtype=dtype))
        self.layers = torch.nn.ModuleList(layers)

        # Final linear classifier from last hidden_dim -> num_classes
        self.fc = torch.nn.Linear(hidden_dims[-1], 64, dtype=dtype)
        self.activation = torch.nn.ReLU()
        self.classifier = torch.nn.Linear(64, num_classes, dtype=dtype)       #hidden_dims[-1]

        # Hyperparameters
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.max_epoch = max_epoch
        self.hidden_dims = hidden_dims  # just for logging
        self.num_layers = len(hidden_dims)
        self.num_classes = num_classes

        # We will append to these during training
        self.train_losses = []        # list of floats
        self.train_accuracies = []    # list of floats (one entry per epoch)

        # Set by Setting_GCN before calling run()
        self.logfile_path = None
        self.log_f = None
        self.count = 1

    def forward(self, A_hat, X):
        """
        Full‐batch forward pass on the entire graph:
          A_hat: [N, N]
          X:     [N, input_dim]
        Returns:
          logits: [N, num_classes]
        """
        H = X  # initial node features
        for layer in self.layers:
            H = layer(A_hat, H) 
            # dropout on hidden dims:
            if(self.count == 1):
                H = F.dropout(H, p=self.dropout, training=self.training)
                self.count += 1
        #hidden = self.activation(self.fc(H))  # [N, 64]
        #logits = self.classifier(hidden)  # [N, num_classes]
        #logits = self.classifier(H)  # [N, num_classes]
        logits = H
        return logits

    def train(self, graph, splits):
        """
        graph:  a dict { 'A': A_hat, 'X': features, 'y': labels, ... }
        splits: a dict { 'idx_train': LongTensor([...]),
                         'idx_val': LongTensor([...]),
                         'idx_test': LongTensor([...]) }
        """

        device = graph["X"].device
        A_hat = graph["A"]
        X = graph["X"]
        y = graph["y"]
        idx_train = splits["idx_train"]
        idx_test = splits["idx_test"]

        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        for epoch in range(self.max_epoch):
            #self.train()
            optimizer.zero_grad()

            logits = self.forward(A_hat, X)  # [N, num_classes]
            loss = F.cross_entropy(logits[idx_train], y[idx_train])

            loss.backward()
            optimizer.step()

            self.train_losses.append(loss.item())

            # Compute train‐accuracy on idx_train
            with torch.no_grad():
                _, preds = torch.max(logits, dim=1)  # [N]
                correct = preds[idx_train].eq(y[idx_train]).sum().item()
                train_acc = correct / float(idx_train.size(0))
                self.train_accuracies.append(train_acc)

            # compute test accuracy
                correct_test = preds[idx_test].eq(y[idx_test]).sum().item()
                test_acc = correct_test / float(idx_test.size(0))

            # Every 10 epochs, print a log line
            if epoch % 1 == 0:
                line = (
                    f"[GCN] Layers {self.num_layers:1d}  "
                    f"Epoch {epoch:3d}  "
                    f"Loss {loss.item():.4f}  "
                    f"Acc {train_acc:.4f}"
                    f" | Test Acc {test_acc:.4f}"
                )
                # to logfile
                print(line, file=self.log_f)
                # to stdout
                print(line)

    def test(self, graph, splits):
        """
        Returns a dict with keys 'pred_y' and 'true_y', both on CPU, for idx_test nodes.
        """
        device = graph["X"].device
        A_hat = graph["A"]
        X = graph["X"]
        y = graph["y"]
        idx_test = splits["idx_test"]

        #self.eval()
        with torch.no_grad():
            logits = self.forward(A_hat, X)  # [N, num_classes]
            _, preds = torch.max(logits, dim=1)  # [N]

        return {"pred_y": preds[idx_test].cpu(), "true_y": y[idx_test].cpu()}

    def run(self):
        """
        Called by Setting_GCN. Expects that:
          - self.data = { 'graph': graph, 'splits': splits }
          - self.logfile_path and self.log_f have been set (modes: append).
        """
        graph = self.data["graph"]
        splits = self.data["splits"]

        # 1) Training
        self.train(graph, splits)

        # 2) Testing / prediction
        result = self.test(graph, splits)
        return result

    # Override parameters() so that both GCNLayer‐W's and classifier parameters appear
    def parameters(self):
        param_list = []
        for layer in self.layers:
            param_list += layer.parameters()
        param_list += list(self.classifier.parameters())
        return param_list
