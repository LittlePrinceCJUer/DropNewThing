'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn


class Method_MLP(method, nn.Module):
    # it defines the max rounds to train the model
    max_epoch = 500
    # it defines the default learning rate for gradient descent based optimizer for model learning
    log_interval = 10

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription, lr=1e-3):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.fc_layer_1 = nn.Linear(784, 256)
        # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        self.activation_func_1 = nn.ReLU()
        self.fc_layer_2 = nn.Linear(256, 128)
        self.activation_func_2 = nn.ReLU()
        # layer 3
        self.fc_layer_3 = nn.Linear(128, 10)
        #self.activation_func_3 = nn.ReLU()
        # layer 4
        #self.fc_layer_4 = nn.Linear(64, 10)
        # check here for nn.Softmax doc: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        self.activation_func_3 = nn.Softmax(dim=1)  # don't use it if using CrossEntropyLoss, it involves that

        # to record loss curves
        self.train_losses = []
        self.test_losses = []
        self.learning_rate = lr

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        # hidden layer embeddings
        h_1 = self.activation_func_1(self.fc_layer_1(x))
        h_2 = self.activation_func_2(self.fc_layer_2(h_1))
        #h_3 = self.activation_func_3(self.fc_layer_3(h_2))
        # outout layer result
        logits = self.fc_layer_3(h_2)
        return logits
    
    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X_train, y_train, X_test=None, y_test=None):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        #loss_function = nn.MSELoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('train-acc', '')

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            logits = self.forward(X_train)
            # calculate the training loss
            train_loss = loss_function(logits, y_train)

            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            optimizer.zero_grad()
            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            self.train_losses.append(train_loss.item())
            
            # # ---- test loss ----
            # if X_test is not None and epoch % self.loss_record_interval == 0:
            #     with torch.no_grad():
            #         t_logits = self.forward(X_test)
            #         t_loss = loss_function(t_logits, y_test)
            #         self.test_losses.append(t_loss.item())

            # ---- periodic logging ----
            if epoch % self.log_interval == 0:
                preds = logits.argmax(dim=1)
                accuracy_evaluator.data = {'true_y': y_train, 'pred_y': preds}
                acc = accuracy_evaluator.evaluate()
                print(f'Epoch {epoch:3d} | Train Loss: {train_loss.item():.4f} | Train Acc: {acc:.4f}')
            
    
    def test(self, X):
        # do the testing, and result the result
        logits = self.forward(X)
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return logits.argmax(dim = 1)
    
    def run(self):
        print('method_MLP running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}