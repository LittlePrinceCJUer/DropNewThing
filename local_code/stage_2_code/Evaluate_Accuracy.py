'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score

class Evaluate_Accuracy(evaluate):
    def evaluate(self):
        print('evaluating performance...')
        y_true = self.data['true_y']
        y_pred = self.data['pred_y']
        # move tensors to CPU & convert to numpy if needed
        if hasattr(y_true, 'cpu'):
            y_true = y_true.cpu().numpy()
        if hasattr(y_pred, 'cpu'):
            y_pred = y_pred.cpu().numpy()
        return accuracy_score(y_true, y_pred)
        