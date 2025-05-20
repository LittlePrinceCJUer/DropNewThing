from local_code.base_class.evaluate import evaluate
from sklearn.metrics import f1_score

class Evaluate_F1Score(evaluate):
    def __init__(self, eName=None, eDescription=None, average='macro'):
        super().__init__(eName, eDescription)
        self.average = average

    def evaluate(self):
        print(f'evaluating f1 score ({self.average})...')
        y_true = self.data['true_y']
        y_pred = self.data['pred_y']
        if hasattr(y_true, 'cpu'):
            y_true = y_true.cpu().numpy()
            y_pred = y_pred.cpu().numpy()
        return f1_score(y_true, y_pred, average=self.average)