from sklearn.metrics import recall_score
from local_code.base_class.evaluate import evaluate

class Evaluate_Recall(evaluate):
    def __init__(self, eName=None, eDescription=None, average="macro"):
        super().__init__(eName, eDescription)
        self.average = average

    def evaluate(self):
        print(f"Evaluating Recall (average={self.average})...")
        y_true = self.data["true_y"]
        y_pred = self.data["pred_y"]
        if hasattr(y_true, "cpu"):
            y_true = y_true.cpu().numpy()
        if hasattr(y_pred, "cpu"):
            y_pred = y_pred.cpu().numpy()
        return recall_score(y_true, y_pred, average=self.average, zero_division=0)
