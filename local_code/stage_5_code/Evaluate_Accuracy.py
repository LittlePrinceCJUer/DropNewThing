from sklearn.metrics import accuracy_score
from local_code.base_class.evaluate import evaluate

class Evaluate_Accuracy(evaluate):
    def evaluate(self):
        print("evaluating accuracy...")
        y_true = self.data["true_y"]
        y_pred = self.data["pred_y"]
        if hasattr(y_true, "cpu"):
            y_true = y_true.cpu().numpy()
        if hasattr(y_pred, "cpu"):
            y_pred = y_pred.cpu().numpy()
        return accuracy_score(y_true, y_pred)
