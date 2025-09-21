import numpy as np
from sklearn.metrics import top_k_accuracy_score


class TopK:
    def __init__(self, k: int, levels: list = None, topk: float = None, y_score: np.ndarray = None, y_true: np.ndarray = None):
        self.k = k
        # zero are considered None
        self.top_k_accuracy_score = topk if topk is not None else top_k_accuracy_score(y_true=y_true, y_score=y_score, normalize=False, k=k, labels=np.arange(1, len(levels) + 1))
