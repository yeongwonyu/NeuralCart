import numpy as np

class MSELoss:
    def __init__(self):
        # backward 계산을 위해 forward 값 저장
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        """
        MSE Loss, Mean Squared Error Loss

        parameters
        ==================================
        y_pred: np.ndarray - 모델의 예측값
        y_true: np.ndarray - 실제 정답값
        ==================================

        Returns
        ==================================
        float
            MSE Loss 값
        ==================================        
        """
        self.y_pred = y_pred
        self.y_true = y_true

        return np.mean((y_pred - y_true)**2)
    
    def backward(self):
        """
        MSE Loss를 y_pred에 대해 미분한 값 반환
        """
        n = self.y_pred.size
        return 2*(self.y_pred - self.y_true) / n

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)