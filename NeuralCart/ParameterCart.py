import numpy as np

class Parameter:
    def __init__(self, data, requires_grad = True):
        """
        학습 가능한 파라미터를 저장하는 클래스

        Parameters
        ======================================
        data : np.ndarray
            실제 파라미터 값
        requires_grad : bool
            학습 대상 여부        
        """
        self.data = np.array(data, dtype=float)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data)

    def zero_grad(self):
        """
        gradient 초기화
        """
        self.grad = np.zeros_like(self.data)

    def __repr__(self):
        return (
            f"Parameter(shape={self.data.shape}, "
            f"requires_grad={self.requires_grad})"
        )