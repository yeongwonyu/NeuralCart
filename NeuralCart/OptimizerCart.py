import numpy as np

class Optimizer:
    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr = lr

    def _trainable_params(self):
        # requires_grad=True인 Parameter만 업데이트 대상
        return [
            p for p in self.model.parameters()
            if getattr(p, "requires_grad", True)
        ]
    
    def zero_grad(self):
        # 모든 gradient 초기화
        for p in self._trainable_params():
            p.zero_grad()
    
    def step(self):
        raise NotImplementedError
    
class GD(Optimizer):
    """
    Gradient Descent

    현재 NeuralCart 구조에서는 전체 데이터를 한 번에 넣으면 GD 처럼 동작한다.
    """
    def step(self):
        for p in self._trainable_params():
            p.data -= self.lr * p.grad


class SGD(Optimizer):
    """
    Stochastic Gradient Descent

    현재 구현에서는 업데이트 방식은 GD와 동일하다
    학습 루프에서 데이터를 한 개 또는 mini-batch 단위로 넣을 때 생긴다.
    """
    def step(self):
        for p in self._trainable_params():
            p.data -= self.lr * p.grad


class Adam(Optimizer):
    """
    Adam Optimizer

    m: gradient의 1차 모멘트
    v: gradient의 2차 모멘트
    beta1, beta2: 모멘트 이동평균 계수
    eps: 0으로 나누는 것을 방지하는 작은 값
    """
    def __init__(self, model, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(model, lr)

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        self.m = {}
        self.v = {}

        for idx, p in enumerate(self._trainable_params()):
            self.m[idx] = np.zeros_like(p.data)
            self.v[idx] = np.zeros_like(p.data)

    def step(self):
        self.t += 1

        for idx, p in enumerate(self._trainable_params()):
            g = p.grad

            # 1차 모멘트 업데이트
            self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * g

            # 2차 모멘트 업데이트
            self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (g ** 2)

            # bias correction
            m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
            v_hat = self.v[idx] / (1 - self.beta2 ** self.t)

            # parameter update
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)