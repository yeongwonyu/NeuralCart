from .ModuleCart import Module
import numpy as np


class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)
    
    def backward(self, dout):
        return dout * (self.x > 0)
    
    def get_output_shape(self, input_shape):
        """
        활성화 함수는 입력 shape을 바꾸지 않는다.
        """
        return input_shape


class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out
    
    def backward(self, dout):
        return dout * self.out * (1 - self.out)
    
    def get_output_shape(self, input_shape):
        return input_shape


class Tanh(Module):
    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, x):
        self.out = np.tanh(x)
        return self.out
    
    def backward(self, dout):
        return dout * (1 - self.out ** 2)
    
    def get_output_shape(self, input_shape):
        return input_shape