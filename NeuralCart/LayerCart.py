import numpy as np
from .InitCart import Xavier, He
from .ModuleCart import Module
from .ParameterCart import Parameter

class Linear(Module):
    def __init__(
            self, 
            in_features, 
            out_features, 
            activation = None,
            init = None,
            distribution = None,
            gain = 1.0
            ):
        
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.gain = gain

        if init is None:
            if activation in ["relu", "leaky_relu"]:
                init = "he"
            elif activation in ["sigmoid", "tanh", "linear", None]:
                init = "xavier"
            else:
                raise ValueError(f"지원하지 않는 activation입니다: {activation}")
            
        if distribution is None:
            if init == "xavier":
                distribution = "uniform"
            elif init == "he":
                distribution = "normal"
            else:
                raise ValueError(f"지원하지 않는 init입니다: {init}")
            
        self.init = init
        self.distribution = distribution

        if init == "xavier":
            if distribution == "normal":
                W = Xavier.normal(in_features, out_features, gain=gain)
            elif distribution == "uniform":
                W = Xavier.uniform(in_features, out_features, gain=gain)
            else:
                raise ValueError(f"지원하지 않는 distribution입니다: {distribution}")
            
        elif init == "he":
            if distribution == "normal":
                W = He.normal(in_features, out_features)
            elif distribution == "uniform":
                W = He.uniform(in_features, out_features)
            else:
                raise ValueError(f"지원하지 않는 distribution입니다: {distribution}")
            
        else:
            raise ValueError(f"지원하지 않는 init입니다: {init}")
        
        self.W = Parameter(W)
        self.b = Parameter(np.zeros((1, out_features)))
        
        self.add_parameter(self.W)
        self.add_parameter(self.b)

        self.x = None

    def forward(self, x):
        self.x = x
        return x @ self.W.data + self.b.data
    
    def backward(self, dout):
        # dout: Loss를 현재 Layer의 출력으로 미분한 값

        # dL/dW = x^T @ dout
        self.W.grad = self.x.T @ dout

        # dL/db = batch 방향 합
        self.b.grad = np.sum(dout, axis = 0, keepdims = True)

        # dL/dx = dout @ W^T
        dx = dout @ self.W.data.T
        return dx
    
    def get_output_shape(self, input_shape):
        """
        Linear Layer의 출력 shape을 계산한다.

        Linear는 마지막 차원이 in_features와 같아야 한다.

        예:
        input_shape = (batch_size, in_features)
        output_shape = (batch_size, out_features)
        """

        if len(input_shape) != 2:
            raise ValueError(
                f"Linear는 2차원 입력을 기대합니다. "
                f"예: (batch_size, in_features), 현재 입력 shape: {input_shape}"
            )

        batch_size, in_dim = input_shape

        if in_dim != self.in_features:
            raise ValueError(
                f"Linear shape mismatch: "
                f"입력 feature 수가 맞지 않습니다. "
                f"expected {self.in_features}, got {in_dim}"
            )

        return (batch_size, self.out_features)