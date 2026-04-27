class Module:
    def __init__(self):
        self._parameters = []
        self._modules = []

    def forward(self, x):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError
    
    def __call__(self, x):
        return self.forward(x)
    
    def parameters(self):
        params = []

        # 자기 파라미터
        params.extend(self._parameters)

        # 자식 파라미터
        for module in self._modules:
            params.extend(module.parameters())

        return params
    
    def num_parameters(self):
        total = 0
        for p in self.parameters():
            if getattr(p, "requires_grad", True):
                total += p.data.size
        return total

    def add_module(self, module):
        self._modules.append(module)

    def add_parameter(self, param):
        self._parameters.append(param)

    def get_config(self):
        return {}