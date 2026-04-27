from .ModuleCart import Module

class Sequential(Module):
    def __init__(self, *layers):
        """
        여러 레이어(또는 활성화 함수)를 '순서대로' 저장하는 클래스

        Parameters
        ===================================================
        *layers :
            Linear, ReLU, Sigmoid 같은 객체들을 순서대로 입력
        ===================================================
        
        """
        super().__init__()
        self.layers = layers

        for layer in layers:
            self.add_module(layer)

    def forward(self, x):
        """
        입력 x를 저장된 레이어들에 순서대로 통과시킴
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def summary(self, input_shape):
        """
        모델 구조를 사람이 보기 좋게 출력한다.

        Parameters
        =================================
        input_shape : tuple
            모델 입력 shape
            예: (batch_size, input_dim)
        =================================
        """
        shape = input_shape
        total_params = 0

        print("=" * 72)
        print(f"{'Index':<8}{'Layer':<18}{'Input Shape':<18}{'Output Shape':<18}{'Params':<10}")
        print("=" * 72)

        for idx, layer in enumerate(self.layers):
            layer_name = layer.__class__.__name__
            input_s = shape

            if not hasattr(layer, "get_output_shape"):
                raise AttributeError(
                    f"{layer_name} 클래스에 get_output_shape()가 없습니다. "
                    f"summary()를 사용하려면 해당 메서드가 필요합니다."
                )

            output_s = layer.get_output_shape(input_s)
            params = layer.num_parameters()

            total_params += params

            print(
                f"{idx:<8}"
                f"{layer_name:<18}"
                f"{str(input_s):<18}"
                f"{str(output_s):<18}"
                f"{params:<10}"
            )

            shape = output_s

        print("=" * 72)
        print(f"Total params: {total_params}")
        print(f"Final output shape: {shape}")
        print("=" * 72)

    def __repr__(self):
        layer_names = [layer.__class__.__name__ for layer in self.layers]
        return f"Sequential({', '.join(layer_names)})"