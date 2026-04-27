class ShapeChecker:
    """
    모델의 레이어들이 서로 연결 가능한지 shape을 기준으로 검사하는 클래스

    현재 버전은 Sequential 모델을 기준으로 동작한다.
    즉, 입력이 layer1 → layer2 → layer3 순서로 흐르는 구조를 검사한다.
    """

    @staticmethod
    def check(model, input_shape, verbose=True):
        """
        모델 전체의 shape 흐름을 검사한다.

        Parameters
        ==================================================
        model :
            Sequential 모델 객체

        input_shape : tuple
            모델에 들어가는 입력 shape
            예: (batch_size, input_dim)
            예: (4, 2)

        verbose : bool
            True이면 각 레이어별 shape 변화 출력
        ==================================================

        Returns
        ==================================================
        shape : tuple
            최종 출력 shape
        ==================================================
        """

        shape = input_shape

        if verbose:
            print("========== Shape Check ==========")
            print(f"Input shape: {shape}")

        for idx, layer in enumerate(model.layers):
            layer_name = layer.__class__.__name__

            # 각 레이어는 get_output_shape 메서드를 가지고 있어야 함
            if not hasattr(layer, "get_output_shape"):
                raise AttributeError(
                    f"{layer_name} 클래스에 get_output_shape() 메서드가 없습니다. "
                    f"ShapeCart를 사용하려면 각 레이어에 get_output_shape()를 구현해야 합니다."
                )

            prev_shape = shape
            shape = layer.get_output_shape(shape)

            if verbose:
                print(f"[{idx}] {layer_name}: {prev_shape} -> {shape}")

        if verbose:
            print(f"Final output shape: {shape}")
            print("=================================")

        return shape