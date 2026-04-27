class Registry:
    """
    NeuralCart에서 사용할 구성요소들을 등록하고 불러오는 클래스

    예:
    "Linear" -> Linear 클래스
    "ReLU" -> ReLU 클래스
    "MSELoss" -> MSELoss 클래스
    """

    def __init__(self):
        # 등록된 객체들을 저장하는 딕셔너리
        self._items = {}

    def register(self, name, obj):
        """
        이름과 객체를 등록한다.

        Parameters
        ======================================
        name : str
            등록할 이름
        obj : class 또는 function
            실제 사용할 클래스 또는 함수
        ======================================
        """
        if name in self._items:
            raise ValueError(f"이미 등록된 이름입니다: {name}")

        self._items[name] = obj

    def get(self, name):
        """
        이름에 해당하는 객체를 반환한다.
        """
        if name not in self._items:
            raise KeyError(f"등록되지 않은 이름입니다: {name}")

        return self._items[name]

    def exists(self, name):
        """
        해당 이름이 등록되어 있는지 확인한다.
        """
        return name in self._items

    def list_items(self):
        """
        현재 등록된 이름 목록을 반환한다.
        """
        return list(self._items.keys())
    

from .LayerCart import Linear
from .ActivationCart import ReLU, Sigmoid, Tanh
from .SequentialCart import Sequential
from .LossCart import MSELoss
from .OptimizerCart import GD, SGD, Adam


LAYER_REGISTRY = Registry()
LOSS_REGISTRY = Registry()
OPTIMIZER_REGISTRY = Registry()


# Layer / Activation 등록
LAYER_REGISTRY.register("Linear", Linear)
LAYER_REGISTRY.register("ReLU", ReLU)
LAYER_REGISTRY.register("Sigmoid", Sigmoid)
LAYER_REGISTRY.register("Tanh", Tanh)
LAYER_REGISTRY.register("Sequential", Sequential)


# Loss 등록
LOSS_REGISTRY.register("MSELoss", MSELoss)


# Optimizer 등록
OPTIMIZER_REGISTRY.register("GD", GD)
OPTIMIZER_REGISTRY.register("SGD", SGD)
OPTIMIZER_REGISTRY.register("Adam", Adam)