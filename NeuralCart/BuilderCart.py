import json

from .RegistryCart import (
    LAYER_REGISTRY,
    LOSS_REGISTRY,
    OPTIMIZER_REGISTRY
)
from .SequentialCart import Sequential


class Builder:
    """
    config 파일 또는 dict를 기반으로 NeuralCart 모델, loss, optimizer를 생성하는 클래스
    """

    @staticmethod
    def build_layer(layer_config):
        """
        하나의 layer config를 실제 layer 객체로 변환한다.
        """

        layer_type = layer_config["type"]

        # type을 제외한 나머지 값은 생성자 인자로 사용
        kwargs = {
            key: value
            for key, value in layer_config.items()
            if key != "type"
        }

        layer_class = LAYER_REGISTRY.get(layer_type)
        return layer_class(**kwargs)

    @staticmethod
    def build_model(model_config):
        """
        model config를 실제 모델 객체로 변환한다.

        현재는 Sequential 모델만 지원한다.
        """

        model_type = model_config.get("type", "Sequential")

        if model_type != "Sequential":
            raise ValueError(
                f"현재 Builder는 Sequential만 지원합니다. 입력된 type: {model_type}"
            )

        layers = []

        for layer_config in model_config["layers"]:
            layer = Builder.build_layer(layer_config)
            layers.append(layer)

        return Sequential(*layers)

    @staticmethod
    def build_loss(loss_config):
        """
        loss config를 실제 loss 객체로 변환한다.
        """

        loss_type = loss_config["type"]

        kwargs = {
            key: value
            for key, value in loss_config.items()
            if key != "type"
        }

        loss_class = LOSS_REGISTRY.get(loss_type)
        return loss_class(**kwargs)

    @staticmethod
    def build_optimizer(optimizer_config, model):
        """
        optimizer config를 실제 optimizer 객체로 변환한다.

        optimizer는 model이 필요하므로 model을 함께 받는다.
        """

        optimizer_type = optimizer_config["type"]

        kwargs = {
            key: value
            for key, value in optimizer_config.items()
            if key != "type"
        }

        optimizer_class = OPTIMIZER_REGISTRY.get(optimizer_type)
        return optimizer_class(model, **kwargs)

    @staticmethod
    def build_from_config(config):
        """
        dict config를 받아 model, loss_fn, optimizer를 생성한다.
        """

        model = Builder.build_model(config["model"])
        loss_fn = Builder.build_loss(config["loss"])
        optimizer = Builder.build_optimizer(config["optimizer"], model)

        return model, loss_fn, optimizer

    @staticmethod
    def build_from_json(json_path):
        """
        json 파일 경로를 받아 model, loss_fn, optimizer를 생성한다.
        """

        with open(json_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        return Builder.build_from_config(config)