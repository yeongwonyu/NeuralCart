import numpy as np

class Xavier:
    @staticmethod
    def normal(in_features, out_features, gain=1.0):
        # Xavier Normal Initialization
        # 목적: forward와 backward에서 분산을 일정하게 유지
        # 주로 sigmoid, tanh 같은 활성화 함수에서 사용
        std = np.sqrt(2.0 / (in_features + out_features))
        return np.random.randn(in_features, out_features) * std * gain

    @staticmethod
    def uniform(in_features, out_features, gain=1.0):
        # Xavier Uniform Initialization
        # 목적: Xavier normal과 동일하지만 균등분포 사용
        limit = np.sqrt(6.0 / (in_features + out_features))
        return np.random.uniform(-gain * limit, gain * limit, (in_features, out_features))


class He:
    @staticmethod
    def normal(in_features, out_features):
        # He Normal Initialization (Kaiming Normal)
        # 목적: ReLU 계열에서 주로 사용
        std = np.sqrt(2.0 / in_features)
        return np.random.randn(in_features, out_features) * std

    @staticmethod
    def uniform(in_features, out_features):
        # He Uniform Initialization (Kaiming Uniform)
        # 목적: He normal과 동일 목적
        limit = np.sqrt(6.0 / in_features)
        return np.random.uniform(-limit, limit, (in_features, out_features))