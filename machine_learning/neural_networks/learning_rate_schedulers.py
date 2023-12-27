from abc import ABC, abstractmethod


class LearningRateScheduler(ABC):
    @abstractmethod
    def __call__(self, epoch: int) -> float:
        pass

    def __str__(self) -> str:
        fields = vars(self).items()
        return f"#{type(self).__name__}-" + "-".join([f"{k}_{v}" for k, v in fields]) + "#"


class SquareRootScheduler(LearningRateScheduler):
    def __init__(self, lr: int=0.01):
        self.lr = lr

    def __call__(self, epoch: int) -> float:
        return self.lr * pow(epoch + 1.0, -0.5)


class FactorScheduler(LearningRateScheduler):
    def __init__(self, factor: int, stop_factor: int, base_lr: int):
        self.factor = factor
        self.stop_factor = stop_factor
        self.base_lr = base_lr

    def __call__(self, epoch: int) -> float:
        self.base_lr = max(self.stop_factor, self.base_lr * self.factor)
        return self.base_lr
