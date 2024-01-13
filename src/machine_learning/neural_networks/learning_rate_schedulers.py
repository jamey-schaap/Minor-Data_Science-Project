from abc import ABC, abstractmethod


class LearningRateScheduler(ABC):
    """LearningRateScheduler base class"""
    @abstractmethod
    def __call__(self, epoch: int) -> float:
        """
        The implementation of when the instance gets called (like a function).
        :param epoch: int, Current epoch (number).
        :return: float, The new learning rate.
        """
        pass

    def __str__(self) -> str:
        fields = vars(self).items()
        return f"#{type(self).__name__}-" + "-".join([f"{k}_{v}" for k, v in fields]) + "#"


class SquareRootScheduler(LearningRateScheduler):
    def __init__(self, lr: float = 0.01) -> None:
        """
        :param lr: float, Initial learning rate.
        """
        self.lr = lr

    def __call__(self, epoch: int) -> float:
        """
        :param epoch: int, Current epoch (number).
        :return: float, The new learning rate.
        """
        return self.lr * pow(epoch + 1.0, -0.5)


class FactorScheduler(LearningRateScheduler):
    def __init__(self, factor: int | float, stop_factor: int | float, base_lr: float):
        """
        :param factor: int | float, The factor which the learning rate changes by each epoch.
        :param stop_factor: int | float, The min/max learning rate value. The scheduler won't go lower/higher than this.
        :param base_lr: float, The initial learning rate.
        """
        self.factor = factor
        self.stop_factor = stop_factor
        self.base_lr = base_lr

    def __call__(self, epoch: int) -> float:
        """
        :param epoch: int, Current epoch (number).
        :return: float, The new learning rate.
        """
        self.base_lr = max(self.stop_factor, self.base_lr * self.factor)
        return self.base_lr
