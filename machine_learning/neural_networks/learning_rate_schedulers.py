class SquareRootScheduler:
    def __init__(self, lr: int=0.01):
        self.lr = lr

    def __call__(self, epoch: int):
        return self.lr * pow(epoch + 1.0, -0.5)

class FactorScheduler:
    def __init__(self, factor: int, stop_factor: int, base_lr: int):
        self.factor = factor
        self.stop_factor = stop_factor
        self.base_lr = base_lr

    def __call__(self, epoch: int):
        self.base_lr = max(self.stop_factor, self.base_lr * self.factor)
        return self.base_lr
