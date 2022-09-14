class Optimizer:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def zero_grads(self):
        for parameter in self.parameters:
            parameter.zero_grad()

    def step(self):
        pass


class GD(Optimizer):
    def __init__(self, parameters, lr=1e-4):
        super().__init__(parameters, lr)

    def step(self):
        for parameter in self.parameters:
            parameter.data += parameter.grad.data * (-self.lr)


class SGD(Optimizer):
    def __init__(self, parameters, lr=1e-4):
        super().__init__(parameters, lr)

    def step(self):
        pass


class Adam(Optimizer):
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-08):
        super().__init__(parameters, lr)
        self.betas = betas
        self.eps = eps

    def step(self):
        # ToDo
        pass
