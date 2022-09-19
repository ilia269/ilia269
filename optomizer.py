import numpy as np


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
        for parameter in self.parameters:
            grad = np.array(parameter.grad.data)

            v_next_data = self.betas[0] * parameter.v_prev + (1.0 - self.betas[0]) * grad
            g_next_data = self.betas[1] * parameter.g_prev + (1.0 - self.betas[1]) * grad**2

            parameter.v_prev = v_next_data
            parameter.g_prev = g_next_data

            parameter.data -= self.lr * v_next_data / (g_next_data + self.eps)**0.5
