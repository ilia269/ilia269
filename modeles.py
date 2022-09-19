from tensor import Tensor, Parameter
from math import sqrt
import numpy as np
from hooks import SigmoidBackwardHook, ReLUBackwardHook, SinBackwardHook


class Module:
    def parameters(self):
        return []

    def forward(self, *args, **kwargs):
        return None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(np.ones(shape=(out_features, in_features)))

        if bias:
            self.bias = Parameter(np.ones(shape=(out_features, 1)))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data = np.matrix(np.random.uniform(low=-1 / sqrt(self.in_features),
                                                       high=1 / sqrt(self.in_features),
                                                       size=(self.out_features, self.in_features)))

        if self.bias is not None:
            self.bias.data = np.matrix(np.random.uniform(low=-1 / sqrt(self.in_features),
                                                         high=1 / sqrt(self.in_features),
                                                         size=self.out_features))

    def parameters(self):
        # if self.bias: return [self.weight, self.bias]
        # return [self.weight]

        return [self.weight, self.bias]

    def forward(self, x: Tensor):
        res = x @ self.weight.transpose()
        if self.bias is not None: res += self.bias
        return res


class Power(Module):
    def __init__(self, in_features, power_degree):
        super().__init__()
        self.in_features = in_features
        self.power_degree = power_degree
        self.weight = Parameter(np.ones(in_features))

    def reset_parameter(self):
        self.weight.data = np.matrix(np.random.uniform(low=-1 / sqrt(self.in_features),
                                                       high=1 / sqrt(self.in_features),
                                                       size=self.in_features))

    def forward(self, x: Tensor):
        return self.weight * x**self.power_degree

    def parameters(self):
        return [self.weight]


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        res = Tensor(data=self.activation(x))
        backward_hook = SigmoidBackwardHook(tensors=[x, res])
        res.backward_hook = backward_hook
        return res

    @staticmethod
    def activation(x):
        one = np.ones(shape=x.data.shape)
        return one / (one + np.exp((-1.0) * x.data))


class Sin(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        res = Tensor(data=self.activation(x))
        backward_hook = SinBackwardHook(tensors=[x, res])
        res.backward_hook = backward_hook
        return res

    @staticmethod
    def activation(x):
        return np.sin(x.data)


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        new_data = np.array(x.data)
        self.apply(self.activation, new_data)
        res = Tensor(data=new_data)
        backward_hook = ReLUBackwardHook(tensors=[x, res])
        res.backward_hook = backward_hook
        return res

    @staticmethod
    def activation(x):
        return max(0, x)

    @staticmethod
    def apply(function, array):
        h, w = array.shape
        for i in range(h):
            for j in range(w):
                array[i][j] = function(array[i][j])


class LeakyReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        res = Tensor(data=[self.activation(x) for x in x.data])
        return res

    @staticmethod
    def activation(x, negative_slope=0.01):
        res = max(0, x) + min(0, x) * negative_slope
        return res


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def parameters(self):
        res = []

        for m in self.modules:
            res += m.parameters()

        return res

    def forward(self, x: Tensor):
        x_in, x_out = x, None

        for module in self.modules:
            x_out = module(x_in)
            x_in = x_out

        return x_out

    def predict(self, x, trash_hold=0.5):
        res = float(self.forward(x).data)

        if res >= trash_hold: return 1

        return 0
