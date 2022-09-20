import numpy as np


class BackwardHook:
    def __init__(self, tensors, params=None):
        self.tensors = tensors
        self.params = params
        self.name = None

    def update_gradient(self, delta):
        pass

    def get_delta(self):
        pass

    def __repr__(self):
        res_str = f"operation is {self.name}\n"
        for t in self.tensors:
            res_str += str(t)+"\n\n"

        return res_str


class AddBackwardHook(BackwardHook):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.name = "Add"

    def get_delta(self):
        pass

    def update_gradient(self, grad):
        self.tensors[0].backward(grad)
        self.tensors[1].backward(grad)


class MatrixMultBackwardHook(BackwardHook):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.name = "Matrix mul"

    def get_delta(self):
        pass

    def update_gradient(self, grad):
        t1, t2 = self.tensors[0], self.tensors[1]
        grad_1 = grad @ t2.transpose()
        grad_2 = t1.transpose() @ grad

        self.tensors[0].backward(grad_1)
        self.tensors[1].backward(grad_2)


class TensorMultBackwardHook(BackwardHook):
    def get_delta(self):
        pass

    def update_gradient(self, grad):
        t1, t2 = self.tensors[0], self.tensors[1]

        grad_1 = grad * t2
        grad_2 = grad * t1

        self.tensors[0].backward(grad_1)
        self.tensors[1].backward(grad_2)


class ClampBackwardHook(BackwardHook):
    def update_gradient(self, delta):
        #ToDo
        pass

    def get_delta(self):
        pass


class TransposeBackwardHook(BackwardHook):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.name = "Transpose"

    def update_gradient(self, grad):
        self.tensors[0].backward(grad.transpose())


class SigmoidBackwardHook(BackwardHook):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.name = "Sigmoid"

    def update_gradient(self, grad):
        input_t, output_t = self.tensors[0], self.tensors[1]
        back_grad = grad * self.derivative(output_t)

        input_t.backward(back_grad)

    @staticmethod
    def sigmoid(x):
        one = np.ones(shape=x.shape)
        return np.array(one / (one + np.exp((-1.0)*x)))

    def derivative(self, x_tensor):
        one = np.ones(shape=x_tensor.data.shape)
        new_data = (one - self.sigmoid(x_tensor.data))*self.sigmoid(x_tensor.data)
        x_tensor.data = np.matrix(new_data)

        t = np.array(x_tensor.data)
        new_data1 = np.exp((-1.0)*t**2) / (one + np.exp((-1.0)*t**2))**2
        x_tensor.data = np.matrix(new_data1)
        return x_tensor

    def get_delta(self):
        pass


class SinBackwardHook(BackwardHook):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.name = "Sin"

    def update_gradient(self, grad):
        input_t, output_t = self.tensors[0], self.tensors[1]
        back_grad = grad * self.derivative(output_t)

        input_t.backward(back_grad)

    def derivative(self, x_tensor):
        new_data = np.cos(x_tensor.data)
        x_tensor.data = np.matrix(new_data)

        return x_tensor

    def get_delta(self):
        pass


class ReLUBackwardHook(BackwardHook):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.name = "ReLU"

    def update_gradient(self, grad):
        input_t = self.tensors[0]
        back_grad = grad * self.derivative(input_t)

        input_t.backward(back_grad)

    def derivative(self, x_tensor):
        data = np.array(x_tensor.data)
        self.apply(self.relu_derivative, data)
        x_tensor.data = np.matrix(data)
        return x_tensor

    @staticmethod
    def apply(function, array):
        h, w = array.shape
        for i in range(h):
            for j in range(w):
                array[i][j] = function(array[i][j])

    @staticmethod
    def relu_derivative(t):
        if t >= 0:
            return 1.0
        return 0.0

    def get_delta(self):
        pass


class LeakyReLUBackwardHook(BackwardHook):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.negative_slope = None
        self.name = "LeakyReLU"

    def update_gradient(self, grad):
        input_t = self.tensors[0]
        back_grad = grad * self.derivative(input_t)

        input_t.backward(back_grad)

    def derivative(self, x_tensor):
        data = np.array(x_tensor.data)
        self.apply(self.relu_derivative, data)
        x_tensor.data = np.matrix(data)
        return x_tensor

    @staticmethod
    def apply(function, array):
        h, w = array.shape
        for i in range(h):
            for j in range(w):
                array[i][j] = function(array[i][j])

    def relu_derivative(self, t):
        if t >= 0:
            return 1.0
        return self.negative_slope

    def get_delta(self):
        pass
