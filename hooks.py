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

    def backward(self, *args):
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

    def update_gradient(self, delta):
        #ToDo получить дельты и отправить их на левый тензор
        pass

    def get_delta(self):
        pass

    def backward(self, grad):
        self.tensors[0].backward(grad)
        self.tensors[1].backward(grad)


class MatrixMultBackwardHook(BackwardHook):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.name = "Matrix mul"

    def update_gradient(self, delta):
        #ToDo
        pass

    def get_delta(self):
        pass

    def backward(self, grad):
        t1, t2 = self.tensors[0], self.tensors[1]
        grad_1 = grad @ t2.transpose()
        grad_2 = t1.transpose() @ grad

        self.tensors[0].backward(grad_1)
        self.tensors[1].backward(grad_2)


class TensorMultBackwardHook(BackwardHook):
    def update_gradient(self, delta):
        #ToDo
        pass

    def get_delta(self):
        pass

    def backward(self, grad):
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

    def backward(self, grad):
        self.tensors[0].backward(grad.transpose())


class SigmoidBackwardHook(BackwardHook):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.name = "Sigmoid"

    def backward(self, grad):
        t = self.tensors[0]

        back_grad = grad * self.derivative(t)

        self.tensors[0].backward(back_grad)

    @staticmethod
    def derivative(x_tensor):
        x = x_tensor.data
        # print(x_tensor.data)
        for i in x:
            i = (1 - i)*i

        return x_tensor

    def update_gradient(self, delta):
        #ToDo
        pass

    def get_delta(self):
        pass


class ReLUBackwardHook(BackwardHook):
    def update_gradient(self, delta):
        #ToDo
        pass

    def get_delta(self):
        pass


class LeakyReLUBackwardHook(BackwardHook):
    def update_gradient(self, delta):
        #ToDo
        pass

    def get_delta(self):
        pass
