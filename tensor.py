from hooks import *
import copy


class Tensor:
    def __init__(self, data: np.array, requires_grad=False):
        self.data = np.matrix(data)
        self.backward_hook = None
        self.requires_grad = requires_grad
        self.grad = None

    def __add__(self, other):
        if np.shape(self.data) != np.shape(other.data):
            raise ValueError("Тензоры должны быть одинаковой размерности")

        new_data = self.data+other.data
        backward_hook = AddBackwardHook(tensors=[self, other])

        res = Tensor(data=new_data)
        res.backward_hook = backward_hook

        return res

    def __mul__(self, other):
        if isinstance(other, float):
            new_data = np.matrix(np.array(self.data) * other)
            backward_hook = TensorMultBackwardHook(tensors=[self, other])

            res = Tensor(data=new_data)
            res.backward_hook = backward_hook
            return res

        if np.shape(self.data) != np.shape(other.data):
            raise ValueError("Тензоры должны быть одинаковой размерности")

        new_data = np.matrix(np.array(self.data)*np.array(other.data))
        backward_hook = TensorMultBackwardHook(tensors=[self, other])

        res = Tensor(data=new_data)
        res.backward_hook = backward_hook

        return res

    def __pow__(self, pow_degree):
        res = self
        for _ in range(pow_degree-1):
            res *= self

        return res

    def __matmul__(self, other):
        new_data = self.data @ other.data
        backward_hook = MatrixMultBackwardHook(tensors=[self, other])

        res = Tensor(data=new_data)

        res.backward_hook = backward_hook

        return res

    def transpose(self):
        new_data = copy.copy(self.data).transpose()

        backward_hook = TransposeBackwardHook(tensors=[self])
        res = Tensor(data=new_data)

        res.backward_hook = backward_hook

        return res

    def clamp(self, min_, max_):
        if min_ > max_:
            raise ValueError("min должен быть меньше max")

        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                element = self.data[i][j]
                res = None

                if min_ <= element <= max_:
                    res = element
                if element < min_:
                    res = min_
                if element > max_:
                    res = max_

                self.data[i][j] = res

        return self

    def backward(self, grad=None):

        if grad is None:
            grad = Tensor(data=np.eye(self.data.shape[0], self.data.shape[1]))

        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad

        if self.backward_hook is not None:
            self.backward_hook.update_gradient(grad)

    def __repr__(self):
        return str(self.data)

    def zero_grad(self):
        if not hasattr(self, "grad"):
            return None

        self.grad *= 0.0


class Parameter(Tensor):
    def __init__(self, data: np.array, requires_grad=True):
        super().__init__(data, requires_grad)
        self.grad = None
