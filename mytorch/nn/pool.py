import numpy as np
from resampling import Downsample2d

class MaxPool2d_stride1:
    def __init__(self, filter_size):
        self.filter_size = filter_size
        self.cache = {}

    def forward(self, input_tensor):
        self.cache['input'] = input_tensor
        n, c, h, w = input_tensor.shape
        out_h = h - self.filter_size + 1
        out_w = w - self.filter_size + 1

        output = np.zeros((n, c, out_h, out_w))
        self.cache['max_pos'] = np.zeros((n, c, out_h, out_w, 2), dtype=int)

        for i in range(out_h):
            for j in range(out_w):
                window = input_tensor[:, :, i:i+self.filter_size, j:j+self.filter_size]
                output[:, :, i, j] = np.max(window, axis=(2, 3))
                max_pos = np.argmax(window.reshape(n, c, -1), axis=2)
                self.cache['max_pos'][:, :, i, j] = np.array(np.unravel_index(max_pos, (self.filter_size, self.filter_size))).T

        return output

    def backward(self, grad_output):
        input_tensor = self.cache['input']
        grad_input = np.zeros_like(input_tensor)
        n, c, out_h, out_w = grad_output.shape

        for i in range(out_h):
            for j in range(out_w):
                max_pos = self.cache['max_pos'][:, :, i, j]
                np.add.at(grad_input, (np.arange(n)[:, None], np.arange(c)[None, :], i + max_pos[:, :, 0], j + max_pos[:, :, 1]), grad_output[:, :, i, j])

        return grad_input

class MeanPool2d_stride1:
    def __init__(self, filter_size):
        self.filter_size = filter_size

    def forward(self, input_tensor):
        n, c, h, w = input_tensor.shape
        out_h = h - self.filter_size + 1
        out_w = w - self.filter_size + 1

        output = np.zeros((n, c, out_h, out_w))
        for i in range(out_h):
            for j in range(out_w):
                output[:, :, i, j] = np.mean(input_tensor[:, :, i:i+self.filter_size, j:j+self.filter_size], axis=(2, 3))

        return output

    def backward(self, grad_output):
        n, c, out_h, out_w = grad_output.shape
        grad_input = np.zeros((n, c, out_h + self.filter_size - 1, out_w + self.filter_size - 1))

        for i in range(out_h):
            for j in range(out_w):
                grad_input[:, :, i:i+self.filter_size, j:j+self.filter_size] += grad_output[:, :, i, j][:, :, None, None] / (self.filter_size ** 2)

        return grad_input

class MaxPool2d:
    def __init__(self, filter_size, stride):
        self.filter_size = filter_size
        self.stride = stride
        self.maxpool2d_stride1 = MaxPool2d_stride1(filter_size)
        self.downsample2d = Downsample2d(stride)

    def forward(self, input_tensor):
        intermediate = self.maxpool2d_stride1.forward(input_tensor)
        return self.downsample2d.forward(intermediate)

    def backward(self, grad_output):
        grad_upsampled = self.downsample2d.backward(grad_output)
        return self.maxpool2d_stride1.backward(grad_upsampled)

class MeanPool2d:
    def __init__(self, filter_size, stride):
        self.filter_size = filter_size
        self.stride = stride
        self.meanpool2d_stride1 = MeanPool2d_stride1(filter_size)
        self.downsample2d = Downsample2d(stride)

    def forward(self, input_tensor):
        intermediate = self.meanpool2d_stride1.forward(input_tensor)
        return self.downsample2d.forward(intermediate)

    def backward(self, grad_output):
        grad_upsampled = self.downsample2d.backward(grad_output)
        return self.meanpool2d_stride1.backward(grad_upsampled)