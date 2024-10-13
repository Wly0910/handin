import numpy as np

class Flatten():

    def forward(self, input_tensor):
        """
        Argument:
            input_tensor (np.array): (batch_size, in_channels, in_width)
        Return:
            flattened_output (np.array): (batch_size, in_channels * in width)
        """
        self.input_dimensions = input_tensor.shape  # 存储输入维度用于反向传播
        batch_size = input_tensor.shape[0]
        flattened_size = np.prod(input_tensor.shape[1:])
        flattened_output = input_tensor.reshape(batch_size, flattened_size)

        return flattened_output

    def backward(self, gradient):
        """
        Argument:
            gradient (np.array): (batch size, in channels * in width)
        Return:
            reshaped_gradient (np.array): (batch size, in channels, in width)
        """
        
        reshaped_gradient = np.reshape(gradient, self.input_dimensions)

        return reshaped_gradient