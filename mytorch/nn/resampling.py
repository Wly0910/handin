import numpy as np

class Upsample1d:
    def __init__(self, scale):
        """
        初始化1D上采样操作。
        
        参数:
            scale (int): 上采样的比例因子。
        """
        self.scale = scale

    def forward(self, input_data):
        """
        执行1D上采样的前向传播。
        
        参数:
            input_data (np.array): 输入数组，形状为(batch_size, channels, input_length)。
        
        返回:
            np.array: 上采样后的数组，形状为(batch_size, channels, output_length)。
        """
        batch_size, channels, input_length = input_data.shape
        output_length = self.scale * (input_length - 1) + 1
        
        output = np.zeros((batch_size, channels, output_length))
        output[:, :, ::self.scale] = input_data
    
        return output

    def backward(self, grad_output):
        """
        执行1D上采样的反向传播。
        
        参数:
            grad_output (np.array): 关于上采样输出的梯度，形状为(batch_size, channels, output_length)。
        
        返回:
            np.array: 关于上采样输入的梯度，形状为(batch_size, channels, input_length)。
        """
        return grad_output[:, :, ::self.scale]

class Downsample1d:
    def __init__(self, scale):
        """
        初始化1D下采样操作。
        
        参数:
            scale (int): 下采样的比例因子。
        """
        self.scale = scale
        self.input_length = None

    def forward(self, input_data):
        """
        执行1D下采样的前向传播。
        
        参数:
            input_data (np.array): 输入数组，形状为(batch_size, channels, input_length)。
        
        返回:
            np.array: 下采样后的数组，形状为(batch_size, channels, output_length)。
        """
        self.input_length = input_data.shape[2]
        return input_data[:, :, ::self.scale]

    def backward(self, grad_output):
        """
        执行1D下采样的反向传播。
        
        参数:
            grad_output (np.array): 关于下采样输出的梯度，形状为(batch_size, channels, output_length)。
        
        返回:
            np.array: 关于下采样输入的梯度，形状为(batch_size, channels, input_length)。
        """
        batch_size, channels, _ = grad_output.shape
        grad_input = np.zeros((batch_size, channels, self.input_length))
        grad_input[:, :, ::self.scale] = grad_output
        return grad_input

class Upsample2d:
    def __init__(self, scale):
        """
        初始化2D上采样操作。
        
        参数:
            scale (int): 上采样的比例因子。
        """
        self.scale = scale

    def forward(self, input_data):
        """
        执行2D上采样的前向传播。
        
        参数:
            input_data (np.array): 输入数组，形状为(batch_size, channels, height, width)。
        
        返回:
            np.array: 上采样后的数组，形状为(batch_size, channels, new_height, new_width)。
        """
        batch_size, channels, height, width = input_data.shape
        new_height = (height - 1) * self.scale + 1
        new_width = (width - 1) * self.scale + 1

        output = np.zeros((batch_size, channels, new_height, new_width), dtype=input_data.dtype)
        output[:, :, ::self.scale, ::self.scale] = input_data

        return output

    def backward(self, grad_output):
        """
        执行2D上采样的反向传播。
        
        参数:
            grad_output (np.array): 关于上采样输出的梯度，形状为(batch_size, channels, new_height, new_width)。
        
        返回:
            np.array: 关于上采样输入的梯度，形状为(batch_size, channels, height, width)。
        """
        return grad_output[:, :, ::self.scale, ::self.scale]

class Downsample2d:
    def __init__(self, scale):
        """
        初始化2D下采样操作。
        
        参数:
            scale (int): 下采样的比例因子。
        """
        self.scale = scale
        self.input_shape = None

    def forward(self, input_data):
        """
        执行2D下采样的前向传播。
        
        参数:
            input_data (np.array): 输入数组，形状为(batch_size, channels, height, width)。
        
        返回:
            np.array: 下采样后的数组，形状为(batch_size, channels, new_height, new_width)。
        """
        self.input_shape = input_data.shape
        return input_data[:, :, ::self.scale, ::self.scale]

    def backward(self, grad_output):
        """
        执行2D下采样的反向传播。
        
        参数:
            grad_output (np.array): 关于下采样输出的梯度，形状为(batch_size, channels, new_height, new_width)。
        
        返回:
            np.array: 关于下采样输入的梯度，形状为(batch_size, channels, height, width)。
        """
        grad_input = np.zeros(self.input_shape, dtype=grad_output.dtype)
        grad_input[:, :, ::self.scale, ::self.scale] = grad_output
        return grad_input