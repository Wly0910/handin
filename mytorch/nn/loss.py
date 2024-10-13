import numpy as np

class Linear:
    def __init__(self, input_dim, output_dim, enable_debug=False):
        """
        初始化线性层。
        
        参数:
        input_dim (int): 输入特征的维度
        output_dim (int): 输出特征的维度
        enable_debug (bool): 是否启用调试模式
        """
        self.weights = np.zeros((output_dim, input_dim))
        self.bias = np.zeros((output_dim, 1))
        
        self.enable_debug = enable_debug
        self.input_cache = None
        self.batch_size = None
        self.ones_vector = None

    def forward(self, input_data):
        """
        执行前向传播。
        
        参数:
        input_data (np.array): 输入数据，形状为 (batch_size, input_dim)
        
        返回:
        np.array: 输出数据，形状为 (batch_size, output_dim)
        """
        self.input_cache = input_data
        self.batch_size = input_data.shape[0]
        self.ones_vector = np.ones((self.batch_size, 1))
        
        output = np.dot(input_data, self.weights.T) + np.dot(self.ones_vector, self.bias.T)
        return output

    def backward(self, output_gradient):
        """
        执行反向传播。
        
        参数:
        output_gradient (np.array): 输出梯度，形状为 (batch_size, output_dim)
        
        返回:
        np.array: 输入梯度，形状为 (batch_size, input_dim)
        """
        input_gradient = np.dot(output_gradient, self.weights)
        self.weight_gradient = np.dot(output_gradient.T, self.input_cache)
        self.bias_gradient = np.dot(output_gradient.T, self.ones_vector)

        if self.enable_debug:
            self.debug_input_gradient = input_gradient

        return input_gradient

    def get_parameters(self):
        """
        获取层的参数。
        
        返回:
        tuple: (weights, bias)
        """
        return self.weights, self.bias

    def set_parameters(self, weights, bias):
        """
        设置层的参数。
        
        参数:
        weights (np.array): 权重矩阵
        bias (np.array): 偏置向量
        """
        self.weights = weights
        self.bias = bias

    def get_gradients(self):
        """
        获取层的梯度。
        
        返回:
        tuple: (weight_gradient, bias_gradient)
        """
        return self.weight_gradient, self.bias_gradient