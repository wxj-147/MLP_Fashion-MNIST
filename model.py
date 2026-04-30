import numpy as np
from typing import List, Tuple, Optional, Callable

class Layer:
    """神经网络层的基类"""
    def __init__(self):
        self.params = {}
        self.grads = {}
        self.cache = {}  # 用于存储前向传播的中间结果，供反向传播使用
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        
    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        
    def update(self, lr: float, weight_decay: float = 0.0):
        """更新层参数（在优化器中调用）"""
        for param_name in self.params:
            # 应用L2正则化（权重衰减）
            self.params[param_name] -= lr * (self.grads[param_name] + weight_decay * self.params[param_name])

class Linear(Layer):
    """全连接层"""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        # 使用Xavier初始化
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        self.params['W'] = np.random.randn(input_dim, output_dim) * scale
        self.params['b'] = np.zeros((1, output_dim))
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播: y = xW + b
        
        参数:
            x: 输入，形状 (batch_size, input_dim)
        
        返回:
            输出，形状 (batch_size, output_dim)
        """
        self.cache['x'] = x
        out = np.dot(x, self.params['W']) + self.params['b']
        return out
        
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        反向传播
        
        参数:
            dout: 上游梯度，形状 (batch_size, output_dim)
        
        返回:
            dx: 对输入的梯度，形状 (batch_size, input_dim)
        """
        x = self.cache['x']
        batch_size = x.shape[0]
        
        # 计算梯度
        self.grads['W'] = np.dot(x.T, dout) / batch_size
        self.grads['b'] = np.sum(dout, axis=0, keepdims=True) / batch_size
        
        # 计算对输入的梯度
        dx = np.dot(dout, self.params['W'].T)
        return dx

class Activation(Layer):
    """激活函数层"""
    def __init__(self, activation: str = 'relu'):
        super().__init__()
        self.activation = activation
        self.activation_fn, self.activation_grad_fn = self._get_activation(activation)
        
    def _get_activation(self, activation: str) -> Tuple[Callable, Callable]:
        """获取激活函数及其梯度函数"""
        if activation == 'relu':
            return self._relu, self._relu_grad
        elif activation == 'sigmoid':
            return self._sigmoid, self._sigmoid_grad
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    def _relu_grad(self, x: np.ndarray) -> np.ndarray:
        """ReLU激活函数的梯度"""
        return (x > 0).astype(float)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid激活函数"""
        # 数值稳定的实现
        x_clipped = np.clip(x, -50, 50)  # 防止溢出
        return 1 / (1 + np.exp(-x_clipped))
    
    def _sigmoid_grad(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid激活函数的梯度"""
        sigmoid_x = self._sigmoid(x)
        return sigmoid_x * (1 - sigmoid_x)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        self.cache['x'] = x
        return self.activation_fn(x)
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """反向传播"""
        x = self.cache['x']
        grad = self.activation_grad_fn(x)
        return dout * grad

class SoftmaxWithCrossEntropy(Layer):
    """Softmax + 交叉熵损失组合层"""
    def forward(self, x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        前向传播
        
        参数:
            x: 输入，形状 (batch_size, num_classes)
            y: 真实标签，形状 (batch_size,)，如果为None则不计算损失
        
        返回:
            如果y不为None，返回损失值；否则返回softmax概率
        """
        # 数值稳定的softmax
        x_max = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - x_max)
        softmax = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
        self.cache['softmax'] = softmax
        
        if y is not None:
            self.cache['y'] = y
            # 计算交叉熵损失
            batch_size = x.shape[0]
            # 获取每个样本正确类别的概率
            correct_probs = softmax[np.arange(batch_size), y]
            # 避免log(0)的情况
            correct_probs = np.clip(correct_probs, 1e-15, 1.0)
            loss = -np.sum(np.log(correct_probs)) / batch_size
            return loss
        
        return softmax
    
    def backward(self) -> np.ndarray:
        """反向传播
        
        返回:
            梯度，形状 (batch_size, num_classes)
        """
        softmax = self.cache['softmax']
        y = self.cache['y']
        batch_size = softmax.shape[0]
        
        # softmax + 交叉熵的梯度计算
        grad = softmax.copy()
        grad[np.arange(batch_size), y] -= 1
        grad /= batch_size
        
        return grad

class ThreeLayerNet:
    """三层神经网络模型"""
    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, 
                 output_dim: int = 10, activation: str = 'relu'):
        """
        初始化三层神经网络
        
        参数:
            input_dim: 输入维度 (Fashion-MNIST为784)
            hidden_dim: 隐藏层维度
            output_dim: 输出维度 (10个类别)
            activation: 激活函数类型，'relu' 或 'sigmoid'
        """
        self.layers = []
        
        # 输入层 -> 隐藏层
        self.layers.append(Linear(input_dim, hidden_dim))
        self.layers.append(Activation(activation))
        
        # 隐藏层 -> 输出层
        self.layers.append(Linear(hidden_dim, output_dim))
        
        # 损失层
        self.loss_layer = SoftmaxWithCrossEntropy()
        
        # 存储参数
        self.params = []
        for layer in self.layers:
            if hasattr(layer, 'params'):
                for param_name, param_value in layer.params.items():
                    self.params.append((layer, param_name, param_value))
    
    def forward(self, x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        前向传播
        
        参数:
            x: 输入数据，形状 (batch_size, input_dim)
            y: 真实标签，形状 (batch_size,)，如果为None则不计算损失
        
        返回:
            如果y不为None，返回损失值；否则返回预测概率
        """
        # 逐层前向传播
        for layer in self.layers:
            x = layer.forward(x)
        
        # 计算损失（如果需要）
        if y is not None:
            loss = self.loss_layer.forward(x, y)
            return loss
        else:
            return self.loss_layer.forward(x)
    
    def backward(self) -> None:
        """反向传播"""
        # 计算损失层的梯度
        dout = self.loss_layer.backward()
        
        # 逐层反向传播
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        预测
        
        参数:
            x: 输入数据，形状 (batch_size, input_dim)
        
        返回:
            预测的类别，形状 (batch_size,)
        """
        # 前向传播，不计算损失
        probs = self.forward(x)
        return np.argmax(probs, axis=1)
    
    def get_accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        计算准确率
        
        参数:
            x: 输入数据
            y: 真实标签
        
        返回:
            准确率
        """
        predictions = self.predict(x)
        accuracy = np.mean(predictions == y)
        return accuracy
    
    def get_first_layer_weights(self) -> np.ndarray:
        """
        获取第一层权重矩阵，用于可视化
        
        返回:
            权重矩阵，形状 (input_dim, hidden_dim)
        """
        first_linear_layer = self.layers[0]
        if isinstance(first_linear_layer, Linear):
            return first_linear_layer.params['W']
        else:
            raise ValueError("第一层不是Linear层")
    
# SGD优化器
class SGD:
    """随机梯度下降优化器"""
    def __init__(self, model: ThreeLayerNet, lr: float = 0.01, 
                 weight_decay: float = 0.0, momentum: float = 0.0):
        """
        初始化SGD优化器
        
        参数:
            model: 要优化的模型
            lr: 学习率
            weight_decay: L2正则化系数
            momentum: 动量系数
        """
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        
        # 动量变量
        self.velocities = []
        for layer in model.layers:
            if hasattr(layer, 'params'):
                layer_vel = {}
                for param_name, param_value in layer.params.items():
                    layer_vel[param_name] = np.zeros_like(param_value)
                self.velocities.append(layer_vel)
    
    def step(self) -> None:
        """执行一步参数更新"""
        velocity_idx = 0
        for layer in self.model.layers:
            if hasattr(layer, 'params'):
                layer_vel = self.velocities[velocity_idx]
                for param_name in layer.params:
                    # 计算动量
                    if self.momentum > 0:
                        layer_vel[param_name] = (
                            self.momentum * layer_vel[param_name] - 
                            self.lr * layer.grads[param_name]
                        )
                        update = layer_vel[param_name]
                    else:
                        update = -self.lr * layer.grads[param_name]
                    
                    # 应用权重衰减
                    if self.weight_decay > 0:
                        update -= self.lr * self.weight_decay * layer.params[param_name]
                    
                    # 更新参数
                    layer.params[param_name] += update
                velocity_idx += 1
    
    def zero_grad(self) -> None:
        """清空梯度"""
        for layer in self.model.layers:
            if hasattr(layer, 'grads'):
                for param_name in layer.grads:
                    layer.grads[param_name].fill(0)

# 学习率衰减策略
class LearningRateScheduler:
    """学习率调度器"""
    def __init__(self, optimizer: SGD, decay_type: str = 'step', 
                 decay_rate: float = 0.1, decay_steps: int = 10):
        """
        初始化学习率调度器
        
        参数:
            optimizer: 优化器
            decay_type: 衰减类型，'step'或'exponential'
            decay_rate: 衰减率
            decay_steps: 每多少步衰减一次（仅对'step'类型有效）
        """
        self.optimizer = optimizer
        self.decay_type = decay_type
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.step_count = 0
        self.base_lr = optimizer.lr
    
    def step(self) -> None:
        """更新学习率"""
        self.step_count += 1
        
        if self.decay_type == 'step':
            # 阶梯衰减
            if self.step_count % self.decay_steps == 0:
                self.optimizer.lr *= self.decay_rate
        elif self.decay_type == 'exponential':
            # 指数衰减
            self.optimizer.lr = self.base_lr * (self.decay_rate ** (self.step_count / self.decay_steps))
        else:
            raise ValueError(f"不支持的衰减类型: {self.decay_type}")

'''
# 测试
if __name__ == '__main__':
    # 设置随机种子以确保可复现性
    np.random.seed(42)
    
    # 创建模型（使用ReLU激活函数，隐藏层维度256）
    model = ThreeLayerNet(input_dim=784, hidden_dim=256, output_dim=10, activation='relu')
    
    # 创建优化器
    optimizer = SGD(model, lr=0.01, weight_decay=0.0001, momentum=0.9)
    
    # 创建学习率调度器
    scheduler = LearningRateScheduler(optimizer, decay_type='step', decay_rate=0.5, decay_steps=20)
    
    print("模型结构:")
    print(f"  输入层: 784个神经元")
    print(f"  隐藏层: 256个神经元，使用ReLU激活函数")
    print(f"  输出层: 10个神经元")
    print(f"  总参数数: 大约{sum(p[2].size for p in model.params):,}个")
    
    # 为了演示，这里使用随机数据
    batch_size = 32
    x_demo = np.random.randn(batch_size, 784)
    y_demo = np.random.randint(0, 10, batch_size)
    
    # 前向传播演示
    loss = model.forward(x_demo, y_demo)
    print(f"\n演示 - 前向传播损失: {loss:.4f}")
    
    # 反向传播演示
    model.backward()
    
    # 预测演示
    predictions = model.predict(x_demo)
    accuracy = np.mean(predictions == y_demo)
    print(f"演示 - 随机数据的准确率: {accuracy:.2%}")
    
    # 获取第一层权重用于可视化
    first_layer_weights = model.get_first_layer_weights()
    print(f"第一层权重形状: {first_layer_weights.shape}")
    
    # 测试Sigmoid激活函数
    print("\n" + "="*50)
    print("测试Sigmoid激活函数版本:")
    model_sigmoid = ThreeLayerNet(input_dim=784, hidden_dim=256, output_dim=10, activation='sigmoid')
    loss_sigmoid = model_sigmoid.forward(x_demo, y_demo)
    print(f"Sigmoid模型 - 前向传播损失: {loss_sigmoid:.4f}")
'''