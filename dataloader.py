import numpy as np
import os
import gzip
import urllib.request
from typing import Tuple

class FashionMNISTLoader:
    """
    自动下载、加载和预处理Fashion-MNIST数据集
    数据下载并保存到 data_dir
    """
    
    # 数据集地址
    base_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    files = [
        'train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz'
    ]
    
    # 类别名称
    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

    def __init__(self, data_dir: str = './data'):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def _download_and_save(self, url: str, filename: str) -> str:
        """下载文件并返回本地路径"""
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            print(f'Downloading {filename}...')
            try:
                urllib.request.urlretrieve(url, filepath)
                print('Done!')
            except Exception as e:
                print(f'Download failed: {e}')
                raise
        return filepath

    def _load_images(self, filepath: str) -> np.ndarray:
        """从idx格式的gz文件中加载图像"""
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
        # 重塑为 (样本数, 通道, 高度, 宽度)，通道为1
        return data.reshape(-1, 1, 28, 28).astype(np.float32)

    def _load_labels(self, filepath: str) -> np.ndarray:
        """从idx格式的gz文件中加载标签"""
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        return data.astype(np.int64)  

    def load_data(self, validation_ratio: float = 0.1) -> Tuple[Tuple[np.ndarray, np.ndarray], 
                                                                 Tuple[np.ndarray, np.ndarray], 
                                                                 Tuple[np.ndarray, np.ndarray]]:
        """
        加载数据，并进行预处理和训练/验证集拆分。
        
        参数:
            validation_ratio: 从训练集中划分出验证集的比例。
        
        返回:
            (x_train, y_train), (x_val, y_val), (x_test, y_test)
            图像数据范围[0, 1]， 并进行了扁平化处理，形状为 (样本数, 784)。
        """
        # 1. 下载或定位文件
        paths = []
        for f in self.files:
            url = self.base_url + f
            paths.append(self._download_and_save(url, f))
        
        # 2. 加载原始数据
        x_train = self._load_images(paths[0])  # 训练图像
        y_train = self._load_labels(paths[1])  # 训练标签
        x_test = self._load_images(paths[2])   # 测试图像
        y_test = self._load_labels(paths[3])   # 测试标签
        
        # 3. 数据预处理
        # 3.1 归一化
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        
        # 3.2 将图像数据从 (N, 1, 28, 28) 扁平化为 (N, 784)
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
        
        # 4. 从训练集中拆分验证集
        num_val = int(x_train.shape[0] * validation_ratio)
        indices = np.random.permutation(x_train.shape[0])
        
        x_val = x_train[indices[:num_val]]
        y_val = y_train[indices[:num_val]]
        x_train = x_train[indices[num_val:]]
        y_train = y_train[indices[num_val:]]
        
        print(f"Training set:   {x_train.shape} {y_train.shape}")
        print(f"Validation set: {x_val.shape} {y_val.shape}")
        print(f"Test set:       {x_test.shape} {y_test.shape}")
        
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)
    
'''
# 数据集测试
if __name__ == '__main__':
    loader = FashionMNISTLoader(data_dir='./data/fashion_mnist')
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = loader.load_data(validation_ratio=0.1)
    
    print(f"一个训练样本的维度: {x_train[0].shape}")  # (784,)
    print(f"对应的标签: {y_train[0]} -> {loader.class_names[y_train[0]]}")
'''