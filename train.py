import numpy as np
import os
import pickle
import time
from dataloader import FashionMNISTLoader
from model import ThreeLayerNet, SGD, LearningRateScheduler
from typing import Tuple, Dict, List, Optional

class Trainer:
    """训练器类，负责管理整个训练过程"""
    
    def __init__(self, model: ThreeLayerNet, optimizer: SGD, 
                 scheduler: Optional[LearningRateScheduler] = None):
        """
        初始化训练器
        
        参数:
            model: 要训练的模型
            optimizer: 优化器
            scheduler: 学习率调度器（可选）
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
    def train_epoch(self, x_train: np.ndarray, y_train: np.ndarray, 
                   batch_size: int = 32) -> Tuple[float, float]:
        """
        训练一个epoch
        
        返回:
            (平均训练损失, 训练准确率)
        """
        num_samples = x_train.shape[0]
        num_batches = num_samples // batch_size
        total_loss = 0.0
        
        # 打乱数据
        indices = np.random.permutation(num_samples)
        x_shuffled = x_train[indices]
        y_shuffled = y_train[indices]
        
        # 用于计算训练准确率
        correct_predictions = 0
        total_predictions = 0
        
        for i in range(num_batches):
            # 获取小批量数据
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            x_batch = x_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            
            # 前向传播
            loss = self.model.forward(x_batch, y_batch)
            total_loss += loss
            
            # 计算训练准确率（使用当前批次的预测）
            predictions = self.model.predict(x_batch)
            batch_correct = np.sum(predictions == y_batch)
            correct_predictions += batch_correct
            total_predictions += len(y_batch)
            
            # 反向传播
            self.model.backward()
            
            # 参数更新
            self.optimizer.step()
            
            # 清空梯度
            self.optimizer.zero_grad()
        
        # 计算平均损失和准确率
        avg_loss = total_loss / num_batches
        train_acc = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return avg_loss, train_acc
    
    def validate(self, x_val: np.ndarray, y_val: np.ndarray, 
                 batch_size: int = 100) -> Tuple[float, float]:
        """
        在验证集上评估模型
        
        返回:
            (验证损失, 验证准确率)
        """
        num_samples = x_val.shape[0]
        num_batches = num_samples // batch_size
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            x_batch = x_val[start_idx:end_idx]
            y_batch = y_val[start_idx:end_idx]
            
            # 前向传播
            loss = self.model.forward(x_batch, y_batch)
            total_loss += loss
            
            # 预测
            predictions = self.model.predict(x_batch)
            batch_correct = np.sum(predictions == y_batch)
            correct_predictions += batch_correct
            total_predictions += len(y_batch)
        
        # 如果有剩余样本
        if num_samples % batch_size != 0:
            start_idx = num_batches * batch_size
            x_batch = x_val[start_idx:]
            y_batch = y_val[start_idx:]
            
            loss = self.model.forward(x_batch, y_batch)
            total_loss += loss
            
            predictions = self.model.predict(x_batch)
            batch_correct = np.sum(predictions == y_batch)
            correct_predictions += batch_correct
            total_predictions += len(y_batch)
        
        avg_loss = total_loss / (num_batches + (1 if num_samples % batch_size != 0 else 0))
        val_acc = correct_predictions / total_predictions
        
        return avg_loss, val_acc
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray, 
              x_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 50, batch_size: int = 32,
              save_dir: str = './checkpoints',
              save_every: int = 5) -> Dict:
        """
        完整的训练过程
        
        返回:
            训练历史记录
        """
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"开始训练，共 {epochs} 个epoch")
        print(f"训练集大小: {x_train.shape[0]}，验证集大小: {x_val.shape[0]}")
        print(f"批量大小: {batch_size}")
        print("-" * 60)
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            # 训练一个epoch
            train_loss, train_acc = self.train_epoch(x_train, y_train, batch_size)
            
            # 在验证集上评估
            val_loss, val_acc = self.validate(x_val, y_val)
            
            # 记录历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            self.train_history['learning_rates'].append(self.optimizer.lr)
            
            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 打印进度
            if epoch % 1 == 0:  # 每个epoch都打印
                print(f"Epoch {epoch:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                      f"LR: {self.optimizer.lr:.6f}")
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.save_model(os.path.join(save_dir, 'best_model.pkl'))
                print(f"  新的最佳模型！验证准确率: {val_acc:.4f}")
            
            # 定期保存检查点
            if epoch % save_every == 0 or epoch == epochs:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pkl')
                self.save_model(checkpoint_path)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print("-" * 60)
        print(f"训练完成！总时间: {training_time:.2f}秒")
        print(f"最佳验证准确率: {self.best_val_acc:.4f} (epoch {self.best_epoch})")
        
        # 保存训练历史
        history_path = os.path.join(save_dir, 'training_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(self.train_history, f)
        print(f"训练历史已保存到: {history_path}")
        
        return self.train_history
    
    def save_model(self, filepath: str) -> None:
        """保存模型和优化器状态"""
        param_values = [param_value for (_, _, param_value) in self.model.params]
        checkpoint = {
            'model_params': param_values,
            'optimizer_lr': self.optimizer.lr,
            'train_history': self.train_history,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    def load_model(self, filepath: str) -> None:
        """加载模型和优化器状态"""
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # 加载模型参数
        for (layer, param_name, _), saved_param_value in zip(self.model.params, checkpoint['model_params']):
            layer.params[param_name] = saved_param_value.copy()
        
        # 加载其他状态
        self.optimizer.lr = checkpoint['optimizer_lr']
        self.train_history = checkpoint['train_history']
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_epoch = checkpoint['best_epoch']
        
        print(f"模型已从 {filepath} 加载")
        print(f"最佳验证准确率: {self.best_val_acc:.4f} (epoch {checkpoint['best_epoch']})")

def test_model(model: ThreeLayerNet, x_test: np.ndarray, y_test: np.ndarray, 
               batch_size: int = 100) -> Tuple[float, np.ndarray]:
    """
    在测试集上评估模型，并计算混淆矩阵
    
    返回:
        (测试准确率, 混淆矩阵)
    """
    num_samples = x_test.shape[0]
    num_classes = 10
    
    # 初始化混淆矩阵
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    # 分批处理测试集
    num_batches = (num_samples + batch_size - 1) // batch_size
    correct_predictions = 0
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        x_batch = x_test[start_idx:end_idx]
        y_batch = y_test[start_idx:end_idx]
        
        # 预测
        predictions = model.predict(x_batch)
        
        # 统计正确预测
        batch_correct = np.sum(predictions == y_batch)
        correct_predictions += batch_correct
        
        # 更新混淆矩阵
        for true_label, pred_label in zip(y_batch, predictions):
            confusion_matrix[true_label, pred_label] += 1
    
    # 计算准确率
    accuracy = correct_predictions / num_samples
    
    return accuracy, confusion_matrix

def print_confusion_matrix(confusion_matrix: np.ndarray, class_names: List[str]) -> None:
    """打印格式化的混淆矩阵"""
    num_classes = len(class_names)
    
    print("\n混淆矩阵:")
    print(" " * 10, end="")
    for i in range(num_classes):
        print(f"{i:3d} ", end="")
    print()
    print(" " * 8 + "-" * (num_classes * 4 + 1))
    
    for i in range(num_classes):
        print(f"{class_names[i]:9s} |", end="")
        for j in range(num_classes):
            if confusion_matrix[i, j] > 0:
                print(f"{confusion_matrix[i, j]:3d} ", end="")
            else:
                print("  0 ", end="")
        print(f"| ({np.sum(confusion_matrix[i, :])})")
    
    print(" " * 8 + "-" * (num_classes * 4 + 1))
    
    # 计算并打印每类的准确率
    print("\n每个类别的准确率:")
    for i in range(num_classes):
        class_correct = confusion_matrix[i, i]
        class_total = np.sum(confusion_matrix[i, :])
        if class_total > 0:
            class_acc = class_correct / class_total
            print(f"  {class_names[i]:12s}: {class_acc:.3f} ({class_correct}/{class_total})")
        else:
            print(f"  {class_names[i]:12s}: 0.000 (0/0)")

''' 
    # 测试

def main():
    """主训练函数"""
    # 设置随机种子以确保可复现性
    np.random.seed(42)
    
    # 1. 加载数据
    print("加载Fashion-MNIST数据集...")
    loader = FashionMNISTLoader(data_dir='./data/fashion_mnist')
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = loader.load_data(validation_ratio=0.1)
    
    # 2. 创建模型
    print("\n初始化模型...")
    input_dim = x_train.shape[1] 
    hidden_dim = 256
    output_dim = 10
    activation = 'relu'  
    
    model = ThreeLayerNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        activation=activation
    )
    
    # 3. 创建优化器和学习率调度器
    print("设置优化器...")
    learning_rate = 0.01
    weight_decay = 0.0001
    momentum = 0.9
    
    optimizer = SGD(
        model=model,
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum
    )
    
    # 创建学习率调度器（每10个epoch学习率减半）
    scheduler = LearningRateScheduler(
        optimizer=optimizer,
        decay_type='step',
        decay_rate=0.5,
        decay_steps=10
    )
    
    # 4. 创建训练器并开始训练
    print("开始训练过程...")
    trainer = Trainer(model, optimizer, scheduler)
    
    # 训练参数
    epochs = 40
    batch_size = 64
    save_dir = './checkpoints'
    
    # 开始训练
    history = trainer.train(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        save_dir=save_dir,
        save_every=10
    )
    
    # 5. 加载最佳模型并在测试集上评估
    print("\n在测试集上评估最佳模型...")
    
    # 重新创建一个新模型
    test_model_instance = ThreeLayerNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        activation=activation
    )
    
    # 创建一个临时的优化器
    temp_optimizer = SGD(
        model=test_model_instance,
        lr=learning_rate,  # 使用相同的学习率
        weight_decay=weight_decay,
        momentum=momentum
    )
    
    # 创建一个临时的训练器来加载最佳模型
    temp_trainer = Trainer(test_model_instance, temp_optimizer, None)  
    best_model_path = os.path.join(save_dir, 'best_model.pkl')
    
    if os.path.exists(best_model_path):
        temp_trainer.load_model(best_model_path)
        
        # 在测试集上评估
        test_accuracy, confusion_matrix = test_model(
            model=test_model_instance,
            x_test=x_test,
            y_test=y_test,
            batch_size=100
        )
        
        print(f"\n测试集准确率: {test_accuracy:.4f}")
        
        # 打印混淆矩阵
        print_confusion_matrix(confusion_matrix, loader.class_names)
    else:
        print(f"未找到最佳模型文件: {best_model_path}")
    
    # 6. 可视化训练过程
    print("\n" + "="*60)
    print("训练过程摘要:")
    print(f"  最终训练损失: {history['train_loss'][-1]:.4f}")
    print(f"  最终训练准确率: {history['train_acc'][-1]:.4f}")
    print(f"  最终验证损失: {history['val_loss'][-1]:.4f}")
    print(f"  最终验证准确率: {history['val_acc'][-1]:.4f}")
    print(f"  最佳验证准确率: {trainer.best_val_acc:.4f} (epoch {trainer.best_epoch})")
    
    # 7. 获取第一层权重用于可视化
    print("\n获取第一层权重用于可视化...")
    first_layer_weights = test_model_instance.get_first_layer_weights()
    print(f"第一层权重形状: {first_layer_weights.shape}")
    print(f"权重范围: [{first_layer_weights.min():.4f}, {first_layer_weights.max():.4f}]")
    
    # 保存权重用于后续可视化
    weights_path = os.path.join(save_dir, 'first_layer_weights.npy')
    np.save(weights_path, first_layer_weights)
    print(f"第一层权重已保存到: {weights_path}")

if __name__ == "__main__":
    main()
'''