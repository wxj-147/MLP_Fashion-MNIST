import numpy as np
import os
import pickle
import time
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from dataloader import FashionMNISTLoader
from model import ThreeLayerNet, SGD, LearningRateScheduler
from train import Trainer, test_model
import random
from matplotlib.ticker import MaxNLocator

class SimpleHyperparameterSearch:
    """超参数随机搜索类"""
    
    def __init__(self, data_dir: str = './data/fashion_mnist', 
                 validation_ratio: float = 0.1,
                 seed: int = 42):
        """
        初始化超参数搜索
        
        参数:
            data_dir: 数据目录
            validation_ratio: 验证集比例
            seed: 随机种子
        """
        self.data_dir = data_dir
        self.validation_ratio = validation_ratio
        self.seed = seed
        
        # 设置随机种子
        np.random.seed(seed)
        random.seed(seed)
        
        # 加载数据
        print("加载Fashion-MNIST数据集...")
        self.loader = FashionMNISTLoader(data_dir=data_dir)
        (self.x_train, self.y_train), (self.x_val, self.y_val), (self.x_test, self.y_test) = \
            self.loader.load_data(validation_ratio=validation_ratio)
        
        print(f"数据集统计:")
        print(f"  训练集: {self.x_train.shape[0]} 个样本")
        print(f"  验证集: {self.x_val.shape[0]} 个样本")
        print(f"  测试集: {self.x_test.shape[0]} 个样本")
        
        # 存储搜索结果的列表
        self.search_results = []
        self.best_result = None
        self.best_val_acc = 0.0
        self.best_model = None

    def _sample_hyperparameters(self) -> Dict[str, Any]:
        """
        从参数分布中随机采样一组超参数
        只调节三个参数：学习率、隐藏层大小、正则化强度
        
        返回:
            采样得到的超参数组合
        """
        params = {}
        
        # 1. 学习率：在对数尺度上均匀采样 [1e-3, 1e-1]
        lr_min, lr_max = 1e-3, 1e-1
        params['learning_rate'] = 10 ** np.random.uniform(np.log10(lr_min), np.log10(lr_max))
        
        # 2. 隐藏层大小：从几个选项中选择
        hidden_options = [64, 128, 256, 512]
        params['hidden_dim'] = random.choice(hidden_options)
        
        # 3. 正则化强度：在对数尺度上均匀采样 [1e-6, 1e-2]
        wd_min, wd_max = 1e-6, 1e-2
        params['weight_decay'] = 10 ** np.random.uniform(np.log10(wd_min), np.log10(wd_max))
        
        # 固定其他参数
        params['activation'] = 'relu'  # 固定使用ReLU，可选Sigmoid
        params['batch_size'] = 64
        params['epochs'] = 30
        params['decay_rate'] = 0.5
        params['decay_steps'] = 10
        params['momentum'] = 0.9
        
        return params
    
    def train_with_params(self, params: Dict[str, Any], 
                         trial_id: int = 0,
                         save_dir: str = './simple_search_results') -> Dict[str, Any]:
        """
        使用给定的超参数训练模型
        
        返回:
            包含训练结果和超参数的字典
        """
        # 创建保存目录
        trial_dir = os.path.join(save_dir, f'trial_{trial_id:03d}')
        os.makedirs(trial_dir, exist_ok=True)
        
        # 设置随机种子（确保每次训练可复现）
        np.random.seed(self.seed + trial_id)

        print(f"\n{'='*60}")
        print(f"试验 #{trial_id}:")
        print(f"  隐藏层大小: {params['hidden_dim']}")
        print(f"  学习率: {params['learning_rate']:.6f}")
        print(f"  正则化强度: {params['weight_decay']:.6f}")
        
        start_time = time.time()
        
        try:
            # 1. 创建模型
            model = ThreeLayerNet(
                input_dim=self.x_train.shape[1],
                hidden_dim=params['hidden_dim'],
                output_dim=10,
                activation=params['activation']
            )
            
            # 2. 创建优化器
            optimizer = SGD(
                model=model,
                lr=params['learning_rate'],
                weight_decay=params['weight_decay'],
                momentum=params['momentum']
            )
            
            # 3. 创建学习率调度器
            scheduler = LearningRateScheduler(
                optimizer=optimizer,
                decay_type='step',
                decay_rate=params['decay_rate'],
                decay_steps=params['decay_steps']
            )
            
            # 4. 创建训练器并训练
            trainer = Trainer(model, optimizer, scheduler)
            
            history = trainer.train(
                x_train=self.x_train,
                y_train=self.y_train,
                x_val=self.x_val,
                y_val=self.y_val,
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                save_dir=trial_dir,
                save_every=params['epochs']  # 只保存最后的结果
            )
            
            # 5. 记录结果
            training_time = time.time() - start_time
            
            result = {
                'trial_id': trial_id,
                'params': params.copy(),
                'model': model,  # 保存模型实例
                'trainer': trainer,  # 保存训练器实例
                'best_val_acc': trainer.best_val_acc,
                'best_epoch': trainer.best_epoch,
                'final_train_loss': history['train_loss'][-1],
                'final_train_acc': history['train_acc'][-1],
                'final_val_loss': history['val_loss'][-1],
                'final_val_acc': history['val_acc'][-1],
                'training_time': training_time,
                'history': {
                    'train_loss': history['train_loss'],
                    'train_acc': history['train_acc'],
                    'val_loss': history['val_loss'],
                    'val_acc': history['val_acc'],
                    'learning_rates': history.get('learning_rates', [])
                },
                'model_path': os.path.join(trial_dir, 'best_model.pkl'),
                'trial_dir': trial_dir
            }
            
            print(f"\n  试验 #{trial_id} 完成!")
            print(f"  最佳验证准确率: {trainer.best_val_acc:.4f} (epoch {trainer.best_epoch})")
            print(f"  训练时间: {training_time:.2f}秒")
            
            return result
            
        except Exception as e:
            print(f"\n  试验 #{trial_id} 失败: {str(e)}")
            return {
                'trial_id': trial_id,
                'params': params.copy(),
                'error': str(e),
                'best_val_acc': 0.0,
                'training_time': time.time() - start_time
            }
        
    def random_search(self, n_iter: int = 20,
                     save_dir: str = './simple_search_results') -> Dict[str, Any]:
        """
        执行随机搜索（只调节三个参数）
        
        参数:
            n_iter: 随机搜索的迭代次数（试验次数）
            save_dir: 结果保存目录
        
        返回:
            包含所有结果的字典
        """
        print("\n" + "="*60)
        print("开始超参数随机搜索")
        print("调节以下三个参数:")
        print("  1. 学习率 (learning_rate)")
        print("  2. 隐藏层大小 (hidden_dim)")
        print("  3. 正则化强度 (weight_decay)")
        print(f"搜索次数: {n_iter} 次")
        print("="*60)
        
        # 创建主保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        all_results = []
        
        # 串行训练
        for i in range(n_iter):
            # 采样一组超参数
            params = self._sample_hyperparameters()
            
            # 训练模型
            result = self.train_with_params(params, trial_id=i, save_dir=save_dir)
            all_results.append(result)
            
            # 更新最佳结果
            if 'error' not in result and result['best_val_acc'] > self.best_val_acc:
                self.best_val_acc = result['best_val_acc']
                self.best_result = result
                self.best_model = result.get('model', None)
                print(f"  新的最佳结果! 准确率: {self.best_val_acc:.4f}")
            
            # 保存中间结果
            self._save_intermediate_results(all_results, save_dir)
        
        # 保存最终结果
        self.search_results = all_results
        self._save_final_results(save_dir)
        
        # 输出总结
        self._print_search_summary()
        
        return {
            'all_results': all_results,
            'best_result': self.best_result,
            'best_val_acc': self.best_val_acc,
            'best_model': self.best_model
        }

    def _save_intermediate_results(self, results: List[Dict], save_dir: str):
        """保存中间结果"""
        intermediate_path = os.path.join(save_dir, 'intermediate_results.pkl')
        with open(intermediate_path, 'wb') as f:
            pickle.dump(results, f)
    
    def _save_final_results(self, save_dir: str):
        """保存最终结果"""
        # 保存所有结果
        results_path = os.path.join(save_dir, 'search_results.pkl')
        with open(results_path, 'wb') as f:
            # 不能直接保存模型，先移除
            serializable_results = []
            for result in self.search_results:
                serializable_result = result.copy()
                if 'model' in serializable_result:
                    del serializable_result['model']
                if 'trainer' in serializable_result:
                    del serializable_result['trainer']
                serializable_results.append(serializable_result)
            
            pickle.dump(serializable_results, f)
        
        # 保存最佳结果
        if self.best_result:
            best_path = os.path.join(save_dir, 'best_result.pkl')
            with open(best_path, 'wb') as f:
                best_serializable = self.best_result.copy()
                if 'model' in best_serializable:
                    del best_serializable['model']
                if 'trainer' in best_serializable:
                    del best_serializable['trainer']
                pickle.dump(best_serializable, f)
            
            # 也保存为JSON（便于查看）
            best_json_path = os.path.join(save_dir, 'best_result.json')
            with open(best_json_path, 'w') as f:
                best_serializable = self.best_result.copy()
                if 'model' in best_serializable:
                    del best_serializable['model']
                if 'trainer' in best_serializable:
                    del best_serializable['trainer']
                if 'history' in best_serializable:
                    del best_serializable['history']
                json.dump(best_serializable, f, indent=2, default=str)
        
        print(f"\n所有搜索结果已保存到: {save_dir}")

    def _print_search_summary(self):
        """打印搜索总结"""
        if not self.search_results:
            print("没有可用的搜索结果")
            return
        
        # 过滤出成功的结果
        successful_results = [r for r in self.search_results if 'error' not in r]
        
        if not successful_results:
            print("所有试验均失败")
            return
        
        print("\n" + "="*60)
        print("随机搜索总结")
        print("="*60)
        
        print(f"总试验次数: {len(self.search_results)}")
        print(f"成功试验次数: {len(successful_results)}")
        
        if self.best_result:
            print(f"\n最佳超参数组合 (试验 #{self.best_result['trial_id']}):")
            print(f"  验证准确率: {self.best_result['best_val_acc']:.4f}")
            print(f"  隐藏层大小: {self.best_result['params']['hidden_dim']}")
            print(f"  学习率: {self.best_result['params']['learning_rate']:.6f}")
            print(f"  正则化强度: {self.best_result['params']['weight_decay']:.6f}")
            print(f"  训练时间: {self.best_result['training_time']:.2f}秒")
        
    def evaluate_best_model_on_test(self, save_dir: str = './simple_search_results') -> Dict[str, Any]:
        """
        在测试集上评估最佳模型
        
        返回:
            测试结果
        """
        if not self.best_model:
            print("没有找到最佳模型")
            return {}
        
        print("\n" + "="*60)
        print("在测试集上评估最佳模型")
        print("="*60)
        
        # 在测试集上评估
        test_accuracy, confusion_matrix = test_model(
            model=self.best_model,
            x_test=self.x_test,
            y_test=self.y_test,
            batch_size=100
        )
        
        print(f"\n测试结果:")
        print(f"  测试准确率: {test_accuracy:.4f}")
        print(f"  验证准确率: {self.best_result['best_val_acc']:.4f} (最佳模型)")
        
        # 打印混淆矩阵
        print_confusion_matrix(confusion_matrix, self.loader.class_names)
        
        # 保存测试结果
        test_result = {
            'test_accuracy': test_accuracy,
            'best_val_accuracy': self.best_result['best_val_acc'],
            'best_params': self.best_result['params']
        }
        
        result_path = os.path.join(save_dir, 'test_results.json')
        with open(result_path, 'w') as f:
            json.dump(test_result, f, indent=2, default=str)
        
        print(f"\n测试结果已保存到: {result_path}")
        
        return test_result      

    def visualize_training_process(self, save_dir: str = './simple_search_results'):
        """
        可视化训练过程
        绘制训练集和验证集上的Loss曲线，以及验证集上的Accuracy曲线
        """
        if not self.best_result or 'history' not in self.best_result:
            print("没有找到最佳模型的训练历史，无法可视化训练过程")
            return
        
        print("\n" + "="*60)
        print("可视化最佳模型的训练过程")
        print("="*60)
        
        history = self.best_result['history']
        epochs = len(history['train_loss'])
        
        # 创建可视化目录
        viz_dir = os.path.join(save_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建大图
        fig = plt.figure(figsize=(16, 10))
        
        # 1. 训练和验证损失曲线
        ax1 = plt.subplot(2, 3, 1)
        epochs_range = range(1, epochs + 1)
        ax1.plot(epochs_range, history['train_loss'], 'b-', linewidth=2, label='训练损失', alpha=0.8)
        ax1.plot(epochs_range, history['val_loss'], 'r-', linewidth=2, label='验证损失', alpha=0.8)
        ax1.set_xlabel('训练轮数 (Epoch)', fontsize=12)
        ax1.set_ylabel('损失 (Loss)', fontsize=12)
        ax1.set_title('训练和验证损失曲线', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # 标记最佳epoch
        best_epoch = self.best_result.get('best_epoch', 0)
        if 0 < best_epoch <= epochs:
            best_val_loss = history['val_loss'][best_epoch-1]
            ax1.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, linewidth=1.5)
            ax1.plot(best_epoch, best_val_loss, 'g*', markersize=12, label=f'最佳epoch {best_epoch}')
            ax1.legend(loc='upper right')
        
        # 2. 训练和验证准确率曲线
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(epochs_range, history['train_acc'], 'b-', linewidth=2, label='训练准确率', alpha=0.8)
        ax2.plot(epochs_range, history['val_acc'], 'r-', linewidth=2, label='验证准确率', alpha=0.8)
        ax2.set_xlabel('训练轮数 (Epoch)', fontsize=12)
        ax2.set_ylabel('准确率 (Accuracy)', fontsize=12)
        ax2.set_title('训练和验证准确率曲线', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='lower right')
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # 设置y轴范围为0-1
        ax2.set_ylim([0, 1])
        
        # 标记最佳epoch
        if 0 < best_epoch <= epochs:
            best_val_acc = history['val_acc'][best_epoch-1]
            ax2.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, linewidth=1.5)
            ax2.plot(best_epoch, best_val_acc, 'g*', markersize=12, label=f'最佳epoch {best_epoch}')
            ax2.legend(loc='lower right')
        
        # 3. 验证准确率单独放大
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(epochs_range, history['val_acc'], 'r-', linewidth=2, label='验证准确率', alpha=0.8)
        ax3.set_xlabel('训练轮数 (Epoch)', fontsize=12)
        ax3.set_ylabel('验证准确率', fontsize=12)
        ax3.set_title('验证集准确率曲线', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 标记最佳准确率
        if 0 < best_epoch <= epochs:
            ax3.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, linewidth=1.5)
            ax3.plot(best_epoch, best_val_acc, 'g*', markersize=12, label=f'最佳: {best_val_acc:.4f}')
            ax3.legend(loc='lower right')
        
        ax3.set_ylim([max(0, min(history['val_acc'])-0.05), 1])
        ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # 4. 损失和准确率对比（双y轴）
        ax4 = plt.subplot(2, 3, 4)
        
        # 损失曲线
        color = 'tab:blue'
        ax4.set_xlabel('训练轮数 (Epoch)', fontsize=12)
        ax4.set_ylabel('损失 (Loss)', color=color, fontsize=12)
        loss_line = ax4.plot(epochs_range, history['val_loss'], color=color, linewidth=2, label='验证损失', alpha=0.8)
        ax4.tick_params(axis='y', labelcolor=color)
        ax4.grid(True, alpha=0.3)
        
        # 准确率曲线（第二个y轴）
        ax5 = ax4.twinx()
        color = 'tab:red'
        ax5.set_ylabel('准确率 (Accuracy)', color=color, fontsize=12)
        acc_line = ax5.plot(epochs_range, history['val_acc'], color=color, linewidth=2, label='验证准确率', alpha=0.8)
        ax5.tick_params(axis='y', labelcolor=color)
        
        # 组合图例
        lines = loss_line + acc_line
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper left')
        
        ax4.set_title('验证集损失和准确率对比', fontsize=14, fontweight='bold')
        ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # 5. 训练损失和准确率对比（双y轴）
        ax6 = plt.subplot(2, 3, 5)
        
        # 训练损失曲线
        color = 'tab:blue'
        ax6.set_xlabel('训练轮数 (Epoch)', fontsize=12)
        ax6.set_ylabel('训练损失', color=color, fontsize=12)
        train_loss_line = ax6.plot(epochs_range, history['train_loss'], color=color, linewidth=2, label='训练损失', alpha=0.8)
        ax6.tick_params(axis='y', labelcolor=color)
        ax6.grid(True, alpha=0.3)
        
        # 训练准确率曲线（第二个y轴）
        ax7 = ax6.twinx()
        color = 'tab:red'
        ax7.set_ylabel('训练准确率', color=color, fontsize=12)
        train_acc_line = ax7.plot(epochs_range, history['train_acc'], color=color, linewidth=2, label='训练准确率', alpha=0.8)
        ax7.tick_params(axis='y', labelcolor=color)
        
        # 组合图例
        lines = train_loss_line + train_acc_line
        labels = [l.get_label() for l in lines]
        ax6.legend(lines, labels, loc='upper left')
        
        ax6.set_title('训练集损失和准确率对比', fontsize=14, fontweight='bold')
        ax6.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # 6. 学习率变化曲线
        ax8 = plt.subplot(2, 3, 6)
        if 'learning_rates' in history and len(history['learning_rates']) > 0:
            ax8.plot(epochs_range, history['learning_rates'], 'purple', linewidth=2, alpha=0.8)
            ax8.set_xlabel('训练轮数 (Epoch)', fontsize=12)
            ax8.set_ylabel('学习率', fontsize=12)
            ax8.set_title('学习率衰减曲线', fontsize=14, fontweight='bold')
            ax8.grid(True, alpha=0.3)
            ax8.set_yscale('log')  # 对数尺度显示学习率
        else:
            ax8.text(0.5, 0.5, '无学习率数据', ha='center', va='center', fontsize=12)
            ax8.set_title('学习率衰减曲线', fontsize=14, fontweight='bold')
            ax8.axis('off')
        
        ax8.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # 添加超参数信息
        params = self.best_result['params']
        params_text = (
            f"最佳模型参数:\n"
            f"试验ID: {self.best_result['trial_id']}\n"
            f"隐藏层大小: {params['hidden_dim']}\n"
            f"学习率: {params['learning_rate']:.6f}\n"
            f"正则化强度: {params['weight_decay']:.6f}\n"
            f"最佳验证准确率: {self.best_result['best_val_acc']:.4f}\n"
            f"最佳epoch: {best_epoch}"
        )
        
        plt.figtext(0.02, 0.02, params_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor="lightyellow", edgecolor="gray", alpha=0.8))
        
        plt.suptitle(f'最佳模型训练过程可视化 (试验 #{self.best_result["trial_id"]})', 
                    fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # 保存图片
        training_plot_path = os.path.join(viz_dir, 'training_process.png')
        plt.savefig(training_plot_path, dpi=150, bbox_inches='tight')
        print(f"训练过程可视化图已保存到: {training_plot_path}")
        
        plt.show()
        
        # 创建简化的训练过程图
        self._create_simple_training_plot(history, best_epoch, params, viz_dir)
        
        return fig
    
    def _create_simple_training_plot(self, history, best_epoch, params, viz_dir):
        """简化的训练过程图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = len(history['train_loss'])
        epochs_range = range(1, epochs + 1)
        
        # 1. 损失曲线
        ax1.plot(epochs_range, history['train_loss'], 'b-', linewidth=2, label='训练损失')
        ax1.plot(epochs_range, history['val_loss'], 'r-', linewidth=2, label='验证损失')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 标记最佳epoch
        if 0 < best_epoch <= epochs:
            ax1.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, linewidth=1)
        
        # 2. 准确率曲线
        ax2.plot(epochs_range, history['train_acc'], 'b-', linewidth=2, label='训练准确率')
        ax2.plot(epochs_range, history['val_acc'], 'r-', linewidth=2, label='验证准确率')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim([0, 1])
        
        # 标记最佳epoch
        if 0 < best_epoch <= epochs:
            ax2.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, linewidth=1)
            best_val_acc = history['val_acc'][best_epoch-1]
            ax2.plot(best_epoch, best_val_acc, 'g*', markersize=10, 
                    label=f'Best: {best_val_acc:.4f}')
            ax2.legend()
        
        # 添加超参数信息
        params_text = (
            f"Hidden Dim: {params['hidden_dim']}\n"
            f"Learning Rate: {params['learning_rate']:.4f}\n"
            f"Weight Decay: {params['weight_decay']:.6f}\n"
            f"Best Val Acc: {self.best_result['best_val_acc']:.4f}"
        )
        plt.figtext(0.02, 0.02, params_text, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        
        plt.suptitle('Training Process Visualization', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # 保存简化的图片
        simple_plot_path = os.path.join(viz_dir, 'training_process_simple.png')
        plt.savefig(simple_plot_path, dpi=150, bbox_inches='tight')
        
        plt.show()

    def visualize_first_layer_weights(self, save_dir: str = './simple_search_results'):
        """
        可视化第一层隐藏层权重矩阵
        将权重矩阵恢复成图像尺寸 (28×28) 并可视化
        """
        if not self.best_model:
            print("没有找到最佳模型，无法可视化权重")
            return
        
        print("\n" + "="*60)
        print("可视化第一层隐藏层权重")
        print("="*60)
        
        # 获取第一层权重
        weights = self.best_model.get_first_layer_weights()  # 形状: (784, hidden_dim)
        hidden_dim = weights.shape[1]
        
        print(f"第一层权重矩阵形状: {weights.shape}")
        print(f"隐藏层神经元数量: {hidden_dim}")
        
        # 创建可视化目录
        viz_dir = os.path.join(save_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # 可视化权重矩阵
        fig = plt.figure(figsize=(15, 10))
        
        # 1. 显示权重矩阵的热图
        plt.subplot(2, 3, 1)
        plt.imshow(weights.T, aspect='auto', cmap='RdBu_r')
        plt.colorbar(label='权重值')
        plt.xlabel('输入像素 (784维)')
        plt.ylabel('隐藏层神经元')
        plt.title('第一层权重矩阵 (整体视图)')
        
        # 2. 随机选择16个隐藏层神经元的权重，恢复为28×28图像
        plt.subplot(2, 3, 2)
        # 随机选择16个神经元
        if hidden_dim >= 16:
            selected_neurons = np.random.choice(hidden_dim, 16, replace=False)
        else:
            selected_neurons = np.arange(hidden_dim)
        
        # 创建子图网格
        fig2, axes = plt.subplots(4, 4, figsize=(12, 12))
        for idx, neuron_idx in enumerate(selected_neurons):
            ax = axes[idx // 4, idx % 4]
            # 获取该神经元的权重并reshape为28×28
            neuron_weights = weights[:, neuron_idx].reshape(28, 28)
            
            # 可视化
            im = ax.imshow(neuron_weights, cmap='RdBu_r', vmin=-np.max(np.abs(neuron_weights)), 
                          vmax=np.max(np.abs(neuron_weights)))
            ax.set_title(f'神经元 {neuron_idx}')
            ax.axis('off')
        
        plt.suptitle('随机选择的16个隐藏层神经元的权重可视化', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'first_layer_weights_16neurons.png'), dpi=150, bbox_inches='tight')
        
        # 3. 显示前几个神经元的权重（按权重范数排序）
        plt.figure(figsize=(15, 5))
        
        # 计算每个神经元的权重范数
        weight_norms = np.linalg.norm(weights, axis=0)
        top_indices = np.argsort(weight_norms)[-12:]  # 选择权重范数最大的12个
        
        for i, idx in enumerate(top_indices):
            plt.subplot(3, 4, i+1)
            neuron_weights = weights[:, idx].reshape(28, 28)
            plt.imshow(neuron_weights, cmap='RdBu_r')
            plt.title(f'神经元 {idx}\n范数: {weight_norms[idx]:.3f}')
            plt.axis('off')
        
        plt.suptitle('权重范数最大的12个隐藏层神经元的权重可视化', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'first_layer_weights_top_norms.png'), dpi=150, bbox_inches='tight')
        
        # 4. 分析权重模式
        print("\n权重模式分析:")
        print(f"  权重范围: [{weights.min():.4f}, {weights.max():.4f}]")
        print(f"  权重均值: {weights.mean():.4f}")
        print(f"  权重标准差: {weights.std():.4f}")
        
        # 计算正权重和负权重的比例
        positive_ratio = np.sum(weights > 0) / weights.size
        negative_ratio = np.sum(weights < 0) / weights.size
        zero_ratio = np.sum(weights == 0) / weights.size
        
        print(f"  正权重比例: {positive_ratio:.2%}")
        print(f"  负权重比例: {negative_ratio:.2%}")
        print(f"  零权重比例: {zero_ratio:.2%}")
        
        # 5. 保存权重矩阵
        weights_path = os.path.join(viz_dir, 'first_layer_weights.npy')
        np.save(weights_path, weights)
        print(f"\n权重矩阵已保存到: {weights_path}")
        
        plt.show()
        
        return weights

    def visualize_misclassified_examples(self, num_examples: int = 10, 
                                        save_dir: str = './simple_search_results'):
        """
        可视化分类错误的图像
        
        参数:
            num_examples: 要可视化的错例数量
        """
        if not self.best_model:
            print("没有找到最佳模型，无法可视化错例")
            return
        
        print("\n" + "="*60)
        print(f"可视化分类错误的图像 ({num_examples}个)")
        print("="*60)
        
        # 获取测试集上的预测
        predictions = self.best_model.predict(self.x_test)
        
        # 找到分类错误的索引
        misclassified_indices = np.where(predictions != self.y_test)[0]
        
        if len(misclassified_indices) == 0:
            print("没有找到分类错误的图像")
            return
        
        print(f"已找到 {len(misclassified_indices)} 个分类错误的图像")
        
        # 随机选择一些错例
        if len(misclassified_indices) > num_examples:
            selected_indices = np.random.choice(misclassified_indices, num_examples, replace=False)
        else:
            selected_indices = misclassified_indices
        
        # 创建可视化目录
        viz_dir = os.path.join(save_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # 类名
        class_names = self.loader.class_names
        
        # 创建可视化
        n_cols = 5
        n_rows = (num_examples + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, ax in enumerate(axes.flat):
            if idx < len(selected_indices):
                sample_idx = selected_indices[idx]
                image = self.x_test[sample_idx].reshape(28, 28)
                true_label = self.y_test[sample_idx]
                pred_label = predictions[sample_idx]
                
                # 显示图像
                ax.imshow(image, cmap='gray')
                ax.set_title(f'真: {class_names[true_label]}\n预: {class_names[pred_label]}', 
                           fontsize=10, color='red' if true_label != pred_label else 'green')
                ax.axis('off')
                
                # 添加概率信息
                probs = self.best_model.forward(self.x_test[sample_idx:sample_idx+1])
                true_prob = probs[0, true_label]
                pred_prob = probs[0, pred_label]
                ax.text(0.5, -0.15, f'真: {true_prob:.2f}, 预: {pred_prob:.2f}', 
                       transform=ax.transAxes, ha='center', fontsize=8)
            else:
                ax.axis('off')
        
        plt.suptitle('分类错误的Fashion-MNIST图像示例', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'misclassified_examples.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
        return misclassified_indices

    def visualize_hyperparameter_effects(self, save_dir: str = './simple_search_results'):
        """
        可视化三个超参数对模型性能的影响
        生成散点图显示学习率、隐藏层大小、正则化强度与验证准确率的关系
        """
        if not self.search_results:
            print("没有可用的搜索结果，无法可视化超参数影响")
            return
        
        # 过滤出成功的结果
        successful_results = [r for r in self.search_results if 'error' not in r]
        
        if len(successful_results) < 5:
            print(f"只有 {len(successful_results)} 个成功结果，不足以分析超参数影响")
            return
        
        print("\n" + "="*60)
        print("可视化超参数对模型性能的影响")
        print("="*60)
        
        # 提取数据
        learning_rates = []
        hidden_dims = []
        weight_decays = []
        val_accuracies = []
        
        for result in successful_results:
            params = result['params']
            learning_rates.append(params['learning_rate'])
            hidden_dims.append(params['hidden_dim'])
            weight_decays.append(params['weight_decay'])
            val_accuracies.append(result['best_val_acc'])
        
        # 创建可视化目录
        viz_dir = os.path.join(save_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # 创建图表
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. 学习率 vs 验证准确率
        axes[0].scatter(learning_rates, val_accuracies, c=val_accuracies, cmap='viridis', 
                       s=100, alpha=0.7, edgecolors='black')
        axes[0].set_xscale('log')
        axes[0].set_xlabel('学习率 (log scale)', fontsize=12)
        axes[0].set_ylabel('验证准确率', fontsize=12)
        axes[0].set_title('学习率对验证准确率的影响', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # 添加颜色条
        scatter = axes[0].scatter(learning_rates, val_accuracies, c=val_accuracies, cmap='viridis', 
                                 s=100, alpha=0.7, edgecolors='black')
        plt.colorbar(scatter, ax=axes[0], label='验证准确率')
        
        # 2. 隐藏层大小 vs 验证准确率
        axes[1].scatter(hidden_dims, val_accuracies, c=val_accuracies, cmap='viridis', 
                       s=100, alpha=0.7, edgecolors='black')
        axes[1].set_xlabel('隐藏层大小', fontsize=12)
        axes[1].set_ylabel('验证准确率', fontsize=12)
        axes[1].set_title('隐藏层大小对验证准确率的影响', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # 3. 正则化强度 vs 验证准确率
        axes[2].scatter(weight_decays, val_accuracies, c=val_accuracies, cmap='viridis', 
                       s=100, alpha=0.7, edgecolors='black')
        axes[2].set_xscale('log')
        axes[2].set_xlabel('正则化强度 (log scale)', fontsize=12)
        axes[2].set_ylabel('验证准确率', fontsize=12)
        axes[2].set_title('正则化强度对验证准确率的影响', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle('超参数对模型性能的影响分析', fontsize=16, fontweight='bold', y=1.05)
        plt.tight_layout()
        
        # 保存图片
        hyperparam_plot_path = os.path.join(viz_dir, 'hyperparameter_effects.png')
        plt.savefig(hyperparam_plot_path, dpi=150, bbox_inches='tight')
        print(f"超参数影响分析图已保存到: {hyperparam_plot_path}")
        
        plt.show()
        
        return fig

def main():
    """主函数：执行随机搜索"""
    
    # 1. 初始化搜索器
    search = SimpleHyperparameterSearch(
        data_dir='./data/fashion_mnist',
        validation_ratio=0.1,
        seed=42
    )
    
    # 2. 执行随机搜索
    print("\n开始超参数随机搜索...")
    print("调节: 学习率、隐藏层大小、正则化强度")
    
    results = search.random_search(
        n_iter=50,  # 搜索n组不同的超参数
        save_dir='./simple_search_results'
    )
    
    # 3. 在测试集上评估最佳模型
    search.evaluate_best_model_on_test()
    
    # 4. 可视化训练过程
    search.visualize_training_process()
    
    # 5. 可视化第一层权重
    search.visualize_first_layer_weights()
    
    # 6. 可视化分类错误的图像
    search.visualize_misclassified_examples(num_examples=5)
    
    # 7. 可视化超参数影响
    search.visualize_hyperparameter_effects()
    
    print("\n" + "="*60)
    print("随机搜索完成")
    print("="*60)
    
    return search


if __name__ == "__main__":
    search_instance = main()


