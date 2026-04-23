import numpy as np
import pickle
import matplotlib.pyplot as plt
import random


class NeuralNetwork():
    def __init__(self, layers_config, learning_rate=0.001):
        self.layers_config = layers_config
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        self._init_parameters()

    def _init_parameters(self):
        for i, (input_size, output_size, activation) in enumerate(self.layers_config):
            W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
            b = np.zeros((1, output_size))
            self.weights.append(W)
            self.biases.append(b)

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        return (x > 0).astype(float)

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def _forward(self, X):
        self.activations = [X]
        self.z_values = []
        
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(self.activations[-1], W) + b
            self.z_values.append(z)
            
            _, _, activation = self.layers_config[i]
            if activation == 'relu':
                a = self._relu(z)
            elif activation == 'softmax':
                a = self._softmax(z)
            else:
                a = z
            
            self.activations.append(a)
        
        return self.activations[-1]

    def _compute_loss(self, y_pred, y_true, penalty=0.1):
        """计算交叉熵损失 + 大数字预测为小数字的惩罚"""
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        y_true_flat = y_true.flatten().astype(int)
        
        # 标准交叉熵损失
        correct_class_probs = y_pred[np.arange(m), y_true_flat]
        ce_loss = -np.sum(np.log(correct_class_probs)) / m
        
        # 额外惩罚：当真实标签 > 预测标签时
        pred_labels = np.argmax(y_pred, axis=1)
        large_to_small_error = (y_true_flat > pred_labels).astype(float)
        penalty_loss = penalty * np.sum(large_to_small_error) / m
        
        return ce_loss + penalty_loss

    def _one_hot(self, y, num_classes=10):
        one_hot = np.zeros((y.shape[0], num_classes))
        one_hot[np.arange(y.shape[0]), y.flatten()] = 1
        return one_hot

    def _backward(self, y_true):
        m = y_true.shape[0]
        y_true_onehot = self._one_hot(y_true)
        
        pred_labels = np.argmax(self.activations[-1], axis=1)
        y_true_flat = y_true.flatten().astype(int)
        
        dz = self.activations[-1] - y_true_onehot
        
        for sample_idx in range(m):
            true_label = y_true_flat[sample_idx]
            pred_label = pred_labels[sample_idx]
            if true_label > pred_label:
                dz[sample_idx, true_label] += 0.5
                dz[sample_idx, pred_label] -= 0.5
        
        for i in range(len(self.weights) - 1, -1, -1):
            dW = np.dot(self.activations[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            
            if i > 0:
                dz = np.dot(dz, self.weights[i].T) * self._relu_derivative(self.z_values[i-1])
            
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db

    def fit(self, X, y, epochs=100, batch_size=32):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            total_loss = 0
            n_batches = 0
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                y_pred = self._forward(X_batch)
                loss = self._compute_loss(y_pred, y_batch)
                total_loss += loss
                n_batches += 1
                
                self._backward(y_batch)
            
            avg_loss = total_loss / n_batches
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        probabilities = self._forward(X)
        return np.argmax(probabilities, axis=1)

    def predict_proba(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self._forward(X)

    def save_model(self, file_path):
        """
        将神经网络的权重和偏置保存到文件（使用 pickle 格式）。
        """
        model_data = {
            "layers_config": self.layers_config,
            "weights": self.weights,
            "biases": self.biases,
            "learning_rate": self.learning_rate
        }
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"模型已保存至 {file_path}")

    @staticmethod
    def load_model(file_path):
        """
        从文件加载保存的神经网络模型（使用 pickle 格式）。
        """
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)

        nn = NeuralNetwork(model_data["layers_config"], model_data["learning_rate"])
        nn.weights = model_data["weights"]
        nn.biases = model_data["biases"]

        print(f"模型已从 {file_path} 加载")
        return nn

    def visualize_predictions(self, X, y, n_samples=5):
        """
        可视化正确和错误预测的示例。
        
        参数:
            X: 测试数据
            y: 真实标签
            n_samples: 每个类别显示的样本数
        """
        predictions = self.predict(X)
        
        correct_indices = np.where(predictions == y)[0]
        incorrect_indices = np.where(predictions != y)[0]
        
        fig, axes = plt.subplots(2, n_samples, figsize=(12, 5))
        
        # 绘制正确预测
        for i, idx in enumerate(correct_indices[:n_samples]):
            img = X[idx].reshape(28, 28)
            axes[0, i].imshow(img, cmap='gray')
            axes[0, i].axis('off')
            axes[0, i].set_title(f'True: {y[idx]}\nPred: {predictions[idx]}', color='green')
        
        # 绘制错误预测
        for i, idx in enumerate(incorrect_indices[:n_samples]):
            img = X[idx].reshape(28, 28)
            axes[1, i].imshow(img, cmap='gray')
            axes[1, i].axis('off')
            axes[1, i].set_title(f'True: {y[idx]}\nPred: {predictions[idx]}', color='red')
        
        axes[0, 0].set_ylabel('Correct', fontsize=12)
        axes[1, 0].set_ylabel('Incorrect', fontsize=12)
        
        plt.suptitle('Neural Network Predictions Visualization', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        # 打印统计信息
        print(f"总样本数: {len(y)}")
        print(f"正确预测: {len(correct_indices)} ({100*len(correct_indices)/len(y):.2f}%)")
        print(f"错误预测: {len(incorrect_indices)} ({100*len(incorrect_indices)/len(y):.2f}%)")


def load_data(filepath):
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    labels = data[:, 0].astype(int)
    features = data[:, 1:] / 255.0  # 归一化到 0-1
    return features, labels


def train_test_split(X, y, test_ratio=0.2, random_state=None):
    """
    打乱并分割训练集和测试集。
    
    参数:
        X: 特征数据
        y: 标签数据
        test_ratio: 测试集比例 (默认 0.2)
        random_state: 随机种子 (可选)
    
    返回:
        X_train, y_train, X_test, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    
    split_idx = int(n_samples * (1 - test_ratio))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    return X[train_indices], y[train_indices], X[test_indices], y[test_indices]


if __name__ == "__main__":
    random.seed(42)
    layers_config = [
        (784, 128, 'relu'),
        (128, 64, 'relu'),
        (64, 10, 'softmax')
    ]
    learning_rate = 0.03
    epochs = 200
    
    import os
    
    # 检查是否已有划分好的数据文件
    if os.path.exists('train_data.csv') and os.path.exists('test_data.csv'):
        print("Loading split data from CSV...")
        X_train, y_train = load_data('train_data_1.csv')
        X_test, y_test = load_data('test_data.csv')
        print(f"Loaded train: {X_train.shape[0]} samples, test: {X_test.shape[0]} samples")
    else:
        print("Loading training data...")
        X, y = load_data('mnist1.csv')
        
        # 8:2 分割数据集（打乱）
        X_train, y_train, X_test, y_test = train_test_split(X, y, test_ratio=0.2, random_state=42)
        
        print(f"训练集: {X_train.shape[0]} 样本")
        print(f"测试集: {X_test.shape[0]} 样本")
        
        # 保存划分后的数据到CSV
        print("Saving split data to CSV...")
        train_data = np.column_stack([y_train.reshape(-1, 1), X_train])
        test_data = np.column_stack([y_test.reshape(-1, 1), X_test])
        np.savetxt('train_data.csv', train_data, delimiter=',', fmt='%.6f')
        np.savetxt('test_data.csv', test_data, delimiter=',', fmt='%.6f')
        print("Split data saved to train_data.csv and test_data.csv")
        
        # 从CSV加载训练和测试数据
        print("Loading split data from CSV...")
        X_train, y_train = load_data('train_data.csv')
        X_test, y_test = load_data('test_data.csv')
        print(f"Loaded train: {X_train.shape[0]} samples, test: {X_test.shape[0]} samples")
    
    print("Initializing neural network...")
    model = NeuralNetwork(layers_config, learning_rate)
    
    print("Training model...")
    model.fit(X_train, y_train, epochs=epochs, batch_size=32)
    
    print("Saving model...")
    model.save_model('model.pkl')
    
    print("Visualizing predictions on test data...")
    model.visualize_predictions(X_test, y_test, n_samples=5)
    
    print("Training complete!")