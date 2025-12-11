import torch
import torch.nn as nn
import numpy as np
import copy
from clients.clientBase import ClientBase

class clientFedNova(ClientBase):
    def __init__(self, args, id, modelObj, train_set, test_set, **kwargs):
        super(clientFedNova, self).__init__(args, id, modelObj, train_set, test_set, **kwargs)
        
        self.loss_fn = nn.CrossEntropyLoss()
        # FedNova 建议使用 momentum SGD (rho > 0, 通常 0.9)
        self.rho = getattr(args, 'rho', 0.9)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=self.lr, 
            momentum=self.rho, 
            weight_decay=getattr(args, 'weight_decay', 0.0)
        )
        
        # [FedNova] 缓存训练准确率，因为 train() 结束后模型会被修改为归一化形式
        self._cached_train_acc = None

    def train(self):
        """
        FedNova Local Training
        Returns: mean_loss, a_i (normalization factor)
        """
        # 1. 备份初始全局模型参数 (W_global)
        global_model_params = copy.deepcopy(list(self.model.parameters()))
        
        self.model.train()
        loss_logs = []
        tau = 0

        # 2. 本地训练循环
        for e in range(self.E):
            for data in self.train_loader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                loss_logs.append(loss.item())
                tau += 1
        
        # 3. [重要] 在修改模型前，先计算并缓存训练准确率
        self._cached_train_acc = self._compute_train_accuracy()
        
        # 4. 计算归一化因子 a_i
        # FedNova 论文公式: 
        # - 无 momentum (rho=0): a_i = tau
        # - 有 momentum (rho>0): a_i = tau / (1 - rho^tau)
        if self.rho > 0 and self.rho < 1:
            a_i = tau / (1.0 - pow(self.rho, tau))
        else:
            a_i = float(tau)

        # 5. [核心] 不修改模型参数，直接发送训练后的模型
        # 服务器端会处理归一化
        # 这里只需要返回 a_i 即可

        return np.mean(loss_logs), a_i
    
    def _compute_train_accuracy(self):
        """内部方法：计算训练集准确率（在模型被修改前调用）"""
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for data in self.train_loader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(inputs)
                _, predicts = torch.max(outputs, 1)
                correct += (predicts == labels).sum().item()
                total += len(labels)
        return correct / total if total > 0 else 0.0
    
    def train_accuracy(self):
        """重写：返回缓存的训练准确率（因为 train() 后模型处于归一化状态）"""
        if self._cached_train_acc is not None:
            return self._cached_train_acc
        # Fallback: 如果没有缓存，调用父类方法
        return super().train_accuracy()
