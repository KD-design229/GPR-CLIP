import torch
import torch.nn as nn
import numpy as np
import time
import copy
from clients.clientBase import ClientBase

class clientFedAvg(ClientBase):
    def __init__(self, args, id, modelObj, train_set, test_set, **kwargs):
        super(clientFedAvg, self).__init__(args, id, modelObj, train_set, test_set, **kwargs)

        self.loss_fn = nn.CrossEntropyLoss()
        # 初始化优化器 (与 FedDWA 保持一致的超参数)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=self.lr, 
            momentum=0.0, # FedAvg 通常不加动量，或者设为 args.momentum
            weight_decay=getattr(args, 'weight_decay', 0.0)
        )

    def train(self):
        """
        Standard Local Training for FedAvg
        """
        self.model.train()
        loss_logs = []

        for e in range(self.E): # Local Epochs
            for data in self.train_loader:
                # 获取数据
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                
                self.optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                loss_logs.append(loss.item())

        return np.mean(loss_logs)