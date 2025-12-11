import torch
import torch.nn as nn
import numpy as np
import time
import copy
from clients.clientBase import ClientBase

class clientFedProx(ClientBase):
    def __init__(self, args, id, modelObj, train_set, test_set, **kwargs):
        super(clientFedProx, self).__init__(args, id, modelObj, train_set, test_set, **kwargs)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=self.lr, 
            momentum=0.0,
            weight_decay=getattr(args, 'weight_decay', 0.0)
        )
        
        # 获取 FedProx 的超参数 mu
        self.mu = args.mu

    def train(self):
        """
        Local Training with Proximal Term for FedProx
        """
        # 1. 备份全局模型参数 (w_global)
        # 注意：此时 self.model 已经被 Server 更新为最新的全局模型
        global_weight_collector = copy.deepcopy(list(self.model.parameters()))

        self.model.train()
        loss_logs = []

        for e in range(self.E):
            for data in self.train_loader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                
                self.optimizer.zero_grad()
                
                outputs = self.model(inputs)
                
                # 2. 计算原始 Loss
                loss = self.loss_fn(outputs, labels)

                # 3. 计算 Proximal Term (近端项)
                # Loss += (mu / 2) * || w - w_t ||^2
                fed_prox_reg = 0.0
                for param_index, param in enumerate(self.model.parameters()):
                    fed_prox_reg += ((self.mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
                
                loss += fed_prox_reg
                
                loss.backward()
                self.optimizer.step()
                
                loss_logs.append(loss.item())

        return np.mean(loss_logs)