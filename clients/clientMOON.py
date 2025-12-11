# 文件路径: FedDWA/clients/clientMOON.py
import torch
import torch.nn as nn
import numpy as np
import copy
from clients.clientBase import ClientBase

class clientMOON(ClientBase):
    def __init__(self, args, id, modelObj, train_set, test_set, **kwargs):
        super(clientMOON, self).__init__(args, id, modelObj, train_set, test_set, **kwargs)
        
        # MOON 特有参数
        self.mu = args.moon_mu  # 对比损失权重
        self.temperature = args.moon_temperature
        
        # 优化器
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=self.lr, 
            momentum=0.0, # FedAvg/Prox 通常 momentum=0
            weight_decay=args.weight_decay
        )
        self.cos = nn.CosineSimilarity(dim=-1)

    def train(self):
        """MOON Local Training"""
        # 1. 准备三个模型引用
        # model: 本地正在训练的模型 (z)
        # global_model: 全局模型 (z_glob, 正样本)
        # prev_model: 上一轮的自己 (z_prev, 负样本)
        
        # 备份全局模型 (fixed)
        global_model = copy.deepcopy(self.model)
        for param in global_model.parameters(): param.requires_grad = False
        global_model.eval()
        
        # 备份上一轮模型 (fixed) - 第一次训练时 W_old 是全零/初始值
        # 注意：你需要确保 ClientBase 里正确维护了 self.W_old
        # 这里为了简化，我们假设 prev_model 初始化为 global_model (第一轮时)
        # 实际严谨实现需要 Server 下发上一轮模型，或者 Client 本地缓存
        prev_model = copy.deepcopy(self.model) 
        prev_model.load_state_dict(self.W_old) # 加载上一轮保存的参数
        for param in prev_model.parameters(): param.requires_grad = False
        prev_model.eval()

        self.model.train()
        loss_logs = []

        for e in range(self.E):
            for data in self.train_loader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                self.optimizer.zero_grad()
                
                # 调用模型的 return_features 接口
                outputs, z = self.model(inputs, return_features=True)
                with torch.no_grad():
                    _, z_glob = global_model(inputs, return_features=True)
                    _, z_prev = prev_model(inputs, return_features=True)
                
                # 1. CE Loss
                loss_ce = self.loss_fn(outputs, labels)
                
                # 2. Contrastive Loss (MOON 核心)
                sim_glob = self.cos(z, z_glob)
                sim_prev = self.cos(z, z_prev)
                
                logits_con = torch.cat([sim_glob.reshape(-1, 1), sim_prev.reshape(-1, 1)], dim=1)
                labels_con = torch.zeros(logits_con.size(0), dtype=torch.long).to(self.device)
                loss_con = self.loss_fn(logits_con / self.temperature, labels_con)

                # Total Loss
                loss = loss_ce + self.mu * loss_con
                
                loss.backward()
                self.optimizer.step()
                loss_logs.append(loss.item())

        return np.mean(loss_logs)