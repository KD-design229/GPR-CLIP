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
        
        # 备份上一轮模型 (fixed)
        # W_old 是字典格式 {param_name: tensor}，需要转换为 state_dict 格式
        prev_model = copy.deepcopy(self.model)
        # 将 W_old (named_parameters dict) 转换为 state_dict 格式
        prev_state_dict = {name: param.data.clone() for name, param in self.W_old.items()}
        prev_model.load_state_dict(prev_state_dict, strict=False)
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
                # 计算余弦相似度并应用温度缩放
                sim_glob = self.cos(z, z_glob) / self.temperature
                sim_prev = self.cos(z, z_prev) / self.temperature
                
                # 构造对比损失的 logits (正样本在第0列)
                logits_con = torch.cat([sim_glob.reshape(-1, 1), sim_prev.reshape(-1, 1)], dim=1)
                labels_con = torch.zeros(logits_con.size(0), dtype=torch.long).to(self.device)
                loss_con = self.loss_fn(logits_con, labels_con)

                # Total Loss
                loss = loss_ce + self.mu * loss_con
                
                loss.backward()
                self.optimizer.step()
                loss_logs.append(loss.item())

        return np.mean(loss_logs)