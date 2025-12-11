import torch
import torch.nn as nn
import numpy as np
import copy
from clients.clientBase import ClientBase

# 1. 定义 SAM 优化器 (这是 FedSAM 的灵魂)
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        raise NotImplementedError("SAM doesn't support step(), use first_step and second_step")

    def _grad_norm(self):
        # Put everything on the same device, in case of model parallelism
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
    
    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)
    
    # 兼容性函数，防止报错
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


# 2. 定义 Client 类
class clientFedSAM(ClientBase):
    def __init__(self, args, id, modelObj, train_set, test_set, **kwargs):
        super(clientFedSAM, self).__init__(args, id, modelObj, train_set, test_set, **kwargs)
        
        self.loss_fn = nn.CrossEntropyLoss()
        
        # 获取 SAM 的超参数 rho (扰动半径)
        self.sam_rho = getattr(args, 'sam_rho', 0.05) 
        
        # 初始化 SAM，底层包裹 SGD
        self.optimizer = SAM(
            self.model.parameters(), 
            torch.optim.SGD, 
            rho=self.sam_rho, 
            adaptive=False,
            lr=self.lr,
            momentum=getattr(args, 'rho', 0.0), # 通常 FL 里 momentum=0
            weight_decay=getattr(args, 'weight_decay', 0.0)
        )

    def train(self):
        """
        FedSAM Local Training
        """
        self.model.train()
        loss_logs = []

        for e in range(self.E):
            for data in self.train_loader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                
                # --- Step 1: 第一次前向传播 & 反向传播 ---
                # 计算当前点的梯度
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                
                # 计算扰动 e(w) 并加到权重上: w_adv = w + e(w)
                self.optimizer.first_step(zero_grad=True) 

                # --- Step 2: 第二次前向传播 & 反向传播 ---
                # 在扰动后的点 w_adv 上计算梯度
                self.loss_fn(self.model(inputs), labels).backward()
                
                # --- Step 3: 更新权重 ---
                # 恢复 w 并根据 w_adv 的梯度进行更新
                self.optimizer.second_step(zero_grad=True) 

                loss_logs.append(loss.item())

        return np.mean(loss_logs)