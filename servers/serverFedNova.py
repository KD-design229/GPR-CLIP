import copy
import time
import numpy as np
import torch

from servers.serverBase import ServerBase
from clients.clientFedNova import clientFedNova

class FedNova(ServerBase):
    def __init__(self, args, modelObj, run_times, logger):
        super(FedNova, self).__init__(args, modelObj, run_times, logger)
        
        # 1. 划分数据集
        self.dataset_division()
        
        # 2. 设置客户端（使用 clientFedNova）
        self.set_clients(args, self.global_model, clientFedNova)
        
        # 3. 存储每个 client 的 a_i
        self.client_ai_values = {}
        
        self.logger.info(f"FedNova Server Initialized with rho={args.rho}")

    def train(self):
        for i in range(self.global_rounds):
            # 1. 选择客户端
            self.selected_clients_idx = self.select_client()
            self.selected_clients_logs.append(list(self.selected_clients_idx))
            
            # 2. 发送模型
            self.send_models()

            # 3. 评估（每隔 evaluate_gap 轮）
            if i % self.evaluate_gap == 0:
                clients_test_acc, client_mean_test_acc = self.evaluate_acc()
                self.client_test_acc_logs.append(clients_test_acc)
                self.client_mean_test_acc_logs.append(client_mean_test_acc)
                
                if self.client_train_acc_logs:
                    last_train_accs = self.client_train_acc_logs[-1]
                    if len(last_train_accs) > 0:
                        self.global_train_acc_logs.append(float(np.mean(last_train_accs)))

            t1 = time.time()

            # 4. 客户端训练并收集 a_i
            self.client_ai_values = {}
            loss_logs = []
            train_acc_logs = []
            
            for idx in self.selected_clients_idx:
                loss, a_i = self.clientsObj[idx].train()
                self.client_ai_values[idx] = a_i
                loss_logs.append(loss)
                
                # 计算训练准确率（使用缓存的值）
                client_train_acc = self.clientsObj[idx].train_accuracy()
                train_acc_logs.append(client_train_acc)
                
            self.client_train_loss_logs.append(loss_logs)
            self.client_train_acc_logs.append(train_acc_logs)

            # 5. 接收模型
            self.receive_models()
            
            # 6. FedNova 特殊聚合
            # 备份旧的全局模型 W_old
            w_old_state = copy.deepcopy(self.global_model.state_dict())
            
            # 调用基类的 aggregated()
            # 此时 self.global_model 变成: W_old + sum(p_i * (W_i - W_old)/a_i)
            super().aggregated()
            
            # 7. 计算有效全局步长 tau_eff
            # tau_eff = sum(p_i * a_i)
            total_data = sum(self.receive_client_datasize)
            fed_avg_freqs = {
                idx: self.clientsObj[idx].data_size / total_data 
                for idx in self.selected_clients_idx
            }
            
            tau_eff = sum([fed_avg_freqs[idx] * self.client_ai_values[idx] for idx in self.selected_clients_idx])
            
            # 8. 应用最终缩放
            # W_final = W_old + tau_eff * (W_aggregated - W_old)
            w_new_state = self.global_model.state_dict()
            
            for key in w_new_state:
                update = w_new_state[key] - w_old_state[key]
                w_new_state[key] = w_old_state[key] + tau_eff * update
            
            self.global_model.load_state_dict(w_new_state)

            t2 = time.time()
            self.round_duration_logs.append(t2 - t1)

            self.logger.info(f'Round {i + 1}: cost={t2 - t1:.4f}s, mean_acc={client_mean_test_acc:.4f}, tau_eff={tau_eff:.4f}')
            
        self.save_results()