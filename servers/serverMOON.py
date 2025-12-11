import copy
import time
import numpy as np
import torch

from servers.serverBase import ServerBase
from clients.clientMOON import clientMOON

class FedMOON(ServerBase):
    def __init__(self, args, modelObj, run_times, logger):
        super(FedMOON, self).__init__(args, modelObj, run_times, logger)
        
        # 1. 划分数据集
        self.dataset_division()
        
        # 2. 设置客户端（使用 clientMOON）
        self.set_clients(args, self.global_model, clientMOON)
        
        self.logger.info(f"FedMOON Server Initialized with mu={args.moon_mu}, temperature={args.moon_temperature}")

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
            
            # 4. 客户端本地训练
            loss_logs = []
            train_acc_logs = []
            for idx in self.selected_clients_idx:
                client_loss = self.clientsObj[idx].train()
                loss_logs.append(client_loss)
                client_train_acc = self.clientsObj[idx].train_accuracy()
                train_acc_logs.append(client_train_acc)
                
            self.client_train_loss_logs.append(loss_logs)
            self.client_train_acc_logs.append(train_acc_logs)

            # 5. 接收并聚合模型
            self.receive_models()
            self.aggregated()  # 标准 FedAvg 聚合

            t2 = time.time()
            self.round_duration_logs.append(t2 - t1)

            self.logger.info(f'Round {i + 1}: cost={t2 - t1:.4f}s, mean_acc={client_mean_test_acc:.4f}')
            
        self.save_results()