import copy
import time
import numpy as np
import torch

from clients.clientFedProx import clientFedProx
from servers.serverBase import ServerBase

class FedProx(ServerBase):
    def __init__(self, args, modelObj, run_times, logger):
        super(FedProx, self).__init__(args, modelObj, run_times, logger)
        
        # 1. 划分数据
        self.dataset_division()
        
        # 2. 设置客户端 (指定使用 clientFedProx)
        self.set_clients(args, self.global_model, clientFedProx)
        
        self.logger.info(f"FedProx Server Initialized with mu={args.mu}")

    def train(self):
        # 复用与 FedAvg 相同的训练流程
        for i in range(self.global_rounds):
            self.selected_clients_idx = self.select_client()
            self.selected_clients_logs.append(list(self.selected_clients_idx))

            self.send_models()

            if i % self.evaluate_gap == 0:
                clients_test_acc, client_mean_test_acc = self.evaluate_acc()
                self.client_test_acc_logs.append(clients_test_acc)
                self.client_mean_test_acc_logs.append(client_mean_test_acc)
                
                if self.client_train_acc_logs:
                    last_train_accs = self.client_train_acc_logs[-1]
                    if len(last_train_accs) > 0:
                        self.global_train_acc_logs.append(float(np.mean(last_train_accs)))

            t1 = time.time()
            
            loss_logs = []
            train_acc_logs = []
            for idx in self.selected_clients_idx:
                client_loss = self.clientsObj[idx].train()
                loss_logs.append(client_loss)
                client_train_acc = self.clientsObj[idx].train_accuracy()
                train_acc_logs.append(client_train_acc)
                
            self.client_train_loss_logs.append(loss_logs)
            self.client_train_acc_logs.append(train_acc_logs)

            self.receive_models()
            self.aggregated()

            t2 = time.time()
            self.round_duration_logs.append(t2 - t1)

            self.logger.info(f'Round {i + 1}: cost={t2 - t1:.4f}s, mean_acc={client_mean_test_acc:.4f}')
            
        self.save_results()