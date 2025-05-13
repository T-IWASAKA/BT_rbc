# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

trainer

@author: tadahaya
"""
import torch

from .utils import save_experiment, save_checkpoint, plot_progress

class Trainer:
    def __init__(self, config, model, optimizer, scheduler, loss_fn, exp_name, device):
        self.config = config
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device


    def train(self, trainloader, testloader, classes, save_model_evry_n_epochs=0):
        """
        train the model for the specified number of epochs.
        
        """
        # configの確認
        config = self.config
        assert config["hidden_size"] % config["num_attention_heads"] == 0
        assert config["intermediate_size"] == 4 * config["hidden_size"]
        assert config["image_size"] % config["patch_size"] == 0
        # keep track of the losses and accuracies
        base_dir = "/workspace/data" #後で追加
        train_losses, test_losses, accuracies, train_on_diags, train_off_diags, test_on_diags, test_off_diags = [], [], [], [], [], [], []
        # training
        for i in range(config["epochs"]):
            train_loss, train_on_diag, train_off_diag = self.train_epoch(trainloader)
            accuracy, test_loss, test_on_diag, test_off_diag = self.evaluate(testloader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            train_on_diags.append(train_on_diag)
            train_off_diags.append(train_off_diag)
            test_on_diags.append(test_on_diag)
            test_off_diags.append(test_off_diag)

            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()
            print(
                f"Epoch: {i + 1}, Train_loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, Learning Rate: {current_lr}"
                )
            if save_model_evry_n_epochs > 0 and (i + 1) % save_model_evry_n_epochs == 0 and i + 1 != config["epochs"]:
                print("> Save checkpoint at epoch", i + 1)
                save_checkpoint(self.exp_name, self.model, i + 1)
        # save the experiment
        save_experiment(
            self.exp_name, base_dir, config, self.model, train_losses, test_losses, accuracies
            )
        
        # check loss
        print("on_diags")
        plot_progress(
            "on_diags", train_on_diags, test_on_diags, config["epochs"], base_dir=base_dir
            )
        print("off_diags")
        plot_progress(
            "on_diags", train_off_diags, test_off_diags, config["epochs"], base_dir=base_dir
            )
        # ToDo base_dirを追加


    def train_epoch(self, trainloader):
        """ train the model for one epoch """
        self.model.train()
        total_loss = 0
        total_on_diag = 0
        total_off_diag = 0
        for y1, y2 in trainloader:
            # batchをdeviceへ
            y1, y2 = y1.to(self.device), y2.to(self.device)
            # 勾配を初期化
            self.optimizer.zero_grad()
            # forward / loss
            loss, on_diag, off_diag = self.model(y1, y2) # attentionもNoneで返るので
            # backpropagation
            loss.backward()
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # 外部からさわれるように
            # パラメータ更新
            self.optimizer.step()
            total_loss += loss.item()
            total_on_diag += on_diag.item() #
            total_off_diag += off_diag.item() #

        return total_loss / len(trainloader.dataset), total_on_diag / len(trainloader.dataset), total_off_diag / len(trainloader.dataset)# 全データセットのうちのいくらかという比率になっている
    

    @torch.no_grad()
    def evaluate(self, testloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total_on_diag = 0
        total_off_diag = 0
        with torch.no_grad():
            for y1, y2 in testloader:
                # batchをdeviceへ
                y1, y2 = y1.to(self.device), y2.to(self.device)
                # loss
                loss, on_diag, off_diag = self.model(y1, y2)
                total_loss += loss.item()
                total_on_diag += on_diag.item() #
                total_off_diag += off_diag.item() #
        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        avg_on_diag = total_on_diag / len(testloader.dataset)
        avg_off_diag = total_off_diag / len(testloader.dataset)
        return accuracy, avg_loss, avg_on_diag, avg_off_diag