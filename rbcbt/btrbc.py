# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

ihvit module

@author: tadahaya
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from typing import Tuple
import yaml

from tqdm.auto import tqdm

#from .src.models import *
from .src.models.vit import VitForClassification
from .src.trainer import Trainer
from .src.data_handler import  prep_smeardata_bt, prep_validdataset_lst, get_clstoken, get_dr_feature
from .image_aug import SSLTransform
from .src.barlow import BarlowTwins

from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR

import torch_optimizer as optim_ext #250424追加

class BTRBC:
    def __init__(
            self, config_path: str
            ):
        # configの読み込み
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        self.config["config_path"] = config_path
        self.input_path = None
        self.input_path2 = None
        self.model = None
        self.backbone = None


    def load_model(self, model_path: str, config_path: str=None):
        """ モデルの読み込み """
        if config_path is not None:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
            self.config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            self.config["config_path"] = config_path
        self.model = VitForClassification(self.config)
        self.model.load_state_dict(torch.load(model_path))


    def prep_smeardata_bt(
            self, exp_name: str=None, input_path: str=None,
            num_rbc=2000, show_imagedata=True,
            transform: Tuple[transforms.Compose, transforms.Compose]=(None, None),
            num_workers=2, pin_memory=True,
            dataset_save=None
            ):
        """ dataの読み込み """
        if exp_name is None:
            exp_name = "exp"
        self.config["exp_name"] = exp_name
        self.input_path = input_path
        ssltf = SSLTransform(crop_size=self.config["crop_size"])
        train_loader, test_loader, classes = prep_smeardata_bt(
            image_path=input_path, num_rbc=num_rbc, show_imagedata=show_imagedata,
            batch_size=self.config["batch_size"], 
            transform=transform,
            ssl_transform = ssltf,
            shuffle=(True, False), 
            num_workers=num_workers, pin_memory=pin_memory, 
            dataset_save=dataset_save
            )
        return train_loader, test_loader, classes   
    
    
    def prep_smeardata_ds(
            self, image_path,
            num_image=2000, splitn=5, ditect_type="rbc",
            show_imagedata=True,
            dataset_save=None
            ):
        if os.path.exists(dataset_save+'ds_dataset_lst.pickle'): #
            with open(dataset_save+'ds_dataset_lst.pickle', 'rb') as f:
                dataset_lst = pickle.load(f)
        else:
            dataset_lst = prep_validdataset_lst(image_path, num_image=num_image, splitn=splitn, ditect_type=ditect_type, show_imagedata=show_imagedata)
            with open(dataset_save+'ds_dataset_lst.pickle', 'wb') as f:
                pickle.dump(dataset_lst, f)

        return dataset_lst 


    def fit(self, train_loader, test_loader, classes, btconfig={}, warmup=True):
        """ training """
        # モデル等の準備 (Classの有無でBTとViTを切り替え)
        self.latent_id = btconfig["latent_id"]
        if len(btconfig) != 0:
            self.backbone = VitForClassification(self.config)
            self.model = BarlowTwins(self.backbone, self.latent_id, btconfig["projection_sizes"], btconfig["lambd"], scale_factor=btconfig["scale_factor"])
        else:
            self.model = VitForClassification(self.config)

        # CosineAnnealingLR + warmup の組み合わせ
        if warmup:
            # === ここから ===
            # Optimizer の定義
            optimizer = optim.AdamW(self.model.parameters(), lr=self.config["lr"], weight_decay=1e-2)

            # ウォームアップとコサイン減衰を組み合わせる設定
            num_epochs = self.config["epochs"]
            num_training_steps = len(train_loader) * num_epochs
            num_warmup_steps = int(0.1 * num_training_steps)  # 例: 全体の10%をウォームアップにする

            # ウォームアップスケジューラー
            warmup_scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
            )

            # コサイン減衰スケジューラー (ウォームアップ後に使う)
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - (num_warmup_steps / len(train_loader)))

            # スケジューラーを連結する
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[num_warmup_steps]
            )
            # === ここまで ===

        else:
            optimizer = optim.AdamW(self.model.parameters(), lr=self.config["lr"], weight_decay=1e-2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        loss_fn = nn.CrossEntropyLoss()
        trainer = Trainer(
            self.config, self.model, optimizer, scheduler, loss_fn, self.config["exp_name"], device=self.config["device"]
            )
        # training
        trainer.train(
            train_loader, test_loader, classes, save_model_evry_n_epochs=self.config["save_model_every"]
            )
        
        if self.input_path2 is None:
            accuracy, avg_loss, avg_on_diag, avg_off_diag = trainer.evaluate(test_loader)
            print(f"Accuracy: {accuracy} // Average Loss: {avg_loss}")


    def model_vaild(self, model, dataset_lst, f_type="mean+max"):
        sorted_features = get_clstoken(dataset_lst, model, self.config["device"], f_type=f_type, latent_id=self.latent_id, batch_size=64)
        pca_feature, sample_ind = get_dr_feature(sorted_features)
        return  pca_feature, sample_ind
