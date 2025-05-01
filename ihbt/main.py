# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

main file

@author: tadahaya
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import argparse
import yaml

from tqdm.auto import tqdm

from .src.arguments import get_args
from .src.models import *
from .src.trainer import Trainer
from .src.data_handler import prep_data, prep_test


def get_args():
    """ 引数の取得 """
    parser = argparse.ArgumentParser(description="Yaml file for training")
    parser.add_argument("--config_path", type=str, required=True, help="Yaml file for training")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--input_path", type=str, default=None, help="input data path")
    parser.add_argument("--input_path2", type=str, default=None, help="input data path, test data")
    args = parser.parse_args()
    return args


def test():
    """ CIFAR10を使ったテスト用 """
    # argsの取得
    args = get_args()
    # yamlの読み込み
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["config_path"] = args.config_path
    config["exp_name"] = args.exp_name
    # dataの読み込み
    trainloader, testloader, classes = prep_test(batch_size=config["batch_size"])
    # モデル等の準備
    model = VitForClassification(config)
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-2) # AdamW使っている
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(config, model, optimizer, loss_fn, args.exp_name, device=config["device"])
    trainer.train(
        trainloader, testloader, classes, save_model_evry_n_epochs=config["save_model_every"]
        )


def main():
    # argsの取得
    args = get_args()
    # input_pathのチェック
    if args.input_path is None:
        raise ValueError("!! Give input_path !!")
    # yamlの読み込み
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["config_path"] = args.config_path
    config["exp_name"] = args.exp_name
    # dataの読み込み
    train_loader, test_loader, classes = prep_data(
        image_path=(args.input_path, args.input_path2), 
        batch_size=config["batch_size"], transform=(None, None), shuffle=(True, False)
        )
    # モデル等の準備
    model = VitForClassification(config)
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-2) # AdamW使っている
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(config, model, optimizer, loss_fn, args.exp_name, device=config["device"])
    trainer.train(
        train_loader, test_loader, classes, save_model_evry_n_epochs=config["save_model_every"]
        )
    if args.input_path2 is None:
        accuracy, avg_loss = trainer.evaluate(test_loader)
        print(f"Accuracy: {accuracy} // Average Loss: {avg_loss}")


def main2():
    # argsの取得
    args = get_args()
    # input_pathのチェック
    if args.input_path is None:
        raise ValueError("!! Give input_path !!")
    # yamlの読み込み
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["config_path"] = args.config_path
    config["exp_name"] = args.exp_name
    # dataの読み込み
    # dataの形状を決めて置く
    # 現状では, data, label, dim
    dataset = np.load(args.input_path, allow_pickle=True)
    data = dataset["data"]
    label = dataset["label"]
    dim = dataset["dim"]
    idx = int(input.shape[0] * 0.9)
    input = np.transpose(input, [0,3,1,2]) # nhwc -> nchw
    output = np.transpose(output, [0,3,1,2]) # nhwc -> nchw
    input = torch.tensor(input).float()
    output = torch.tensor(output).float()
    tfn_train, tfn_test = None, None # Noneで初期化
    if dim == 1:
        tfn_train = transforms.Compose([
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.0), fill=255),
            transforms.GaussianBlur(kernel_size=3, sigma=(1.0, 2.0))
            ])
        tfn_test = transforms.Compose([
            transforms.GaussianBlur(kernel_size=3, sigma=1.0)
            ])
        # 1Dの場合はx軸方向のみ不変
    elif dim == 2:
        tfn_train = transforms.Compose([
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), fill=255),
            transforms.GaussianBlur(kernel_size=3, sigma=(1.0, 2.0))
            ])
        tfn_test = transforms.Compose([
            transforms.GaussianBlur(kernel_size=3, sigma=1.0)
            ])
        # 2Dの場合はx,y両方不変
    train_loader, test_loader = prep_data(
        data[:idx], label[:idx], data[idx:], label[idx:],
        batch_size=config["batch_size"], transform=(tfn_train, tfn_test)
        )
    # モデル等の準備
    model = VitForClassification(config)
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-2) # AdamW使っている
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(config, model, optimizer, loss_fn, args.exp_name, device=config["device"])
    trainer.train(
        train_loader, test_loader, save_model_evry_n_epochs=config["save_model_every"]
        )


if __name__ == "__main__":
    main()