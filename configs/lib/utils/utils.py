import logging
import time
import os

import torch
from utils.lr_scheduler import WarmupMultiStepLR
from net import Network
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
import math
import numpy as np
import random


def create_logger(cfg, run_id=0):
    dataset = cfg.DATASET.DATASET
    net_type = cfg.BACKBONE.TYPE
    module_type = cfg.MODULE.TYPE
    log_dir = os.path.join(cfg.OUTPUT_DIR+'_exp_'+str(run_id), cfg.NAME, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    time_str = time.strftime("%Y-%m-%d-%H-%M")
    # log_name = "{}_{}_{}_{}.log".format(dataset, net_type, module_type, time_str)
    log_name = "{}_{}_{}_exp_{}.log".format(dataset, net_type, module_type, str(run_id))
    log_file = os.path.join(log_dir, log_name)
    # set up logger
    print("=> creating log {}".format(log_file))
    head = "%(asctime)-15s %(message)s"
    logging.basicConfig(filename=str(log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger("").addHandler(console)

    logger.info("----------Cfg is set as follow:---------")
    logger.info(cfg)
    logger.info("----------------------------------------")
    return logger, log_file


def get_optimizer(cfg, model):
    base_lr = cfg.TRAIN.OPTIMIZER.BASE_LR
    params = []

    for name, p in model.named_parameters():
        if p.requires_grad:
            params.append({"params": p})

    if cfg.TRAIN.OPTIMIZER.TYPE == "SGD":
        optimizer = torch.optim.SGD(
            params,
            lr=base_lr,
            momentum=cfg.TRAIN.OPTIMIZER.MOMENTUM,
            weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY,
            nesterov=True,
        )
    elif cfg.TRAIN.OPTIMIZER.TYPE == "ADAM":
        optimizer = torch.optim.Adam(
            params,
            lr=base_lr,
            betas=(0.9, 0.999),
            weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY,
        )
    return optimizer


def get_scheduler(cfg, optimizer):
    if cfg.TRAIN.LR_SCHEDULER.TYPE == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            cfg.TRAIN.LR_SCHEDULER.LR_STEP,
            gamma=cfg.TRAIN.LR_SCHEDULER.LR_FACTOR,
        )
    elif cfg.TRAIN.LR_SCHEDULER.TYPE == "cosine":
        if cfg.TRAIN.LR_SCHEDULER.COSINE_DECAY_END > 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.TRAIN.LR_SCHEDULER.COSINE_DECAY_END, eta_min=1e-4
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.TRAIN.MAX_EPOCH, eta_min=1e-5###original: eta_min=1e-4
            )
    elif cfg.TRAIN.LR_SCHEDULER.TYPE == "warmup":
        scheduler = WarmupMultiStepLR(
            optimizer,
            cfg.TRAIN.LR_SCHEDULER.LR_STEP,
            gamma=cfg.TRAIN.LR_SCHEDULER.LR_FACTOR,
            warmup_epochs=cfg.TRAIN.LR_SCHEDULER.WARM_EPOCH,
        )
    else:
        raise NotImplementedError("Unsupported LR Scheduler: {}".format(cfg.TRAIN.LR_SCHEDULER.TYPE))

    return scheduler


def get_model(cfg, num_classes, device, logger):
    model = Network(cfg, mode="train", num_classes=num_classes)

    if cfg.BACKBONE.FREEZE == True:
        model.freeze_backbone()
        logger.info("Backbone has been freezed")

    if cfg.CPU_MODE:
        model = model.to(device)
    else:
        model = model.to(device)
        # model = torch.nn.DataParallel(model).cuda()

    return model

def get_category_list(annotations, num_classes, cfg):
    num_list = [0] * num_classes
    cat_list = []
    print("\nWeight List has been produced\n")
    for anno in annotations:
        category_id = anno["category_id"]
        num_list[category_id] += 1
        cat_list.append(category_id)
    return num_list, cat_list

def plot_curve(train_results_img_save_path, train_statistic, val_statistic, i=0):
    iters = [i + 1 for i in range(0, len(train_statistic['pt_loss']))]

    # plt.figure(figsize=(10, 10))
    # plt.plot(iters, train_statistic['w_diff_loss'], color='red', label='weight_diff_loss')
    # plt.plot(iters, train_statistic['p1_acc'], color='green', label='train_p1_acc')
    # plt.plot(iters, train_statistic['p2_acc'], color='blue', label='train_p2_acc')
    # plt.xlabel('epoch')
    # plt.legend(loc="best")
    # plt.grid(True)
    # plt.savefig(train_results_img_save_path + 'atda_train_process_' + str(i) +'_.png', bbox_inches='tight')

    plt.figure(figsize=(30, 10))
    plt.subplot(131)
    # plt.plot(iters, train_statistic['p1_loss'], color='red', label='train_p1_loss')
    # plt.plot(iters, val_statistic['p1_loss'], color='darkred', label='val_p1_loss')
    # plt.plot(iters, train_statistic['p2_loss'], color='green', label='train_p2_loss')
    # plt.plot(iters, val_statistic['p2_loss'], color='lightgreen', label='val_p2_loss')
    plt.plot(iters, train_statistic['pt_loss'], color='blue', label='train_pt_loss')
    plt.plot(iters, val_statistic['pt_loss'], color='slateblue', label='val_pt_loss')
    # plt.plot(iters, train_statistic['w_diff_loss'], color='pink', label='weight_diff_loss')
    plt.xlabel('epoch')
    plt.legend(loc="best")
    plt.grid(True)

    plt.subplot(132)
    # plt.plot(iters, train_statistic['p1_acc'], color='red', label='train_p1_acc')
    # plt.plot(iters, val_statistic['p1_acc'], color='darkred', label='val_p1_acc')
    # plt.plot(iters, train_statistic['p2_acc'], color='green', label='train_p2_acc')
    # plt.plot(iters, val_statistic['p2_acc'], color='lightgreen', label='val_p2_acc')
    plt.plot(iters, train_statistic['pt_acc'], color='blue', label='train_pt_acc')
    plt.plot(iters, val_statistic['pt_acc'], color='slateblue', label='val_pt_acc')
    plt.xlabel('epoch')
    plt.legend(loc="best")
    plt.grid(True)

    plt.subplot(133)
    plt.plot(iters, val_statistic['val_f1'], color='red', label='val_f1')
    plt.xlabel('epoch')
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(train_results_img_save_path + '/atda_loss_acc_f1_' + str(i) + '_.png', bbox_inches='tight')


class AlexNetforEcg_DS1_to_DS2(nn.Module):
    '''input tensor size:(None,1,3,128)'''
    def __init__(self):
        super(AlexNetforEcg_DS1_to_DS2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 5), padding=(0, 0)),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),#（N,64,1,62)


            # nn.Conv2d(64, 192, kernel_size=(1, 5), padding=(0, 2)),
            nn.Conv2d(64, 128, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),#(N,192,1,30)

            nn.Conv2d(128, 256, kernel_size=(1, 5), padding=(0, 2)),  # (N,_,1,30)
            # nn.Conv2d(192, 256, kernel_size=(1, 5), padding=(0, 2)),#(N,_,1,30)
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 15, 256 * 10),
            # nn.Linear(256 * 14, 256 * 10),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),#0.3
            # nn.BatchNorm1d(256 * 10),
        )


        self.classifier = nn.Sequential(
            nn.Linear(256 * 10, 256 * 5),
            # nn.Linear(256 *5, 256 * 1),
            # nn.BatchNorm1d(256 * 5),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # 0.5
            nn.Linear(256 * 5, 4),

        )

    def forward(self, x):
        x = self.features(x)
        # print("feature size:", x.size())
        # x = x.view(x.size(0), self.num_flat_features(x))
        # print(x.size())

        x = x.view(x.size(0), -1)
        fea = self.fc(x)

        y = self.classifier(fea)
        return fea, y



class EcgClassifier(nn.Module):
    """Feature classifier class for MNIST -> MNIST-M experiment in ATDA."""

    def __init__(self, dropout_keep=None, num_classes=5):
        """Init classifier."""
        super(EcgClassifier, self).__init__()

        self.dropout_keep = dropout_keep

        self.classifier = nn.Sequential(
            nn.Linear(256 * 10, 256 * 5),
            # nn.Linear(256 *5, 256 * 1),
            # nn.BatchNorm1d(256 * 5),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_keep),  # 0.5
            nn.Linear(256 * 5, num_classes),

        )


    def forward(self, x):
        """Forward classifier."""
        out = self.classifier(x)
        return out

