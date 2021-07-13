import _init_paths
import sys
from loss import *
from dataset import *
from config import cfg, update_config
from utils.utils import (
    create_logger,
    get_optimizer,
    get_scheduler,
    get_model,
    get_category_list,
    plot_curve,
    AlexNetforEcg_DS1_to_DS2,
)
from core.function import train_model, valid_model
from core.combiner import Combiner

import torch
import os, shutil
from torch.utils.data import DataLoader
import argparse
import warnings
import click
import random
from net import Network
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
# import ast
import numpy as np
import valid

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def parse_args():
    parser = argparse.ArgumentParser(description="codes for BBN")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="configs/cifar10.yaml",
        type=str,
    )
    parser.add_argument(
        "--ar",
        help="decide whether to use auto resume",
        # type= ast.literal_eval,
        type=str2bool,
        dest='auto_resume',
        required=False,
        default=False,
    )

    parser.add_argument(
        "opts",
        help="modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument("--seed", dest="SEED", type=int, metavar='<int>', default=666)
    parser.add_argument('--gpu', dest="GPU", type=str, default=7, help='cuda_visible_devices')
    parser.add_argument("--run_id", dest="id", type=int, metavar='<int>', default=0)
    parser.add_argument("--alpha", dest="mix_alpha", type=float, metavar='<float>', default=1.0)
    parser.add_argument("--m", dest="margin", type=float, metavar='<float>', default=1.0)
    parser.add_argument("--lambda", dest="_lambda", type=float, metavar='<float>', default=0.001)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    update_config(cfg, args)
    logger, log_file = create_logger(cfg, args.id)
    warnings.filterwarnings("ignore")
    logger.info('\n---------------{}  exp:{}---------------\n'.format(__file__, args.id))
    logger.info(args.__str__())

    cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    cudnn.deterministic = True
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed(args.SEED)
    torch.cuda.manual_seed_all(args.SEED)
    random.seed(args.SEED)
    np.random.seed(args.SEED)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.GPU)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    auto_resume = args.auto_resume

    train_record_list = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203,
                         205, 207, 208, 209, 215, 220, 223, 230]
    test_record_list = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214,
                        219, 221, 222, 228, 231, 232, 233, 234]
    # val_record_list = np.random.choice(train_record_list, 4, replace=False)
    # train_record_list = [i for i in train_record_list if i not in val_record_list]
    print('train record list:\n', train_record_list)
    # print('val record list:\n', val_record_list)
    print('test record list:\n', test_record_list)

    train_set = eval(cfg.DATASET.DATASET)("train", cfg, dataset_name='DS1', ecg_records=train_record_list)
    # val_set = eval(cfg.DATASET.DATASET)("val", cfg, dataset_name='DS1', ecg_records=val_record_list)
    test_set = eval(cfg.DATASET.DATASET)("test", cfg, dataset_name='DS2', ecg_records=test_record_list)

    annotations = train_set.get_annotations(train_set.sample_y)
    num_classes = train_set.get_num_classes()

    num_class_list, cat_list = get_category_list(annotations, num_classes, cfg)

    para_dict = {
        "num_classes": num_classes,
        "num_class_list": num_class_list,
        "cfg": cfg,
        "device": device,
    }
    print('num_class_list: {}\n'.format(num_class_list))

    criterion = eval(cfg.LOSS.LOSS_TYPE)(para_dict=para_dict)
    epoch_number = cfg.TRAIN.MAX_EPOCH

    # ----- BEGIN MODEL BUILDER -----
    model = AlexNetforEcg_DS1_to_DS2().to(device)
    combiner = Combiner(cfg, device)
    optimizer = get_optimizer(cfg, model)
    # scheduler = get_scheduler(cfg, optimizer)
    print('\nNetwork Architecture:')
    for i in model.state_dict():
        print(i)
    print("\nThe entire model {} has {} parameters in total!!\n".format(cfg.BACKBONE.TYPE,
                                                                  sum(x.numel() for x in model.parameters())))

    # ----- END MODEL BUILDER -----

    trainLoader = DataLoader(
        train_set,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=True
    )

    validLoader = DataLoader(
        test_set,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.TEST.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    # testLoader = DataLoader(
    #     test_set,
    #     batch_size=cfg.TEST.BATCH_SIZE,
    #     shuffle=False,
    #     num_workers=cfg.TEST.NUM_WORKERS,
    #     pin_memory=cfg.PIN_MEMORY,
    # )
    # close loop
    model_dir = os.path.join(cfg.OUTPUT_DIR+'_exp_'+str(args.id), cfg.NAME, "models")
    code_dir = os.path.join(cfg.OUTPUT_DIR+'_exp_'+str(args.id), cfg.NAME, "codes")
    train_results_img_save_path = os.path.join(cfg.OUTPUT_DIR + '_exp_' + str(args.id), cfg.NAME, "loss_acc_curve")
    if not (os.path.exists(train_results_img_save_path)):
        os.makedirs(train_results_img_save_path)

    tensorboard_dir = (
        os.path.join(cfg.OUTPUT_DIR+'_exp_'+str(args.id), cfg.NAME, "tensorboard")
        if cfg.TRAIN.TENSORBOARD.ENABLE
        else None
    )

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
        shutil.rmtree(code_dir)
        if tensorboard_dir is not None and os.path.exists(tensorboard_dir):
            shutil.rmtree(tensorboard_dir)
    print("=> output model will be saved in {}".format(model_dir))
    this_dir = os.path.dirname(__file__)
    ignore = shutil.ignore_patterns(
        "*.pyc", "*.so", "*.out", "*pycache*", "*.pth", "*build*", "*output*", "*datasets*"
    )
    shutil.copytree(os.path.join(this_dir, ".."), code_dir, ignore=ignore)

    if tensorboard_dir is not None:
        # dummy_input = torch.rand((1, 1) + cfg.INPUT_SIZE).to(device)
        dummy_input = torch.rand((1, 1) + cfg.INPUT_SIZE).to(device)
        writer = SummaryWriter(log_dir=tensorboard_dir)
        # writer.add_graph(model if cfg.CPU_MODE else model.module, (dummy_input,))
        writer.add_graph(model, (dummy_input,))
    else:
        writer = None

    best_result, best_epoch, start_epoch = 0, 0, 1


    logger.info(
        "-------------------Train start :{}  {}  {}-------------------".format(
            cfg.BACKBONE.TYPE, cfg.MODULE.TYPE, cfg.TRAIN.COMBINER.TYPE
        )
    )
    train_statistic = {'total_loss': [], 'p1_loss': [], 'p2_loss': [], 'pt_loss': [], 'w_diff_loss': [],
                       'p1_acc': [], 'p2_acc': [], 'pt_acc': [], 'new_labeled_samples': []}
    val_statistic = {'total_loss': [], 'p1_loss': [], 'p2_loss': [], 'pt_loss': [], 'w_diff_loss': [],
                     'p1_acc': [], 'p2_acc': [], 'pt_acc': [], 'val_f1': []}

    for epoch in range(start_epoch, epoch_number + 1):
        # scheduler.step()
        train_acc, train_loss = train_model(
            trainLoader,
            model,
            epoch,
            epoch_number,
            optimizer,
            combiner,
            criterion,
            cfg,
            logger,
            args=args,
            device=device,
            writer=writer,
        )


        print('base_lr:{:.5f}  current lr:{:.5f}  random_seed:{}  alpha:{}  margin:{}  lambda:{:.4f}\n'
              .format(cfg.TRAIN.OPTIMIZER.BASE_LR, optimizer.param_groups[0]['lr'], args.SEED,
                      args.mix_alpha, args.margin, args._lambda))
        # print('x_train shape: {}   y_train shape: {}'.format(train_set.sample_x.shape, train_set.sample_y.shape))

        lr_dict = {'lr':optimizer.param_groups[0]['lr']}
        loss_dict, acc_dict, f1_dict = {"train_loss": train_loss}, {"train_acc": train_acc}, {}
        train_statistic['pt_loss'].append(train_loss)
        train_statistic['pt_acc'].append(train_acc)

        if cfg.VALID_STEP != -1 and epoch % cfg.VALID_STEP == 0:
            valid_acc, valid_loss, valid_f1 = valid_model(
                validLoader, epoch, model, cfg, criterion, logger, device, writer=writer
            )
            # print('x_val shape: {}   y_val shape: {}'.format(test_set.sample_x.shape, test_set.sample_y.shape))
            loss_dict["valid_loss"], acc_dict["valid_acc"], f1_dict['valid_f1'] = valid_loss, valid_acc, valid_f1
            val_statistic['pt_loss'].append(valid_loss)
            val_statistic['pt_acc'].append(valid_acc)
            val_statistic['val_f1'].append(valid_f1)
            if valid_f1 > best_result:
            # if valid_acc > best_result:
            #     best_result, best_epoch = valid_acc, epoch
                best_result, best_epoch = valid_f1, epoch
                torch.save({
                        'state_dict': model.state_dict(),
                        'epoch': epoch,
                        'best_result': best_result,
                        'best_epoch': best_epoch,
                        # 'scheduler': scheduler.state_dict(),
                        'optimizer': optimizer.state_dict(),
                }, os.path.join(model_dir, "best_model.pth")
                )
            logger.info(
                # "--------------Best_Epoch:{:>3d}    Best_Acc:{:>5.2f}%--------------".format(
                #     best_epoch, best_result * 100
                "Best_Epoch:{:>3d}    Best_F1:{:>5.3f}".format(
                    best_epoch, best_result
                )
            )
        if cfg.TRAIN.TENSORBOARD.ENABLE:
            writer.add_scalars("scalar/acc", acc_dict, epoch)
            writer.add_scalars("scalar/loss", loss_dict, epoch)
            writer.add_scalars("scalar/f1", f1_dict, epoch)
            writer.add_scalars("scalar/lr", lr_dict, epoch)


    if cfg.TRAIN.TENSORBOARD.ENABLE:
        writer.close()

    plot_curve(train_results_img_save_path, train_statistic, val_statistic, args.id)
    logger.info(
        "-------------------Training Finished :{}-------------------".format(cfg.NAME)
    )

    logger.info(
        "-------------------Testing Start :{}-------------------".format(cfg.NAME)
    )

    num_classes = test_set.get_num_classes()
    # model = Network(cfg, mode="test", num_classes=num_classes)
    model = AlexNetforEcg_DS1_to_DS2().to(device)

    model_dir = os.path.join(cfg.OUTPUT_DIR+'_exp_'+str(args.id), cfg.NAME, "models")
    model_file = cfg.TEST.MODEL_FILE
    if "/" in model_file:
        model_path = model_file
    else:
        model_path = os.path.join(model_dir, model_file)
    print('\nSaved model path: {}'.format(model_path))
    print('Load model...')
    checkpoint = torch.load(model_path)
    print('best epoch:', checkpoint['best_epoch'])
    print('best val f1:', checkpoint['best_result'])
    model.load_state_dict(checkpoint['state_dict'])

    # model.load_model(model_path)

    # if cfg.CPU_MODE:
    #     model = model.to(device)
    # else:
    #     model = model.to(device)
    #     # model = torch.nn.DataParallel(model).cuda()


    valid.valid_model(validLoader, model, cfg, logger, device, writer, num_classes)
