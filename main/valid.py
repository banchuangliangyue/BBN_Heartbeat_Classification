import _init_paths
from net import Network
from config import cfg, update_config
from dataset import *
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from core.evaluate import FusionMatrix
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, \
    roc_auc_score, roc_curve, auc, classification_report, confusion_matrix

import matplotlib.pyplot as plt
plt.switch_backend('agg')


def parse_args():
    parser = argparse.ArgumentParser(description="BBN evaluation")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=True,
        default="configs/cifar10.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args

def valid_model(dataLoader, model, cfg, logger, device, writer, num_classes):
    result_list = []
    pbar = tqdm(total=len(dataLoader))
    model.eval()
    top1_count, top2_count, top3_count, index, fusion_matrix = (
        [],
        [],
        [],
        0,
        FusionMatrix(num_classes),
    )

    func = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        y_true = np.array([]).reshape((0, 1))
        y_pred = np.array([]).reshape((0, 1))
        feature = np.array([]).reshape((0, 2560))

        for i, (image, image_labels, meta) in enumerate(dataLoader):
            image = image.to(device)
            fea, output = model(image)
            result = func(output)
            # _, top_k = result.topk(5, 1, True, True)
            _, top_k = result.topk(4, 1, True, True)
            score_result = result.cpu().numpy()
            fusion_matrix.update(score_result.argmax(axis=1), image_labels.numpy())
            pred = score_result.argmax(axis=1).reshape(image.shape[0], 1)
            label = image_labels.numpy().reshape(image.shape[0], 1)

            fea = fea.detach().cpu().numpy().reshape(image.shape[0], -1)
            feature = np.concatenate((feature, fea), axis=0)

            y_pred = np.concatenate((y_pred, pred), axis=0)
            y_true = np.concatenate((y_true, label), axis=0)
            topk_result = top_k.cpu().tolist()
            if not "image_id" in meta:
                meta["image_id"] = [0] * image.shape[0]
            image_ids = meta["image_id"]
            for i, image_id in enumerate(image_ids):
                result_list.append(
                    {
                        "image_id": image_id,
                        "image_label": int(image_labels[i]),
                        "top_3": topk_result[i],
                    }
                )
                top1_count += [topk_result[i][0] == image_labels[i]]
                top2_count += [image_labels[i] in topk_result[i][0:2]]
                top3_count += [image_labels[i] in topk_result[i][0:3]]
                index += 1
            now_acc = np.sum(top1_count) / index
            pbar.set_description("Now Top1:{:>5.2f}%".format(now_acc * 100))
            pbar.update(1)

        # writer.add_embedding(feature, metadata=y_true, )

    logger.info('y_true shape: {}'.format(y_true.shape))
    logger.info('y_pred shape: {}'.format(y_pred.shape))
    logger.info('========== confusion matrix ==========\n')
    logger.info(classification_report(y_true, y_pred, target_names=['N', 'S', 'V', 'F'], digits=4))



    logger.info(confusion_matrix(y_true, y_pred))
    top1_acc = float(np.sum(top1_count) / len(top1_count))
    top2_acc = float(np.sum(top2_count) / len(top1_count))
    top3_acc = float(np.sum(top3_count) / len(top1_count))
    f1_macro = fusion_matrix.get_f1_per_class()
    logger.info(
        "\nTop1:{:>5.2f}%  Top2:{:>5.2f}%  Top3:{:>5.2f}%  F1_macro:{:.4f}".format(
            top1_acc * 100, top2_acc * 100, top3_acc * 100, np.mean(f1_macro)
        )
    )
    pbar.close()



if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args)

    test_set = eval(cfg.DATASET.DATASET)("valid", cfg)
    num_classes = test_set.get_num_classes()
    device = torch.device("cpu" if cfg.CPU_MODE else "cuda")
    model = Network(cfg, mode="test", num_classes=num_classes)

    model_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "models")
    model_file = cfg.TEST.MODEL_FILE
    if "/" in model_file:
        model_path = model_file
    else:
        model_path = os.path.join(model_dir, model_file)
    model.load_model(model_path)

    if cfg.CPU_MODE:
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model).cuda()

    testLoader = DataLoader(
        test_set,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.TEST.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )
    valid_model(testLoader, model, cfg, device, num_classes)
