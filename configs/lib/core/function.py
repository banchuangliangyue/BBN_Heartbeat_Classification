import _init_paths
from core.evaluate import accuracy, AverageMeter, FusionMatrix

import numpy as np
import torch
import time
from loss import ContrastiveLoss, triplet_loss


def train_model(
    trainLoader,
    model,
    epoch,
    epoch_number,
    optimizer,
    combiner,
    criterion,
    cfg,
    logger,
    args,
    device,
    writer,
):
    if cfg.EVAL_MODE:
        model.eval()
    else:
        model.train()

    combiner.reset_epoch(epoch)

    if cfg.LOSS.LOSS_TYPE in ['LDAMLoss', 'CSCE']:
        criterion.reset_epoch(epoch)

    start_time = time.time()
    number_batch = len(trainLoader)

    all_loss = AverageMeter()
    acc = AverageMeter()
    for i, (image, label, meta) in enumerate(trainLoader):
        cnt = label.shape[0]
        # loss, now_acc = combiner.forward(model, criterion, image, label, meta)

        image_a, image_b = image.to(device), meta["sample_image"].to(device)
        label_a, label_b = label.to(device), meta["sample_label"].to(device)

        lam = np.random.beta(args.mix_alpha, args.mix_alpha)
        index = np.random.permutation(image_a.shape[0])
        image_a_1, label_a_1 = image_a[index],  label_a[index]
        mixed_x = lam * image_a + (1 - lam) * image_a_1
        fea, pred = model(mixed_x)
        loss = lam * criterion(pred, label_a) + (1 - lam) * criterion(pred, label_a_1)
        now_result = torch.argmax(torch.nn.Softmax(dim=1)(pred), 1)
        now_acc = (
                lam * accuracy(now_result.cpu().numpy(), label_a.cpu().numpy())[0]
                + (1 - lam) * accuracy(now_result.cpu().numpy(), label_a_1.cpu().numpy())[0]
        )
        # fea, pred = model(image_a)
        # loss = criterion(pred, label_a)
               # + args._lambda*(triplet_loss(fea, label_a,margin=args.margin))
        # now_result = torch.argmax(torch.nn.Softmax(dim=1)(pred), 1)
        # now_acc = (
        #         accuracy(now_result.cpu().numpy(), label_a.cpu().numpy())[0]
        # )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_loss.update(loss.data.item(), cnt)
        acc.update(now_acc, cnt)

        # if i % cfg.SHOW_STEP == 0:
        #     pbar_str = "Epoch:{:>3d}  Batch:{:>3d}/{}  Batch_Loss:{:>5.3f}  Batch_Accuracy:{:>5.2f}%     ".format(
        #         epoch, i, number_batch, all_loss.val, acc.val * 100
        #     )
        #     logger.info(pbar_str)

    end_time = time.time()
    pbar_str = "\n###Epoch:{:>3d}/{}   Avg_Loss:{:>5.3f}   Epoch_Accuracy:{:>5.2f}%   Epoch_Time:{:>5.2f}min---".format(
        epoch, epoch_number, all_loss.avg, acc.avg * 100, (end_time - start_time) / 60
    )
    logger.info(pbar_str)
    return acc.avg, all_loss.avg


def valid_model(
    dataLoader, epoch_number, model, cfg, criterion, logger, device, writer, **kwargs
):
    model.eval()
    num_classes = dataLoader.dataset.get_num_classes()
    fusion_matrix = FusionMatrix(num_classes)

    # feature = np.array([]).reshape((0, 2560))
    # y_true = np.array([]).reshape((0, 1))
    with torch.no_grad():
        all_loss = AverageMeter()
        acc = AverageMeter()
        func = torch.nn.Softmax(dim=1)
        for i, (image, label, meta) in enumerate(dataLoader):
            image, label = image.to(device), label.to(device)

            # feature = model(image, feature_flag=True)
            # feature_a, feature_b = model(image, feature_flag=True)
            # output_a = model(0.5 * feature_a, classifier_cb=True)
            # output_b = model((1 - 0.5) * feature_b, classifier_rb=True)
            # output = output_a + output_b
            fea, output = model(image)
            loss = criterion(output, label)
            score_result = func(output)

            now_result = torch.argmax(score_result, 1)
            # print('label:', label)
            # print('pred:', now_result)
            all_loss.update(loss.data.item(), label.shape[0])
            fusion_matrix.update(now_result.cpu().numpy(), label.cpu().numpy())
            now_acc, cnt = accuracy(now_result.cpu().numpy(), label.cpu().numpy())
            acc.update(now_acc, cnt)
            # label = label.detach().cpu().numpy().reshape(label.shape[0], 1)
            # fea = fea.detach().cpu().numpy().reshape(label.shape[0], -1)
            # feature = np.concatenate((feature, fea), axis=0)
            # y_true = np.concatenate((y_true, label), axis=0)


        f1_cls = fusion_matrix.get_f1_per_class()
        pbar_str = "Valid: Epoch:{:>3d}  Valid_Loss:{:>5.3f}   Valid_Acc:{:>5.2f}%   Valid_F1_Macro:{:>5.3f}".\
                format(epoch_number, all_loss.avg, acc.avg * 100, np.mean(f1_cls))
        logger.info(pbar_str)

    return acc.avg, all_loss.avg, np.mean(f1_cls)
