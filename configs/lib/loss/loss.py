import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from itertools import combinations

class CrossEntropy(nn.Module):
    def __init__(self, para_dict=None):
        super(CrossEntropy, self).__init__()

    def forward(self, output, target):
        output = output
        loss = F.cross_entropy(output, target)
        return loss


class CSCE(nn.Module):

    def __init__(self, para_dict=None):
        super(CSCE, self).__init__()
        self.num_class_list = para_dict["num_class_list"]
        self.device = para_dict["device"]

        cfg = para_dict["cfg"]
        scheduler = cfg.LOSS.CSCE.SCHEDULER
        self.step_epoch = cfg.LOSS.CSCE.DRW_EPOCH

        if scheduler == "drw":
            self.betas = [0, 0.999999]
        elif scheduler == "default":
            self.betas = [0.999999, 0.999999]
        self.weight = None

    def update_weight(self, beta):
        effective_num = 1.0 - np.power(beta, self.num_class_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.num_class_list)
        self.weight = torch.FloatTensor(per_cls_weights).to(self.device)

    def reset_epoch(self, epoch):
        idx = (epoch-1) // self.step_epoch
        beta = self.betas[idx]
        self.update_weight(beta)

    def forward(self, x, target, **kwargs):
        return F.cross_entropy(x, target, weight= self.weight)


# The LDAMLoss class is copied from the official PyTorch implementation in LDAM (https://github.com/kaidic/LDAM-DRW).
class LDAMLoss(nn.Module):

    def __init__(self, para_dict=None):
        super(LDAMLoss, self).__init__()
        s = 30
        self.num_class_list = para_dict["num_class_list"]
        self.device = para_dict["device"]

        cfg = para_dict["cfg"]
        max_m = cfg.LOSS.LDAM.MAX_MARGIN
        m_list = 1.0 / np.sqrt(np.sqrt(self.num_class_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list).to(self.device)
        self.m_list = m_list
        assert s > 0

        self.s = s
        self.step_epoch = cfg.LOSS.LDAM.DRW_EPOCH
        self.weight = None

    def reset_epoch(self, epoch):
        idx = (epoch-1) // self.step_epoch
        betas = [0, 0.9999]
        effective_num = 1.0 - np.power(betas[idx], self.num_class_list)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.num_class_list)
        self.weight = torch.FloatTensor(per_cls_weights).to(self.device)

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.FloatTensor)
        index_float = index_float.to(self.device)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight= self.weight)


def ContrastiveLoss(s_fea, s_label, margin=10):
    s_norm = torch.norm(s_fea, p=2, dim=1, keepdim=True)
    s_fea = s_fea / s_norm
    # assert s_fea.shape == t_fea.shape

    '''source to target loss'''
    # intra_mask = torch.zeros(s_fea.shape[0], s_fea.shape[0]).to(s_label.device)
    intra_mask = torch.zeros(s_fea.shape[0], s_fea.shape[0]).to(s_label.device)
    for i in range(len(s_label)):
        idx = torch.nonzero(torch.eq(s_label, s_label[i]))
        intra_mask[i, idx] = 1.
    inter_mask = 1 - intra_mask
    # assert (intra_mask.sum() + inter_mask.sum()) == s_fea.shape[0] ** 2
    assert (intra_mask.sum() + inter_mask.sum()) == s_fea.shape[0] **2

    '''compute dist mat of s_feature and t_feature'''
    vecProd = torch.mm(s_fea, torch.transpose(s_fea, 1, 0))
    SqA = s_fea ** 2
    sumSqA = torch.sum(SqA, dim=1).view(1, -1)
    sumSqAEx = torch.transpose(sumSqA, 1, 0).repeat(1, vecProd.shape[1])

    SqB = s_fea ** 2
    sumSqB = torch.sum(SqB, dim=1).view(1, -1)
    sumSqBEx = sumSqB.repeat(vecProd.shape[0], 1)
    # print('sumSqAEx shape:', sumSqAEx)
    # print('sumSqBEx shape:', sumSqBEx)
    ###plus 1 for dist_mat,防止0开根号出现错误
    ##2020/6/7：之前没有加1
    SqED = sumSqBEx + sumSqAEx - 2 * vecProd
    SqED = torch.clamp(SqED, min=1e-20)
    # print(' vecProd shape:', vecProd)
    # print(' dist_mat shape:', SqED)
    SqED = SqED** 0.5
    dist_mat = SqED.to(s_label.device)

    m = margin * torch.ones(s_fea.shape[0], s_fea.shape[0]).to(s_label.device)
    intra_loss = ((intra_mask * dist_mat) ** 2).sum() / (intra_mask.sum() + 1)
    inter_loss = ((inter_mask * torch.max(torch.zeros(s_fea.shape[0], s_fea.shape[0]).to(s_label.device),
                                          (m - dist_mat))) ** 2).sum() / (inter_mask.sum() + 1)

    # print('inter loss :', inter_loss)
    # print('intra loss :', intra_loss)


    return inter_loss + intra_loss

def triplet_loss(feature, label, margin=1):
    # model.train()
    # emb = model(batch["X"].cuda())
    # y = batch["y"].cuda()
    emb =  feature
    y = label

    with torch.no_grad():
        triplets = get_triplets_minority(emb, y, margin)
        # print('triplets.shape:',triplets.shape)

    f_A = emb[triplets[:, 0]]
    f_P = emb[triplets[:, 1]]
    f_N = emb[triplets[:, 2]]

    ap_D = (f_A - f_P).pow(2).sum(1)  # .pow(.5)
    an_D = (f_A - f_N).pow(2).sum(1)  # .pow(.5)
    losses = F.relu(ap_D - an_D + margin)

    return losses.mean()

def get_triplets(embeddings, y, margin):
    # margin = 1
    D = pdist(embeddings)
    D = D.cpu()

    y = y.cpu().data.numpy().ravel()
    trip = []
    ap = np.arange(len(embeddings))
    for label in set(y):
        if label != 0:
            label_mask = (y == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                # continue
                # ap = [(label_indices[0],label_indices[0])]
                label_indices=np.array([label_indices[0],label_indices[0]])
            neg_ind = np.where(np.logical_not(label_mask))[0]


            ap = list(combinations(label_indices, 2))  # All anchor-positive pairs
            ap = np.array(ap)

            ap_D = D[ap[:, 0], ap[:, 1]]
            # if len(neg_ind)==0:
            #     continue
            # # GET HARD NEGATIVE
            # if np.random.rand() < 0.5:
            #   trip += get_neg_hard(neg_ind, hardest_negative,
            #                D, ap, ap_D, margin)
            # else:
            trip += get_neg_hard(neg_ind, random_neg, D, ap, ap_D, margin)

    if len(trip) == 0 :
        # ap = ap[0]
        # if len(neg_ind) > 0:
        #     trip.append([ap[0], ap[1], neg_ind[0]])
        # else:
        #     trip.append([ap[0], ap[1], ap[0]])
        trip.append([0, 1, 0])

    trip = np.array(trip)

    return torch.LongTensor(trip)


def get_triplets_minority(embeddings, y, margin=1):
    # margin = 1
    D = pdist(embeddings)
    D = D.cpu()

    y = y.cpu().data.numpy().ravel()
    trip = []
    # ap = np.arange(len(embeddings))
    num_per_cls = {}
    for label in set(y):
        label_mask = (y == label)
        label_indices = np.where(label_mask)[0]
        num_per_cls[label] = len(label_indices)

    minority_cls = min(num_per_cls, key=num_per_cls.get)


    label_mask = (y == minority_cls)
    label_indices = np.where(label_mask)[0]
    # print('minority_cls:', minority_cls)
    # print('num:', len(label_indices))
    if len(label_indices) < 2:
        # continue
        # ap = [(label_indices[0],label_indices[0])]
        label_indices=np.array([label_indices[0],label_indices[0]])
    neg_ind = np.where(np.logical_not(label_mask))[0]

    ap = list(combinations(label_indices, 2))  # All anchor-positive pairs
    ap = np.array(ap)

    ap_D = D[ap[:, 0], ap[:, 1]]
    # if len(neg_ind)==0:
    #     continue
    # # GET HARD NEGATIVE
    # if np.random.rand() < 0.5:
    #   trip += get_neg_hard(neg_ind, hardest_negative,
    #                D, ap, ap_D, margin)
    # else:
    trip += get_neg_hard(neg_ind, random_neg, D, ap, ap_D, margin)

    if len(trip) == 0 :
        ap = ap[0]
        if len(neg_ind) > 0:
            trip.append([ap[0], ap[1], neg_ind[0]])
        else:
            trip.append([ap[0], ap[1], ap[0]])

    trip = np.array(trip)

    return torch.LongTensor(trip)

def pdist(vectors):
    D = -2 * vectors.mm(torch.t(vectors))
    D += vectors.pow(2).sum(dim=1).view(1, -1)
    D += vectors.pow(2).sum(dim=1).view(-1, 1)

    return D


def get_neg_hard(neg_ind,select_func, D, ap, ap_D, margin):
    trip = []
    if len(neg_ind)>0:
        for ap_i, ap_di in zip(ap, ap_D):
            loss_values = (ap_di -D[torch.LongTensor(np.array([ap_i[0]])),torch.LongTensor(neg_ind)] + margin)

            # loss_values = loss_values.data.cpu().numpy()
            loss_values = loss_values.detach().cpu().numpy()
            neg_hard = select_func(loss_values)

            if neg_hard is not None:
                neg_hard = neg_ind[neg_hard]
                trip.append([ap_i[0], ap_i[1], neg_hard])

    return trip

def random_neg(loss_values):
    neg_hards = np.where(loss_values > 0)[0]
    return np.random.choice(neg_hards) if len(neg_hards) > 0 else None