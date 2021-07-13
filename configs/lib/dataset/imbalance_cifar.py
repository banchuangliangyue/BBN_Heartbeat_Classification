# To ensure fairness, we use the same code in
# LDAM (https://github.com/kaidic/LDAM-DRW) to
# produce long-tailed CIFAR datasets.
import os
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random
from torch.utils.data import DataLoader, Dataset
import torch

M = 73
channel = 1
num2char = {0: 'N', 1: 'S', 2: 'V', 3: 'F'}

def load_data(dataset_name='DS1', n_class=None, records=None, train=False):

    # train_record_list = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203,
    #                      205, 207, 208, 209, 215, 220, 223, 230]
    # test_record_list = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214,
    #                     219, 221, 222, 228, 231, 232, 233, 234]

    if dataset_name == 'DS1':
        # data_dir = '../EMBC/mitdb_dual_beats_DS1/'
        data_dir = '../EMBC/mitdb_DS1_3d_inputs_4classes/'

    if dataset_name == 'DS2':
        # data_dir = '../EMBC/mitdb_dual_beats_DS2/'
        data_dir = '../EMBC/mitdb_DS2_3d_inputs_4classes/'

    print('\ndata_dir path:', data_dir)
    data_list = os.listdir(data_dir)

    # sample_x = np.array([]).reshape((0, M, M, channel))
    # sample_y = np.array([]).reshape((0, n_class))
    sample_x = np.array([]).reshape((0, 3, 128, channel))
    sample_y = np.array([]).reshape((0, n_class))
    loaded_records = []
    for rec in data_list :
        if int(rec.strip('.npz')) in records:
            loaded_records.append(int(rec.strip('.npz')))
            a = np.load(data_dir + rec)
            beat = a['ECG_Data']
            Label = a['Label']
            sample_x = np.concatenate((sample_x, beat), axis=0)
            sample_y = np.concatenate((sample_y, Label), axis=0)

    print('Loaded Records:', loaded_records)

    if train:
        print("shuffle training set...........")
        index = np.arange(sample_x.shape[0])
        np.random.shuffle(index)
        sample_x = sample_x[index]
        sample_y = sample_y[index]

    sample_x = sample_x.astype(np.float32)
    sample_y = sample_y.astype(np.int64)
    sample_y = np.argmax(sample_y, axis=1)
    sample_x = np.transpose(sample_x, (0, 3, 1, 2))
    sample_x, sample_y = map(torch.tensor, (sample_x, sample_y))
    return sample_x, sample_y


class NSVF4(Dataset):
    cls_num = 4

    def __init__(self, mode, cfg, dataset_name=None, ecg_records=None):
        train = True if mode == "train" else False
        super(NSVF4, self).__init__()
        self.cfg = cfg
        self.train = train
        self.dual_sample = True if cfg.TRAIN.SAMPLER.DUAL_SAMPLER.ENABLE and self.train else False
        # rand_number = cfg.DATASET.IMBALANCECIFAR.RANDOM_SEED
        # if dataset_name == 'DS1':
        #     self.train_x, self.train_y, self.valx, self.val_y, self.num_per_cls = self.load_dataset(dataset_name=dataset_name,
        #                                                                                val_ratio=val_ratio)
        # else:
        #     self.test_x, self.test_y = self.load_dataset(dataset_name=dataset_name)
        self.sample_x, self.sample_y = load_data(train=self.train, dataset_name=dataset_name, n_class=self.cls_num,
                                                 records=ecg_records)
        self.len_samples = len(self.sample_x)
        # print("{} Mode: shpae {}".format(mode, len(self.sample_x)))
        print('{} sample_x.shape: {}'.format(mode, self.sample_x.shape))
        print('{} sample_y.shape: {}'.format(mode, self.sample_y.shape))
        for i in range(self.cls_num):
            print(dataset_name + ' ' + mode + '中' + num2char[i] + '类的数量：', torch.eq(self.sample_y, i).sum().item())


        if self.dual_sample or (self.cfg.TRAIN.SAMPLER.TYPE == "weighted sampler" and self.train):
            self.class_weight, self.sum_weight = self.get_weight(self.get_annotations(self.sample_y), self.cls_num)
            self.class_dict = self._get_class_dict()
            print("{} Mode: Dual_sample\n".format(mode))



    def sample_class_index_by_weight(self):
        rand_number, now_sum = random.random() * self.sum_weight, 0
        for i in range(self.cls_num):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i

    def reset_epoch(self, cur_epoch):
        self.epoch = cur_epoch

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations(self.sample_y)):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_weight(self, annotations, num_classes):
        num_list = [0] * num_classes
        cat_list = []
        for anno in annotations:
            category_id = anno["category_id"]
            num_list[category_id] += 1
            cat_list.append(category_id)
        max_num = max(num_list)
        class_weight = [max_num / i for i in num_list]
        sum_weight = sum(class_weight)
        return class_weight, sum_weight

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        '''Python的魔法方法__getitem__ 可以让对象实现迭代功能，这样就可以使用for...in... 来迭代该对象了
        class Animal:
            def __init__(self, animal_list):
                self.animals_name = animal_list

            def __getitem__(self, index):
                return self.animals_name[index]

        animals = Animal(["dog", "cat", "fish"])
        for animal in animals:
            print(animal)
        '''
        if self.cfg.TRAIN.SAMPLER.TYPE == "weighted sampler" and self.train:
            assert self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE in ["balance", "reverse"]
            if self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "balance":
                sample_class = random.randint(0, self.cls_num - 1)
            elif self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "reverse":
                sample_class = self.sample_class_index_by_weight()
            sample_indexes = self.class_dict[sample_class]
            index = random.choice(sample_indexes)


        # img, target = self.data[index], self.targets[index]
        img, target = self.sample_x[index], self.sample_y[index]
        meta = dict()

        if self.dual_sample:
            if self.cfg.TRAIN.SAMPLER.DUAL_SAMPLER.TYPE == "reverse":
                sample_class = self.sample_class_index_by_weight()
                sample_indexes = self.class_dict[sample_class]
                ###random.choice() 使用的算法默认为重复抽样
                sample_index = random.choice(sample_indexes)
            elif self.cfg.TRAIN.SAMPLER.DUAL_SAMPLER.TYPE == "balance":
                sample_class = random.randint(0, self.cls_num - 1)
                sample_indexes = self.class_dict[sample_class]
                sample_index = random.choice(sample_indexes)
            elif self.cfg.TRAIN.SAMPLER.DUAL_SAMPLER.TYPE == "uniform":
                sample_index = random.randint(0, self.__len__() - 1)

            sample_img, sample_label = self.sample_x[sample_index], self.sample_y[sample_index]


            meta['sample_image'] = sample_img
            meta['sample_label'] = sample_label

        return img, target, meta

    def __len__(self):
        return self.len_samples

    def get_num_classes(self):
        return self.cls_num

    def reset_epoch(self, epoch):
        self.epoch = epoch

    def get_annotations(self, targets):
        annos = []
        for target in targets:
            annos.append({'category_id': int(target)})
        return annos

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list



if __name__ == '__main__':

    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # trainset = IMBALANCECIFAR100(root='./data', train=True,
    #                 download=True, transform=transform)
    # trainloader = iter(trainset)
    # data, label = next(trainloader)
    # import pdb; pdb.set_trace()

    train_record_list = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203,
                         205, 207, 208, 209, 215, 220, 223, 230]
    test_record_list = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214,
                        219, 221, 222, 228, 231, 232, 233, 234]
    val_record_list = np.random.choice(train_record_list, 4, )
    train_record_list = [i for i in train_record_list and i not in val_record_list]
    print('train record list:', train_record_list)
    print('val record list:', val_record_list)
    print('test record list:', test_record_list)
    trainset = eval(cfg.DATASET.DATASET)("train", cfg, dataset_name='DS1', ecg_records=train_record_list)
    valset = eval(cfg.DATASET.DATASET)("val", cfg, dataset_name='DS1', ecg_records=val_record_list)
    testset = eval(cfg.DATASET.DATASET)("test", cfg, dataset_name='DS2', ecg_records=test_record_list)

