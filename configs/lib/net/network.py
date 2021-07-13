import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import res50, bbn_res50, res32_cifar, bbn_res32_cifar
from modules import GAP, Identity, FCNorm


class Network(nn.Module):
    def __init__(self, cfg, mode="train", num_classes=1000):
        super(Network, self).__init__()
        pretrain = (
            True
            if mode == "train"
            and cfg.RESUME_MODEL == ""
            and cfg.BACKBONE.PRETRAINED_MODEL != ""
            else False
        )

        self.num_classes = num_classes
        self.cfg = cfg

        self.backbone = eval(self.cfg.BACKBONE.TYPE)(
            self.cfg,
            pretrain=pretrain,
            pretrained_model=cfg.BACKBONE.PRETRAINED_MODEL,
            last_layer_stride=2,
        )
        # self.module = self._get_module()
        ###所以上下两个分支的分类器也是共享的？？但在paper中两个分类器是不同的，目前我也倾向于使用两个分类器
        # self.classifier = self._get_classifer()
        self.classifier_cb = self._get_classifer()
        self.classifier_rb = self._get_classifer()
        self.feature_len = self.get_feature_length()


    def forward(self, x, **kwargs):
        if "feature_flag" in kwargs or "feature_cb" in kwargs or "feature_rb" in kwargs:
            return self.extract_feature(x, **kwargs)
        elif "classifier_cb" in kwargs:
            return self.classifier_cb(x)
        elif "classifier_rb" in kwargs:
            return self.classifier_rb(x)


        xa, xb = self.backbone(x)
        # x = self.module(x)
        xa, xb = xa.view(xa.shape[0], -1), xb.view(xb.shape[0], -1)
        # x = x.view(x.shape[0], -1)
        # x = self.classifier(x)
        xa = self.classifier_cb(xa)
        xb = self.classifier_rb(xb)


        return xa, xb


    def extract_feature(self, x, **kwargs):
        if "feature_flag" in kwargs:
            if "bbn" in self.cfg.BACKBONE.TYPE:
                xa, xb = self.backbone(x, **kwargs)

            else:
                xa, xb = self.backbone(x)
            xa, xb = xa.view(xa.shape[0], -1), xb.view(xb.shape[0], -1)
            return xa, xb

        else:
            if "bbn" in self.cfg.BACKBONE.TYPE:
                x = self.backbone(x, **kwargs)
            else:
                x = self.backbone(x)
            # x = self.module(x)
            x = x.view(x.shape[0], -1)
            return x


    def freeze_backbone(self):
        print("Freezing backbone .......")
        for p in self.backbone.parameters():
            p.requires_grad = False


    def load_backbone_model(self, backbone_path=""):
        self.backbone.load_model(backbone_path)
        print("Backbone has been loaded...")


    def load_model(self, model_path):
        pretrain_dict = torch.load(
            model_path, map_location="cpu" if self.cfg.CPU_MODE else "cuda"
        )
        pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v
        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        print("Model has been loaded...")


    def get_feature_length(self):
        if "cifar" in self.cfg.BACKBONE.TYPE:
            # num_features = 64
            num_features = 256*25
        else:
            num_features = 2048

        # if "bbn" in self.cfg.BACKBONE.TYPE:
        #     num_features = num_features * 2
        return num_features


    def _get_module(self):
        module_type = self.cfg.MODULE.TYPE
        if module_type == "GAP":
            module = GAP()
        elif module_type == "Identity":
            module= Identity()
        else:
            raise NotImplementedError

        return module


    def _get_classifer(self):
        bias_flag = self.cfg.CLASSIFIER.BIAS

        num_features = self.get_feature_length()
        if self.cfg.CLASSIFIER.TYPE == "FCNorm":
            classifier = FCNorm(num_features, self.num_classes)
        elif self.cfg.CLASSIFIER.TYPE == "FC":
            classifier = nn.Linear(num_features, self.num_classes, bias=bias_flag)
        else:
            raise NotImplementedError

        return classifier
