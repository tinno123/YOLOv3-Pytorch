import math
import random

import cv2
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
import collections
import torch
from utils.datasets import Dataset


class CosineDecayLR(object):
    def __init__(self, max_lr, min_lr, total_epochs,warmup_epochs=0.01):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs
        self.warmup_epochs = int(warmup_epochs * total_epochs)
        self.remain_epoch = self.total_epochs - self.warmup_epochs

    def warmup(self,epoch):
        if type(self.max_lr) == list:
            lr = []
            for i in self.max_lr:
                lr.append(i* epoch / self.warmup_epochs)
            return lr
        else:
            return self.max_lr * epoch / self.warmup_epochs

    def cosine_decay(self,epoch):
        current_epoch = epoch - self.warmup_epochs
        theta = (1 - current_epoch / self.remain_epoch) * math.pi

        if type(self.max_lr) == list:
            lr = []
            for maxlr , minlr in zip(self.max_lr,self.min_lr):
                lr.append(maxlr - (maxlr - minlr) * (1 + math.cos(theta)) / 2)

        else:
            lr = self.max_lr - (self.max_lr - self.min_lr) * (1 + math.cos(theta)) / 2
        return lr

    def update_lr(self,optimizer,lr):
        if type(lr) == list and type(optimizer) == list:
            for o,lr in zip(optimizer,lr):
                for param_group in o.param_groups:
                    param_group['lr'] = lr
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    def __call__(self, epoch,optimizer):

        epoch = min(epoch + 1, self.total_epochs)

        #----------------------------#
        # 热身阶段,LR线性上升
        # ---------------------------#
        if epoch <= self.warmup_epochs:
            lr = self.warmup(epoch)
        else:
        # ----------------------------#
        # 余弦退火阶段，从热身之后开始
        # ----------------------------#
            lr = self.cosine_decay(epoch)

        # ----------------------------#
        # 更新优化器的学习率
        # ----------------------------#
        if optimizer is not None:
            self.update_lr(optimizer,lr)

        return lr

def GetOptimizer(model,optimizer_type,lr,momentum,weight_decay):
    pg_normal_weight, pg_bias, pg_bn_weight = [], [], []
    for k , v  in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg_bias.append(v.bias)
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg_bn_weight.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg_normal_weight.append(v.weight)
    optimizer = {
        'adam':
            optim.Adam([
                {'params': pg_normal_weight, 'weight_decay': weight_decay},
                {'params': pg_bias},
                {'params': pg_bn_weight}
            ], lr, betas=(momentum, 0.999)),
        'sgd':
            optim.SGD([
                {'params': pg_normal_weight, 'weight_decay': weight_decay},
                {'params': pg_bias},
                {'params': pg_bn_weight}
            ], lr, momentum=momentum, nesterov=True)
    }[optimizer_type]
    return optimizer


def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def GetLoader(
        root_dir,
        train_annotation_path,
        train_image_path,
        val_annotation_path,
        val_image_path,
        batch_size,
        num_workers,
        anchors
):


    train_dataset = Dataset(root_dir,
                            train_annotation_path,
                            train_image_path,
                            anchors,
                            [416, 416],
                            mode="train"
                            )
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True, persistent_workers=True,
                         drop_last=True)
    val_dataset = Dataset(root_dir,
                          val_annotation_path,
                          val_image_path,
                          anchors,
                          [416, 416],
                          mode="val"
                          )
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=True, persistent_workers=True,
                            drop_last=False)
    return train_loader, val_loader


def GetGridCenter(feature_map_size,num_anchors_perscale,stride,mode ="train"):
    w, h = feature_map_size
    grid_x = torch.arange(w, dtype=torch.float32)
    grid_y = torch.arange(h, dtype=torch.float32)
    grid_x = grid_x.repeat(h)
    grid_y = grid_y.repeat_interleave(w)
    if mode == "train":
        origin_center = torch.stack([(grid_x + 0.5) * stride, (grid_y + 0.5) * stride], dim=-1)
    else :
        origin_center = torch.stack([grid_x, grid_y], dim=-1)
    return origin_center.repeat_interleave(num_anchors_perscale, dim=0)

def image2tensor(image: np.ndarray,device,image_size = [416,416]):
    image = cv2.resize(image, (image_size[0], image_size[1]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).float()
    image = image.permute(2, 0, 1)
    image = image / 255.0
    return image.unsqueeze(0).to(device)


def load_weights_by_shape(model, weight_file):
    """
    此方法针引用模型权重键名不一致且确定结构一致
    :param model:
    :param weight_file:
    :return:
    """
    # -----------------------
    # 加载权重
    # -----------------------
    weights_values = []

    if isinstance(weight_file, str):
        try:
            weights = torch.load(weight_file)
            for k, v in weights.items():
                if "num_batches_tracked" in k:
                    continue
                weights_values.append(v)
        except Exception as e:
            raise ValueError(f"加载模型文件失败：{e}")
    elif isinstance(weight_file, (dict, collections.OrderedDict)):
        weights = weight_file
        for k, v in weights.items():
            if "num_batches_tracked" in k:
                continue
            weights_values.append(v)
    else:
        raise ValueError(f"不支持的权重类型：{type(weight_file)}")

    if len(weights_values) == 0:
        raise ValueError("权重文件为空或格式不正确")
    # -----------------------
    # 模型权重赋值
    # -----------------------
    index = 0
    success_num = 0
    fail = []
    model_state_dict = model.state_dict()
    for k, v in model_state_dict.items():
        if index >= len(weights_values):
            break
        # -----------------------
        # 去除批次数量追踪层
        # -----------------------
        if "num_batches_tracked" in k:
            continue
        if weights_values[index].shape == v.shape:
            model_state_dict[k] = weights_values[index]
            success_num += 1
        else:
            print(f"权重形状不一致: {k}")
            fail.append((k, v.shape, weights_values[index].shape))
        index += 1
    print(f"加载数量 / 权重总量 : {success_num} / {len(weights_values)}" )
    print(f"加载失败:{fail}")

    model.load_state_dict(model_state_dict)















