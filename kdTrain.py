import argparse
from tabnanny import check
from more_itertools import map_except
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import float32, optim
from torch.utils import data
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import numpy as np

import os
import json
import random
import socket
from tqdm import tqdm

from PIL import Image

import utils
import network
import datasets as dt
from metrics import StreamSegMetrics
from utils import ext_transforms as et
from utils import histeq as hq


def get_dataset(opts):
    if opts.is_rgb:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.485]
        std = [0.229]

    train_transform = et.ExtCompose([
        et.ExtResize(size=opts.resize),
        et.ExtRandomCrop(size=opts.crop_size, pad_if_needed=True),
        et.ExtScale(scale=opts.scale_factor),
        et.ExtRandomVerticalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=mean, std=std) 
        ])
    val_transform = et.ExtCompose([
        et.ExtResize(size=opts.resize),
        et.ExtRandomCrop(size=opts.crop_size, pad_if_needed=True),
        et.ExtScale(scale=opts.scale_factor),
        et.ExtToTensor(),
        et.ExtNormalize(mean=mean, std=std)
        ])

    if opts.dataset == "CPN":
        train_dst = dt.CPNSegmentation(root=opts.data_root, datatype=opts.dataset,
                                        image_set='train', transform=train_transform, 
                                        is_rgb=opts.is_rgb)
        val_dst = dt.CPNSegmentation(root=opts.data_root, datatype=opts.dataset,
                                     image_set='val', transform=val_transform, 
                                     is_rgb=opts.is_rgb)
    elif opts.dataset == "CPN_all":
        train_dst = dt.CPNALLSegmentation(root=opts.data_root, datatype=opts.dataset,
                                         image_set='train', transform=train_transform, 
                                         is_rgb=opts.is_rgb)
        val_dst = dt.CPNALLSegmentation(root=opts.data_root, datatype=opts.dataset,
                                        image_set='val', transform=val_transform, 
                                        is_rgb=opts.is_rgb)
    elif opts.dataset == "Median":
        train_dst = dt.Median(root=opts.data_root, datatype=opts.dataset, 
                            image_set='train', transform=train_transform,
                            is_rgb=opts.is_rgb)
        val_dst = dt.Median(root=opts.data_root, datatype=opts.dataset,
                        image_set='val', transform=val_transform,
                        is_rgb=opts.is_rgb)
    else:
        train_dst = dt.CPN(root=opts.data_root, datatype=opts.dataset, 
                            image_set='train', transform=train_transform,
                            is_rgb=opts.is_rgb)
        val_dst = dt.CPN(root=opts.data_root, datatype=opts.dataset,
                        image_set='val', transform=val_transform,
                        is_rgb=opts.is_rgb)
    
    return train_dst, val_dst


def build_log(opts, LOGDIR) -> SummaryWriter:
    # Tensorboard option
    if opts.save_log:
        logdir = os.path.join(LOGDIR, 'log')
        writer = SummaryWriter(log_dir=logdir)

    # Validate option
    if opts.val_results:
        logdir = os.path.join(LOGDIR, 'val_results')
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        opts.val_results_dir = logdir

    # Save best model option
    if opts.save_model:
        logdir = os.path.join(LOGDIR, 'best_param')
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        opts.save_ckpt = logdir
    else:
        logdir = os.path.join(LOGDIR, 'cache_param')
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        opts.save_ckpt = logdir

    # Save Options description
    jsummary = {}
    for key, val in vars(opts).items():
        jsummary[key] = val
    utils.save_dict_to_json(jsummary, os.path.join(LOGDIR, 'summary.json'))

    return writer


def validate(opts, model, loader, device, metrics, epoch, criterion):

    metrics.reset()
    ret_samples = []

    running_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1].detach().cpu().numpy()
            target = labels.detach().cpu().numpy()

            if opts.loss_type == 'ap_cross_entropy':
                weights = labels.detach().cpu().numpy().sum() / (labels.shape[0] * labels.shape[1] * labels.shape[2])
                weights = torch.tensor([weights, 1-weights], dtype=float32).to(device)
                criterion = utils.CrossEntropyLoss(weight=weights)
                loss = criterion(outputs, labels)
            elif opts.loss_type == 'ap_entropy_dice_loss':
                weights = labels.detach().cpu().numpy().sum() / (labels.shape[0] * labels.shape[1] * labels.shape[2])
                weights = torch.tensor([weights, 1-weights], dtype=float32).to(device)
                criterion = utils.EntropyDiceLoss(weight=weights)
                loss = criterion(outputs, labels)
            else:
                loss = criterion(outputs, labels)

            metrics.update(target, preds)
            running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    score = metrics.get_results()

    return score, epoch_loss

    
def train(opts, devices, LOGDIR) -> dict:

    writer = build_log(opts, LOGDIR)

    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)


    ''' (1) Get datasets
    '''
    train_dst, val_dst = get_dataset(opts)
    train_loader = DataLoader(train_dst, batch_size=opts.batch_size,
                                shuffle=True, num_workers=opts.num_workers, drop_last=True)
    val_loader = DataLoader(val_dst, batch_size=opts.val_batch_size, 
                                shuffle=True, num_workers=opts.num_workers, drop_last=True)
    print("Dataset: %s, Train set: %d, Val set: %d" % 
                    (opts.dataset, len(train_dst), len(val_dst)))

    ''' (2) Set up criterion
    '''
    if opts.loss_type == '':
        criterion = ...
    else:
        raise NotImplementedError

    ''' (3) Load model
    '''
    try:
        print("Model selection: {}".format(opts.model))
        if opts.model.startswith("deeplab"):
            model = network.model.__dict__[opts.model](channel=3 if opts.is_rgb else 1, 
                                                        num_classes=opts.num_classes, output_stride=opts.output_stride)
            if opts.separable_conv and 'plus' in opts.model:
                network.convert_to_separable_conv(model.classifier)
            utils.set_bn_momentum(model.backbone, momentum=0.01)
        else:
            model = network.model.__dict__[opts.model](channel=3 if opts.is_rgb else 1, 
                                                        num_classes=opts.num_classes)                         
    except:
        raise Exception

    ''' (4) Set up optimizer
    '''
    

    ''' (5) Resume model & scheduler
    '''
    

    ''' (6) Set up metrics
    '''
    metrics = StreamSegMetrics(opts.num_classes)
    early_stopping = utils.EarlyStopping(patience=opts.patience, verbose=True, delta=opts.delta,
                                            path=opts.save_ckpt, save_model=opts.save_model)
    dice_stopping = utils.DiceStopping(patience=opts.patience, verbose=True, delta=opts.delta,
                                            path=opts.save_ckpt, save_model=opts.save_model)
    best_score = 0.0

    ''' (7) Train
    '''
    B_epoch = 0
    B_val_score = None

    