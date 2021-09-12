import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torchvision.datasets import cifar
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models

from torch.utils.data.distributed import DistributedSampler

def cifar10(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    sampler = None
    train_dataset = datasets.CIFAR10(root=args.data_root, train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True)

    if args.distributed:
        sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank, seed=args.seed)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=(sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=args.data_root, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader

def cifar100(args):
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                    std=[0.2675, 0.2565, 0.2761])

    sampler = None
    train_dataset = datasets.CIFAR100(root=args.data_root, train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True)

    if args.distributed:
        sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank, seed=args.seed)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=(sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root=args.data_root, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader

def cifar10c(args):
    pass

def cifar100c(args):
    pass


def get_dataloaders(args):
    
    if args.dataset == 'cifar10':
        return cifar10(args)
    elif args.dataset == 'cifar100':
        return cifar100(args)

def get_corrupted_dataloaders(args):
    
    if args.dataset == 'cifar10':
        return cifar10c(args)
    elif args.dataset == 'cifar100':
        return cifar100c(args)
 