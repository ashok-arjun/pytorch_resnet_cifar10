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
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import wandb

from .utils import save_checkpoint, accuracy, AverageMeter, set_seed

from .models.safe_region.safe_region_utils import collect_safe_regions_stats

def train(args, train_loader, model, criterion, optimizer, epoch, prefix=""):
    """
    Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    epoch_start = time.time()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(prefix + 'Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, i, len(train_loader), loss=losses, top1=top1, top5=top5))

    wandb.log({prefix + "train/loss": losses(), prefix + "train/top_1": top1()}, step=epoch)
    wandb.log({prefix + "train/batch_time": batch_time(), prefix + "train/top_5": top5()}, step=epoch)
    wandb.log({prefix + "train/data_time": data_time(), prefix + "train/epoch_time": time.time() - epoch_start}, step=epoch)

    safe_regions_stats = collect_safe_regions_stats(model, training=True, run_dir=args.save_dir, step=epoch, prefix=prefix)
    wandb.log({safe_regions_stats}, step=epoch)

    return top1(), top5()