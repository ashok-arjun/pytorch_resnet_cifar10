import argparse
import os
import shutil
import time

import numpy as np

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
import copy

from .utils import save_checkpoint, accuracy, AverageMeter, set_seed, update_dict
from .models.safe_region.safe_region_utils import collect_safe_regions_stats

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

def validate_corrupted(args, val_dataset, model, criterion, epoch, prefix="corrupted-"):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    safe_regions_dict = {}

    # switch to evaluate mode
    model.eval()

    for corruption in CORRUPTIONS:
        val_data = copy.deepcopy(val_dataset)
        val_data.data = np.load(args.data_root + args.dataset + "_corrupted/" + corruption + '.npy')
        val_data.targets = torch.LongTensor(np.load(args.data_root + args.dataset + "_corrupted/" +  + 'labels.npy'))

        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True)

        start_time = time.time()

        end = time.time()
        
        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                target = target.cuda()
                input_var = input.cuda()
                target_var = target.cuda()

                # compute output
                output = model(input_var)
                loss = criterion(output, target_var)

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

                safe_regions_stats = collect_safe_regions_stats(model, training=False, \
                run_dir=args.save_dir, step=epoch, prefix=prefix)
                update_dict(safe_regions_dict, safe_regions_stats)


                if i % args.print_freq == 0:
                    print(prefix + 'Test: [{0}/{1}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                            i, len(val_loader), batch_time=batch_time, loss=losses,
                            top1=top1, top5=top5))

    print(prefix + '* Prec@1 {top1.avg:.3f}\t'
          '* Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    wandb.log({prefix + "val/loss": losses(), prefix + "val/top_1": top1()}, step=epoch)
    wandb.log({prefix + "val/batch_time": batch_time(), prefix + "val/top_5": top5()}, step=epoch)
    wandb.log({prefix + "val/epoch_time": time.time() - start_time}, step=epoch)

    wandb.log(safe_regions_dict, step=epoch)

    return top1.avg, top5.avg