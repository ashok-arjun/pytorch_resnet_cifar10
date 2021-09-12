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
import datetime
import builtins as __builtin__


from .utils import set_seed
from datasets import get_dataloaders, get_corrupted_dataloaders
from validate import validate
from train import train


LOCAL_RANK = None

def print(*args, **kwargs):
    if not LOCAL_RANK or LOCAL_RANK == 0:
        return __builtin__.print(*args, **kwargs)

def save_checkpoint(root, state, is_best):
    if not LOCAL_RANK or LOCAL_RANK == 0: 
        filename = os.path.join(root, "models/", "last.pth.tar")
        torch.save(state, filename)

        if is_best:
            filename = os.path.join(root, "models/", "best.pth.tar")
            torch.save(state, filename)

def main(gpu, args):
    set_seed(args.seed)

    if args.distributed:
        rank = args.nr * args.gpus + gpu
        args.rank = rank
        LOCAL_RANK = rank
        dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
        torch.cuda.set_device(gpu)

    if args.distributed:
        wandb.init(project="safe-regions", entity="arjunashok", config=vars(args), group=args.run_name)
    else:
        wandb.init(project="safe-regions", entity="arjunashok", config=vars(args))
        wandb.run.name = args.run_name

    wandb.config.update({"gpus": os.environ["CUDA_VISIBLE_DEVICES"]}, allow_val_change=True)


    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = getattr(models, args.arch)(num_channels=args.num_channels, \
    batch_norm=args.batch_norm, safe_region=args.safe_region, num_classes=args.num_classes)

    if args.distributed:
        print("Distributed Data Parallel!")        
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)    
        model.cuda(gpu)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)
    else: 
        print("Normal single GPU : ", gpu)        
        model = model.to(gpu)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            best_prec5 = checkpoint['best_prec5']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    train_loader, val_loader = get_dataloaders(args)

    if args.test_corrupted: 
        corrupted_loader = get_corrupted_dataloaders(args)

    criterion = nn.CrossEntropyLoss().cuda()

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', \
                                                              factor=args.factor, patience=args.patience)

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    best_prec1 = -1
    best_prec5 = -1

    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        wandb.log({"general/epoch": epoch})

        train(args, train_loader, model, criterion, optimizer, epoch)
        prec1, prec5 = validate(args, val_loader, model, criterion, epoch)

        is_best1 = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        is_best5 = prec5 > best_prec5
        best_prec5 = max(prec5, best_prec5)

        wandb.log({"val/best_prec1": best_prec1, "val/best_prec5": best_prec5}, step=epoch)

        if args.best_metric == "prec1":         
            lr_scheduler.step(prec1)
        elif args.best_metric == "prec5":
            lr_scheduler.step(prec5)

        save_checkpoint(
            args.save_dir,
            {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'best_prec5': best_prec5,
        }, is_best1 if args.best_metric=="prec1" else is_best5)


        wandb.log({"general/lr": optimizer.param_groups[0]['lr']}, step=epoch)
        if  optimizer.param_groups[0]['lr']<= 1e-6:
            print("LR went below 1e-6. Breaking...")
            break

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                        help='model architecture: ')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save-dir', dest='save_dir',
                        help='The directory used to save the trained models',
                        default='results/'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), type=str)
    parser.add_argument('--num-classes', type=int, default=10)
    
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--run-name', type=str, default='')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--data-root', type=str, default='/home/arjun_ashok/data')
    parser.add_argument('--optim', type=str, default='sgd')
    parser.add_argument('--factor', type=int, default=0.1)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--best-metric', type=str, default="top1")
    
    parser.add_argument('--test-corrupted', action='store_true')

    best_prec1 = 0

    args = parser.parse_args()

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        args.distributed = False
                
    if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) > 1 and not args.distributed:
        raise Exception("Multi GPU - do DDP!")

    if args.distributed:

        args.nodes = 1
        args.gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        args.nr = 0

        if args.gpus == 1:
            raise Exception("Single GPU - cannot do DDP!")

        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29534'

        args.world_size = args.gpus * args.nodes

        mp.spawn(main, nprocs=args.gpus, args=(args, ))
    else:
        main(0, args)

    main()
