"""
    Dynamic Group Convolution
    date: July 5th, 2020
    authors: Zhuo Su, Linpu Fang
    paper: Dynamic Group Convolution for Accelerating Convolutional Neural Networks, ECCV 2020.

    Code forked from "https://github.com/ShichenLiu/CondenseNet"
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import argparse
import os
import time
import models
from utils import *

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

parser = argparse.ArgumentParser(description='PyTorch main code for Dynamic Group Convolution')
parser.add_argument('--data', type=str, default='imagenet',
                    help='name of dataset',
                    choices=['cifar10', 'cifar100', 'imagenet'])
parser.add_argument('--datadir', type=str, default='../data',
                    help='dir to the dataset')
parser.add_argument('--savedir', type=str, default='results/exp',
                    help='path to save result and checkpoint')

parser.add_argument('--model', type=str, default='dydensenet', 
                    help='model to train the dataset')
parser.add_argument('-j', '--workers', type=int, default=8,
                    help='number of data loading workers')
parser.add_argument('--epochs', type=int, default=120,
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', type=int, default=256,
                    help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', type=float, default=0.1,
                    help='initial learning rate')
parser.add_argument('--lr-type', type=str, default='cosine',
                    help='learning rate strategy',
                    choices=['cosine', 'multistep'])
parser.add_argument('--group-lasso-lambda', type=float, default=1e-5,
                    help='group lasso loss weight')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum for sgd')
parser.add_argument('--weight-decay', '--wd', type=float, default=1e-4,
                    help='weight decay')
parser.add_argument('--seed', type=int, default=None,
                    help='manual seed')
parser.add_argument('--gpu', type=str, default='',
                    help='gpu available')

parser.add_argument('--stages', type=str,
                    help='per layer depth')
parser.add_argument('--squeeze-rate', type=int, default=16,
                    help='squeeze rate in SE head')
parser.add_argument('--heads', type=int, default=4,
                    help='number of heads for 1x1 convolution')
parser.add_argument('--group-3x3', type=int, default=4,
                    help='3x3 group convolution')
parser.add_argument('--gate-factor', type=float, default=0.25,
                    help='gate factor')
parser.add_argument('--growth', type=str,
                    help='per layer growth')
parser.add_argument('--bottleneck', type=int, default=4,
                    help='bottleneck in densenet')

parser.add_argument('--print-freq', type=int, default=10,
                    help='print frequency')
parser.add_argument('--save-freq', type=int, default=10, 
                    help='save frequency')
parser.add_argument('--resume', action='store_true',
                    help='use latest checkpoint if have any')
parser.add_argument('--evaluate', type=str, default=None, 
                    help="full path to checkpoint to be evaluated")

args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

best_prec1 = 0

def main():
    global args, best_prec1

    if args.seed is None:
        args.seed = int(time.time())
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    R = 32
    if args.data == 'cifar10':
        args.num_classes = 10
    elif args.data == 'cifar100':
        args.num_classes = 100
    else:
        args.num_classes = 1000
        R = 224

    if 'densenet' in args.model:
        args.stages = list(map(int, args.stages.split('-')))
        args.growth = list(map(int, args.growth.split('-')))


    ### Calculate FLOPs & Param
    model = getattr(models, args.model)(args)
    n_flops, n_params = measure_model(model, R, R)
    print('FLOPs: %.2fM, Params: %.2fM' % (n_flops / 1e6, n_params / 1e6))

    os.makedirs(args.savedir, exist_ok=True)
    log_file = os.path.join(args.savedir, "%s_%d_%d.txt" % \
        (args.model, int(n_params), int(n_flops)))
    del(model)

    ### Create model
    model = getattr(models, args.model)(args)
    model = torch.nn.DataParallel(model).cuda()

    ### Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    cudnn.benchmark = True

    ### Data loading 
    if args.data == "cifar10":
        normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                         std=[0.2471, 0.2435, 0.2616])
        train_set = datasets.CIFAR10(args.datadir, train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.RandomCrop(32, padding=4),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normalize,
                                     ]))
        val_set = datasets.CIFAR10(args.datadir, train=False,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       normalize,
                                   ]))
    elif args.data == "cifar100":
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        train_set = datasets.CIFAR100(args.datadir, train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.RandomCrop(32, padding=4),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normalize,
                                     ]))
        val_set = datasets.CIFAR100(args.datadir, train=False,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       normalize,
                                   ]))
    else: #imagenet
        traindir = os.path.join(args.datadir, 'train')
        valdir = os.path.join(args.datadir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_set = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

        val_set = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    ### Optionally resume from a checkpoint
    args.start_epoch = 0
    if args.resume or (args.evaluate is not None):
        checkpoint = load_checkpoint(args)
        if checkpoint is not None:
            model.load_state_dict(checkpoint['state_dict'])
            try:
                args.start_epoch = checkpoint['epoch'] + 1
                best_prec1 = checkpoint['best_prec1']
                optimizer.load_state_dict(checkpoint['optimizer'])
            except KeyError:
                pass

    ### Evaluate directly if required
    print(args)
    if args.evaluate is not None:
        validate(val_loader, model, criterion, args)
        return

    saveID = None
    for epoch in range(args.start_epoch, args.epochs):
        ### Train for one epoch
        tr_prec1, tr_prec5, loss, lr = \
            train(train_loader, model, criterion, optimizer, epoch, args)

        ### Evaluate on validation set
        val_prec1, val_prec5 = validate(val_loader, model, criterion, args)

        ### Remember best prec@1 and save checkpoint
        is_best = val_prec1 >= best_prec1
        best_prec1 = max(val_prec1, best_prec1)

        log = ("Epoch %03d/%03d: top1 %.4f | top5 %.4f" + \
              " | train-top1 %.4f | train-top5 %.4f | loss %.4f | lr %.5f | Time %s\n") \
              % (epoch, args.epochs, val_prec1, val_prec5, tr_prec1, \
              tr_prec5, loss, lr, time.strftime('%Y-%m-%d %H:%M:%S'))
        with open(log_file, 'a') as f:
            f.write(log)

        saveID = save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
            }, epoch, args.savedir, is_best, 
            saveID, keep_freq=args.save_freq)

    return


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    lasso_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    ### Switch to train mode
    model.train()
    wD = len(str(len(train_loader)))
    wE = len(str(args.epochs))

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        progress = float(epoch * len(train_loader) + i) / \
            (args.epochs * len(train_loader))
        ## Adjust learning rate
        lr = adjust_learning_rate(optimizer, epoch, args, batch=i,
                                  nBatch=len(train_loader), method=args.lr_type)

        ## Measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        ## Compute output
        output, _lasso_list = model(input, progress)
        loss = criterion(output, target)

        ## Add group lasso loss
        lasso_loss = 0
        if args.group_lasso_lambda > 0:
            for lasso_m in _lasso_list:
                lasso_loss = lasso_loss + lasso_m.mean()
        loss = loss + args.group_lasso_lambda * lasso_loss

        ## Measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        lasso_losses.update(lasso_loss.item())
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        ## Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ## Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        ## Record
        if i % args.print_freq == 0:
            print(('Epoch: [{0}/{1}][{2}/{3}]\t' + \
                  'Time {batch_time.val:.3f}\t' + \
                  'Data {data_time.val:.3f}\t' + \
                  'Loss (lasso_loss)  {loss.val:.4f} ({lasso_loss.val:.4f})\t' + \
                  'Prec@1 {top1.val:.3f}\t' + \
                  'Prec@5 {top5.val:.3f}\t' + \
                  'lr {lr: .5f}\t').format(
                      epoch, args.epochs, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, lasso_loss=lasso_losses, 
                      top1=top1, top5=top5, lr=lr))

    return top1.avg, top5.avg, losses.avg, lr


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    ## Switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        ## Compute output
        with torch.no_grad():
            target = target.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)

            output, _ = model(input)
            loss = criterion(output, target)

        ## Measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        ## Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg



if __name__ == '__main__':
    main()
