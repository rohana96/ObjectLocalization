import argparse
import os
import shutil
import time
import sys
import sklearn
import sklearn.metrics
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from AlexNet import *
from voc_dataset import *
from utils import *

import wandb

# batchsize=32, learning rate=0.01, epochs=2
USE_WANDB = True  # use flags, wandb is not convenient for debugging

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', default='localizer_alexnet')
parser.add_argument(
    '-j',
    '--workers',
    default=4,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 4)')
parser.add_argument(
    '--epochs',
    default=30,
    type=int,
    metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '--start-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '-b',
    '--batch-size',
    default=256,
    type=int,
    metavar='N',
    help='mini-batch size (default: 256)')
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=0.1,
    type=float,
    metavar='LR',
    help='initial learning rate')
parser.add_argument(
    '--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument(
    '--weight-decay',
    '--wd',
    default=1e-4,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-4)')
parser.add_argument(
    '--print-freq',
    '-p',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--eval-freq',
    default=2,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '-e',
    '--evaluate',
    dest='evaluate',
    action='store_true',
    help='evaluate model on validation set')
parser.add_argument(
    '--pretrained',
    dest='pretrained',
    action='store_true',
    default=True,
    help='use pre-trained model')
parser.add_argument(
    '--world-size',
    default=1,
    type=int,
    help='number of distributed processes')
parser.add_argument(
    '--dist-url',
    default='tcp://224.66.41.62:23456',
    type=str,
    help='url used to set up distributed training')
parser.add_argument(
    '--dist-backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--vis', action='store_true')
parser.add_argument('--log_freq', default=50)

best_prec1 = 0
val_step = 0
train_step = 0

def main():
    global args, best_prec1
    args = parser.parse_args()
    args.distributed = args.world_size > 1
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'localizer_alexnet':
        model = localizer_alexnet(pretrained=args.pretrained)
    elif args.arch == 'localizer_alexnet_robust':
        model = localizer_alexnet_robust(pretrained=args.pretrained)
    print(model)

    model.features = torch.nn.DataParallel(model.features)
    model.cuda()
    # TODO:
    # define loss function (criterion) and optimizer
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, 
                                weight_decay=args.weight_decay,
                                momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #                                                         optimizer,
    #                                                         mode='min',
    #                                                         factor=0.1,
    #                                                         patience=10,
    #                                                         threshold=0.0001,
    #                                                         threshold_mode='abs'
    #                                                     )
    # also use an LR scheduler to decay LR by 10 every 30 epochs
    # you can also use PlateauLR scheduler, which usually works well

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code

    # TODO: Create Datasets and Dataloaders using VOCDataset - Ensure that the sizes are as required
    # Also ensure that data directories are correct - the ones use for testing by TAs might be different
    # Resize the images to 512x512
    train_dataset, val_dataset = VOCDataset('trainval', 512, top_n=30), VOCDataset('test', 512, top_n=30)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        # shuffle=(train_sampler is None),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    # TODO: Create loggers for wandb - ideally, use flags since wandb makes it harder to debug code.
    if USE_WANDB:
        wandb.init(project="vlr2-hw2-task1", name="q1_7", reinit=True)


    for epoch in range(args.start_epoch, args.epochs):
        # scheduler(optimizer, epoch)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args.log_freq)
        scheduler.step()

        # evaluate on validation set
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            m1, m2 = validate(val_loader, model, criterion, epoch, args.log_freq)

            score = m1 * m2
            # remember best prec@1 and save checkpoint
            is_best = score > best_prec1
            best_prec1 = max(score, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)


# TODO: You can add input arguments if you wish
def train(train_loader, model, criterion, optimizer, epoch, log_freq=10):


    global train_step 
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, (data) in enumerate(train_loader):

        train_step += 1
        # measure data loading time
        data_time.update(time.time() - end)
        # TODO: Get inputs from the data dict
        image, label = data['image'].cuda(), data['label'].cuda()
        # TODO: Get output from model

        pred_feat = model(image)
        _, _, H, W = pred_feat.shape

        if args.arch == 'localizer_alexnet':
            pred = nn.MaxPool2d(kernel_size=(H, W))(pred_feat)
        if args.arch == 'localizer_alexnet_robust':
            pred = nn.AvgPool2d(kernel_size=(H, W))(pred_feat)

        pred = pred.squeeze()
        pred = torch.sigmoid(pred)
        # TODO: Perform any necessary functions on the output such as clamping
        # TODO: Compute loss using ``criterion``
        loss = criterion(pred, label).cuda()

        # measure metrics and record loss
        m1 = metric1(pred.data, label)
        m2 = metric2(pred.data, label)
        losses.update(loss.item(), image.size(0))

        # train_loss.append([i, losses.val])
        # train_loss_avg.append([i, losses.avg])

        avg_m1.update(m1)
        avg_m2.update(m2)

        # TODO:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # compute gradient and do SGD step

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                epoch,
                i,
                len(train_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                avg_m1=avg_m1,
                avg_m2=avg_m2))

        # TODO: Visualize/log things as mentioned in handout
        # TODO: Visualize at appropriate intervals
        if USE_WANDB:
            wandb.log({
                'epoch': epoch,
                'Train/loss': loss.item(),
                'Train/metric1': m1,
                'Train/metric2': m2,
                'Train/step': train_step,
            })

        if USE_WANDB and i % log_freq == 0:
            index = torch.where(label[0] == 1)[0][0].item()
            heatmap = pred_feat[0, index, :, :].sigmoid()
            plt.imsave('heat_map.png', -heatmap.cpu().detach().numpy(), cmap='jet')
            heatmap = Image.open('heat_map.png')
            heatmap = heatmap.resize((512, 512))
            wandb.log({f'Train/heatmap': [wandb.Image(heatmap, caption="heatmap")]})
            wandb.log({f'Train/image': [wandb.Image(image[0, :, :, :], caption="image")]})

    # End of train()


def validate(val_loader, model, criterion, epoch=0, log_freq=10):

    global val_step

    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (data) in enumerate(val_loader):
        
        val_step += 1
        # TODO: Get inputs from the data dict
        image, label = data['image'].cuda(), data['label'].cuda()
        # TODO: Get output from model
        pred_feat = model(image)
        _, _, H, W = pred_feat.shape
        if args.arch == 'localizer_alexnet':
            pred = nn.MaxPool2d(kernel_size=(H, W))(pred_feat)
        if args.arch == 'localizer_alexnet_robust':
            pred = nn.AvgPool2d(kernel_size=(H, W))(pred_feat)
        pred = pred.squeeze()
        pred = torch.sigmoid(pred)
        # TODO: Perform any necessary functions on the output
        # TODO: Compute loss using ``criterion``
        loss = criterion(pred, label)

        # measure metrics and record loss
        m1 = metric1(pred.data, label)
        m2 = metric2(pred.data, label)
        losses.update(loss.item(), image.size(0))
        avg_m1.update(m1)
        avg_m2.update(m2)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                i,
                len(val_loader),
                batch_time=batch_time,
                loss=losses,
                avg_m1=avg_m1,
                avg_m2=avg_m2))

        # TODO: Visualize things as mentioned in handout
        # TODO: Visualize at appropriate intervals
        if USE_WANDB:
            wandb.log({
                'Val/metric1': m1,
                'Val/metric2': m2,
                'Val/step': val_step,
            })

        if USE_WANDB and i % log_freq == 0:
            index = torch.where(label[0] == 1)[0][0].item()
            heatmap = pred_feat[0, index, :, :].sigmoid()
            plt.imsave('heat_map.png', -heatmap.cpu().detach().numpy(), cmap='jet')
            heatmap = Image.open('heat_map.png')
            heatmap = heatmap.resize((512, 512))
            wandb.log({f'Val/heatmap': [wandb.Image(heatmap, caption="heatmap")]})
            wandb.log({f'Val/image': [wandb.Image(image[0, :, :, :], caption="image")]})

    print(' * Metric1 {avg_m1.avg:.3f} Metric2 {avg_m2.avg:.3f}'.format(
        avg_m1=avg_m1, avg_m2=avg_m2))

    return avg_m1.avg, avg_m2.avg


# TODO: You can make changes to this function if you wish (not necessary)
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def metric1(output, target):
    # TODO: Ignore for now - proceed till instructed

    ## PRECISION
    nclasses = target.shape[1]
    output_ = output.cpu().detach().numpy()
    target_ = target.cpu().detach().numpy()

    AP = []
    for class_idx in range(nclasses):
        gt_cls = target_[:, class_idx].astype('float32')
        pred_cls = output_[:, class_idx].astype('float32')
        if np.count_nonzero(gt_cls) != 0:
            pred_cls -= 1e-5 * gt_cls
            ap = sklearn.metrics.average_precision_score(gt_cls, pred_cls)
            AP.append(ap)
    return np.mean(AP)


def metric2(output, target):
    # TODO: Ignore for now - proceed till instructed

    ## RECALL
    output_ = output.cpu().detach().numpy()
    target_ = target.cpu().detach().numpy()
    gt_cls = target_.astype('float32')
    pred_cls = output_.astype('float32')
    recall = sklearn.metrics.recall_score(gt_cls, pred_cls > 0.5, average='samples')
    return recall


if __name__ == '__main__':
    main()

