import argparse
import builtins
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
import torchvision.models.video as models
from torchvision.datasets.samplers.clip_sampler import DistributedSampler, RandomClipSampler
from dataset.kinetics import Kinetics400
import moco.loader
import moco.builder
from torch.utils.tensorboard import SummaryWriter
from utils.train_utils import adjust_learning_rate, accuracy, set_requires_grad, save_checkpoint, AverageMeter, \
    ProgressMeter
import warnings

warnings.filterwarnings("ignore")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch MoCo Video Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='r3d_18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-cs', '--crop_size', default=112, type=int, metavar='N',
                    help='crop size for video clip (default: 112)')
parser.add_argument('-fpc', '--frame_per_clip', default=16, type=int, metavar='N',
                    help='number of frame per video clip (default: 16)')
parser.add_argument('-sbc', '--step_between_clips', default=1, type=int, metavar='N',
                    help='number of steps between video clips (default: 1)')

parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_decay', '--learning-rate_decay', default=0.1, type=float,
                    metavar='LRD', help='learning rate decay', dest='lr_decay')
parser.add_argument('--warmup', action='store_true',
                    help='use warm up lr schedule')
parser.add_argument('--wp_lr', '--warmup_learning-rate', default=0.0025, type=float,
                    metavar='WLR', help='initial warmup learning rate', dest='wp_lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--log_dir', default='logs_moco', type=str,
                    help='path to the tensorboard log directory')
parser.add_argument('--ckp_dir', default='checkpoints_moco', type=str,
                    help='path to the moco model directory')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resumeG', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world_size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug_plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')


def main():
    args = parser.parse_args()
    print(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    print('ngpus_per_node {}'.format(ngpus_per_node))

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    netG = moco.builder.MaskGenerator()
    netD = moco.builder.MoCo(
        models.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
    print(netG)
    print(netD)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            netG.cuda(args.gpu)
            netD.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            netG = torch.nn.parallel.DistributedDataParallel(netG, device_ids=[args.gpu], find_unused_parameters=True)
            netD = torch.nn.parallel.DistributedDataParallel(netD, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            netG.cuda()
            netD.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            netG = torch.nn.parallel.DistributedDataParallel(netG)
            netD = torch.nn.parallel.DistributedDataParallel(netD)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        netG = netG.cuda(args.gpu)
        netD = netD.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        pass  # raise NotImplementedError("Only DistributedDataParallel is supported.") for debug on cpu
    # torch.cuda.synchronize()
    optimizer_g = torch.optim.SGD(netG.parameters(), args.lr,
                                  momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_d = torch.optim.SGD(netD.parameters(), args.lr,
                                  momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    G_criterion = nn.L1Loss().cuda(args.gpu)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            netD.load_state_dict(checkpoint['state_dict'])
            #optimizer_d.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

        if os.path.isfile(args.resumeG):
            print("=> loading checkpoint '{}'".format(args.resumeG))
            if args.gpu is None:
                checkpoint = torch.load(args.resumeG)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resumeG, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            netG.load_state_dict(checkpoint['state_dict'])
            #optimizer_g.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resumeG, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resumeG))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    video_augmentation = transforms.Compose(
        [
            transforms_video.ToTensorVideo(),
            transforms_video.RandomResizedCropVideo(args.crop_size, (0.2, 1)),
        ]
    )
    audio_augmentation = moco.loader.DummyAudioTransform()
    augmentation = {'video': video_augmentation, 'audio': audio_augmentation}

    augmentation_gpu = moco.loader.MoCoAugmentV2(args.crop_size) if args.aug_plus else moco.loader.MoCoAugment(
        args.crop_size)

    train_dataset = Kinetics400(
        traindir,
        args.frame_per_clip,
        args.step_between_clips,
        extensions='mp4',
        transform=augmentation,
        num_workers=4
    )

    train_sampler = RandomClipSampler(train_dataset.video_clips, 1)

    if args.distributed:
        # train_sampler = torch.utils.data.distributed.DistributedSampler(train_sampler)
        train_sampler = DistributedSampler(train_sampler)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True,
        multiprocessing_context="fork")
    if args.multiprocessing_distributed and args.gpu == 0:
        log_dir = "{}_bs={}_lr={}_cs={}_fpc={}".format(args.log_dir, args.batch_size, args.lr, args.crop_size,
                                                       args.frame_per_clip)
        writer = SummaryWriter(log_dir)
    else:
        writer = None
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer_d, epoch, args)
        adjust_learning_rate(optimizer_g, epoch, args)

        # train for one epoch
        train(train_loader, augmentation_gpu, criterion, G_criterion, netG, netD, optimizer_g, optimizer_d, epoch, args,
              writer)

        if (epoch + 1) % 10 == 0 and (not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                                               and args.rank % ngpus_per_node == 0)):
            ckp_dir = "{}_bs={}_lr={}_cs={}_fpc={}".format(args.ckp_dir, args.batch_size, args.lr, args.crop_size,
                                                           args.frame_per_clip)
            save_checkpoint(epoch, {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': netG.state_dict(),
            }, ckp_dir + '/netG', max_save=20, is_best=False)

            save_checkpoint(epoch, {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': netD.state_dict(),
            }, ckp_dir + '/netD', max_save=20, is_best=False)


def train(train_loader, augmentation_gpu, criterion, G_criterion, netG, netD, optimizer_g, optimizer_d, epoch, args,
          writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_g = AverageMeter('Loss_G', ':.4f')
    losses_d = AverageMeter('Loss_D', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses_g, losses_d, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    netG.train()
    netD.train()
    end = time.time()
    for i, (video, audio) in enumerate(train_loader):
        # measure data loading time
        # print('========================================')
        data_time.update(time.time() - end)
        if args.gpu is not None:
            video[0] = video[0].cuda(args.gpu, non_blocking=True)
            video[1] = video[1].cuda(args.gpu, non_blocking=True)
        video[0] = augmentation_gpu(video[0])
        video[1] = augmentation_gpu(video[1])
        im_q_fake = netG(video[0])
        q_fake, q_real, output, target = netD(im_q_fake, im_q=video[0], im_k=video[1])
        set_requires_grad([netD], False)  # Ds require no gradients when optimizing Gs
        optimizer_g.zero_grad()  # set G_A and G_B's gradients to zero
        loss_g = -100 * G_criterion(q_fake, q_real)
        loss_g.backward(retain_graph=True)
        optimizer_g.step()  # update generator's weights
        set_requires_grad([netD], True)
        optimizer_d.zero_grad()  # set discriminator's gradients to zero
        loss_d = criterion(output, target)
        loss_d.backward()
        optimizer_d.step()  # update D_A and D_B's weights
        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses_g.update(loss_g.item(), video[0].size(0))
        losses_d.update(loss_d.item(), video[0].size(0))
        top1.update(acc1[0], video[0].size(0))
        top5.update(acc5[0], video[0].size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)
            if writer is not None:
                total_iter = i + epoch * len(train_loader)
                writer.add_scalar('moco_train/loss', loss_d, total_iter)
                writer.add_scalar('moco_train/loss', loss_g, total_iter)
                writer.add_scalar('moco_train/acc1', acc1, total_iter)
                writer.add_scalar('moco_train/acc5', acc5, total_iter)
                writer.add_scalar('moco_train_avg/loss_g', losses_g.avg, total_iter)
                writer.add_scalar('moco_train_avg/loss_d', losses_d.avg, total_iter)
                writer.add_scalar('moco_train_avg/acc1', top1.avg, total_iter)
                writer.add_scalar('moco_train_avg/acc5', top5.avg, total_iter)
                # print("iter:%d: loss = %2f, acc1 = %2f, acc5 = %2f" %(total_iter, loss, acc1, acc5))
                # print("iter:%d: loss_avg = %2f, acc1_avg = %2f, acc5_avg = %2f" %(total_iter, losses.avg, top1.avg, top5.avg))

if __name__ == '__main__':
    main()
