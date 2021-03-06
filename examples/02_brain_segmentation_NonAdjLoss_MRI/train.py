import argparse
import math
import os
import time
import torch

from torchmed.utils.logger_plotter import LoggerPlotter, MetricLogger
from torchmed.utils.loss import dice_loss

from architecture import ModSegNet
from datasets.mappings import Miccai12Mapping
from datasets.training import MICCAI2012Dataset, SemiDataset
from loss import AdjacencyEstimator, LambdaControl
from utils import *

parser = argparse.ArgumentParser(
    description='PyTorch Automatic Segmentation of brain MRI - Training')
parser.add_argument('data', metavar='DIR', help='path to the train dataset')
parser.add_argument('output_dir', default='', metavar='OUTPUT_DIR',
                    help='path to the output directory (default: current dir)')

parser.add_argument('--semi-data-dir', metavar='SEMI_DATA_DIR', default=None,
                    help='path to the dataset for semi-supervision')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run (default: 200)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--exp-id', default='willis', type=str, metavar='EXP_ID',
                    help='name of the experiment')


def main():
    global args, nb_classes
    args = parser.parse_args()
    args.semi_data_dir = None if args.semi_data_dir == 'None' else args.semi_data_dir

    #####
    #
    #                              Logging
    #
    #####
    log_plot = LoggerPlotter(args.output_dir,
                             ['train.csv', 'validation.csv',
                              'average_train.csv', 'average_validation.csv',
                              'learning_rates.csv'],
                             ['loss.png',
                              'average_dice.png', 'learning_rate.png',
                              'losses_train.png', 'losses_validation.png',
                              'average_iou.png', 'average_nonadjloss.png'])

    log_metrics = ';'.join(['epoch', 'cross_entropy_loss', 'dice_loss',
                            'dice_metric', 'iou_metric', 'nonadj_loss'])
    log_plot.log('train.csv', log_metrics)
    log_plot.log('validation.csv', log_metrics)
    log_plot.log('average_train.csv', log_metrics)
    log_plot.log('average_validation.csv', log_metrics)
    log_plot.log('learning_rates.csv', ';'.join(['epoch', 'lr', 'dice', 'nonadjloss']))

    ce = MetricLogger('ce', 'CE Loss', '{:.2e}', '{:.3e}', '{:.5f}')
    dice_ls = MetricLogger('dice', 'Dice Loss', '{:.2e}', '{:.3e}', '{:.5f}')
    dice_met = MetricLogger('dice_metric', 'Dice', '{:>6.3f}', '{:>6.3f}', '{:.5f}')
    iou_met = MetricLogger('iou_metric', 'IoU', '{:>6.3f}', '{:>6.3f}', '{:.5f}')
    nonadj_met = MetricLogger('nonadjloss', 'NonAdjLoss', '{:.2e}', '{:.3e}', '{:.5e}')
    log_plot.add_metric([ce, dice_ls, dice_met, iou_met, nonadj_met])

    #####
    #
    #                           Model loading
    #
    #####
    torch.backends.cudnn.benchmark = True
    nb_classes = Miccai12Mapping().nb_classes
    model = ModSegNet(num_classes=nb_classes, n_init_features=7).cuda()

    # optionally resume from a checkpoint
    if args.resume is not None and args.resume != 'None':
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            try:
                model.load_state_dict(checkpoint['state_dict'])
            except RuntimeError as e:
                model = torch.nn.DataParallel(model).cuda()
                model.load_state_dict(checkpoint['state_dict'])
                model = model.module

            if 'epoch' in checkpoint.keys():
                print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    #####
    #
    #                            Data loading
    #
    #####
    sup_batch_size = args.batch_size
    semi_batch_size = None
    semi_loader = None
    if args.semi_data_dir is not None:
        semi_batch_size = args.batch_size * 3
        semi_dataset = SemiDataset(args.semi_data_dir, args.workers // 2)
        semi_loader = torch.utils.data.DataLoader(
            semi_dataset.train_dataset,
            batch_size=semi_batch_size,
            shuffle=False,
            num_workers=args.workers // 2,
            pin_memory=True)

    image_dataset = MICCAI2012Dataset(args.data, args.workers)
    train_loader = torch.utils.data.DataLoader(
        image_dataset.train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        image_dataset.validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    #####
    #
    #                         Optimization / Loss
    #
    #####
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # median frequency balancing (useful for highly imbalanced problems)
    # Error Corrective Boosting https://arxiv.org/abs/1705.00938
    weights = image_dataset.class_freq.median() / image_dataset.class_freq
    nll_loss = torch.nn.NLLLoss(ignore_index=-1, weight=weights).cuda()
    dice_weight = 5

    # binarize matrix and reverse to obtain only abnormal adjacencies
    adjacency_mat = 1 - (image_dataset.adjacency_mat > 0).float()
    adjacency_mat = adjacency_mat.cuda()
    tuning_epoch = (args.epochs * 7) // 10
    # utility class to optimize lambda (weighting term of the NonAdjLoss)
    lambda_control = LambdaControl(adjacency_mat, tuning_epoch)

    write_config(model, args, len(train_loader), len(val_loader))
    start_time = time.time()

    #####
    #                        Train / Validation Loop
    #####
    for epoch in range(args.start_epoch, args.epochs):
        # update learning rate with poly rate policy
        lr = poly_lr_scheduler(optimizer, args.lr, epoch, 1, args.epochs)

        # gets the weighting term of the penalization
        train_nan_flag = False
        nonadj_config = lambda_control.get_config()

        # Nan values can happen during training. We just need to monitor
        # them for the update of `lambda_`.
        try:
            if semi_loader is None:
                train(train_loader, model, optimizer, nll_loss, nonadj_config,
                      dice_weight, epoch, log_plot)
            else:
                train_semi(train_loader, semi_loader, model, optimizer,
                           nll_loss, nonadj_config, dice_weight, epoch, log_plot)
        except ValueError as ve:
            print('--NaN generated during the training')
            train_nan_flag = True
        except Exception as ex:
            raise ex

        # train metrics
        train_dice = log_plot.metrics['dice_metric'].avg
        train_nonadjloss = log_plot.metrics['nonadjloss'].avg

        validate(val_loader, model, nll_loss, nonadj_config, dice_weight, epoch, log_plot)

        # validation metrics
        val_dice = log_plot.metrics['dice_metric'].avg
        val_iou = log_plot.metrics['iou_metric'].avg
        val_nonadjloss = log_plot.metrics['nonadjloss'].avg

        # update lambda
        ret = lambda_control.update(epoch, train_dice, train_nonadjloss,
                                    val_dice, train_nan_flag)
        # exit training or reload to a previous model
        if ret == -1:
            break
        elif ret == -2:
            # reload
            filename = os.path.join(args.output_dir,
                                    'checkpoint_' + str(lambda_control.last_good_epoch) + '.pth.tar')
            checkpoint = torch.load(filename)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
            lambda_control.train_nan_count = 0

        # log learning rate and update figures
        log_plot.log('learning_rates.csv', '{:d};{:.7f};{:.3f};{:.5e}'.format(epoch, lr, dice_weight, nonadj_config[-2]))
        update_figures(log_plot)

        # save checkpoint
        if train_nan_flag is False and lambda_control.last_good_epoch == epoch:
            save_checkpoint({
                'epoch': epoch,
                'experiment_id': args.exp_id,
                'state_dict': model.state_dict(),
                'dice_metric': val_dice,
                'iou_metric': val_iou,
                'nonadjloss': val_nonadjloss
            }, epoch, args.output_dir)

    end_time = time.time()
    write_end_config(args, end_time - start_time)

    #####
    #
    #                            Model Selection
    #
    #####
    find_best_model(args.output_dir)
    os.system('rm {}'.format(os.path.join(args.output_dir, 'checkpoint_*')))


def train(train_loader, model, optimizer, nll_loss, nonadj_config, dice_weight, epoch, logger):
    logger.clear_metrics()

    model.train()
    adjacencyLayer = AdjacencyEstimator(nb_classes).train().cuda()
    # description of the variables in the same order :
    # ground truth, number of maximal abnormal connections, lambda parameter
    # boolean flag indicating if NonAdjLoss is used for optimization
    gt_graph, nb_conn_ab_max, lambda_coef, activate_nonadjloss = nonadj_config

    for i, (_, img, target) in enumerate(train_loader):
        target_gpu = target.cuda()

        # compute output
        output = model(img.cuda())

        # segmentation loss
        ce = nll_loss(output, target_gpu)
        dice = dice_loss(output.exp(), target_gpu, ignore_index=-1)
        dice *= dice_weight

        ##########
        #
        #                      Non-Adjacency loss
        #
        ##########

        # labels non-adjacency matrix evaluated from the segmentation output
        nonadjloss = adjacencyLayer(output.exp()) * gt_graph

        # sum and normalize to [0, 1] range, weight by lambda
        nonadjloss = (nonadjloss.sum() / nb_conn_ab_max) * lambda_coef

        loss = ce + dice

        # if NonAdjLoss is activated
        if activate_nonadjloss:
            loss += nonadjloss
        else:
            nonadjloss.detach_()

        if math.isnan(loss.item()):
            raise ValueError("Loss is NaN")

        # compute gradient and do optim step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure dice metric
        indices = output.data.max(dim=1)[1].cpu().numpy()
        metrics_res = eval_metrics(indices, target.numpy())
        dice_sim, iou_metric = metrics_res
        logger.metrics['ce'].update(ce.data.item(), img.size(0))
        logger.metrics['dice'].update(dice.data.item(), img.size(0))
        logger.metrics['dice_metric'].update(dice_sim, img.size(0))
        logger.metrics['iou_metric'].update(iou_metric, img.size(0))
        logger.metrics['nonadjloss'].update(nonadjloss, img.size(0))

        if i % args.print_freq == 0:
            logger.print_metrics(epoch, i, len(train_loader))
            logger.write_val_metrics(epoch + (i / len(train_loader)), 'train.csv')

    logger.write_avg_metrics(epoch, 'average_train.csv')


def train_semi(train_loader, semi_loader, model, optimizer, nll_loss,
               nonadj_config, dice_weight, epoch, logger):
    logger.clear_metrics()

    model.train()
    adjacencyLayer = AdjacencyEstimator(nb_classes).train().cuda()
    # description of the variables in the same order :
    # ground truth, number of maximal abnormal connections, lambda parameter
    # boolean flag indicating if NonAdjLoss is used for optimization
    gt_graph, nb_conn_ab_max, lambda_coef, activate_nonadjloss = nonadj_config

    miccai_iter = iter(train_loader)
    for i, (_, batch_oasis) in enumerate(semi_loader):
        try:
            batch_miccai = next(miccai_iter)
        except StopIteration:
            miccai_iter = iter(train_loader)
            batch_miccai = next(miccai_iter)

        oasis_size = batch_oasis.size(0)
        _, batch_img, target = batch_miccai
        target_gpu = target.cuda()
        batch_img = torch.cat([batch_img, batch_oasis], dim=0)

        # compute output
        output = model(batch_img.cuda())

        ##########
        #
        #                      Non-Adjacency loss
        #
        ##########

        # labels non-adjacency matrix evaluated from the segmentation output
        nonadjloss = adjacencyLayer(output.exp()) * gt_graph

        # sum and normalize to [0, 1] range, weight by lambda
        nonadjloss = (nonadjloss.sum() / nb_conn_ab_max) * lambda_coef

        # segmentation loss
        output = output[:-oasis_size]
        ce = nll_loss(output, target_gpu)
        dice = dice_loss(output.exp(), target_gpu, ignore_index=-1)
        dice *= dice_weight

        loss = ce + dice

        # if NonAdjLoss is activated
        if activate_nonadjloss:
            loss += nonadjloss
        else:
            nonadjloss.detach_()

        if math.isnan(loss.item()):
            raise ValueError("Loss is NaN")

        # compute gradient and do optim step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure dice metric
        indices = output.data.max(dim=1)[1].cpu().numpy()
        metrics_res = eval_metrics(indices, target.numpy())
        dice_sim, iou_metric = metrics_res
        logger.metrics['ce'].update(ce.data.item(), batch_img.size(0) - oasis_size)
        logger.metrics['dice'].update(dice.data.item(), batch_img.size(0) - oasis_size)
        logger.metrics['dice_metric'].update(dice_sim, batch_img.size(0) - oasis_size)
        logger.metrics['iou_metric'].update(iou_metric, batch_img.size(0) - oasis_size)
        logger.metrics['nonadjloss'].update(nonadjloss, batch_img.size(0))

        if i % args.print_freq == 0:
            logger.print_metrics(epoch, i, len(semi_loader))
            logger.write_val_metrics(epoch + (i / len(semi_loader)), 'train.csv')

    logger.write_avg_metrics(epoch, 'average_train.csv')


def validate(val_loader, model, nll_loss, nonadj_config, dice_weight, epoch, logger):
    logger.clear_metrics()

    model.eval()
    adjacencyLayer = AdjacencyEstimator(nb_classes).train().cuda()
    gt_graph, nb_conn_ab_max, lambda_coef, activate_nonadjloss = nonadj_config

    with torch.no_grad():
        for i, (_, img, target) in enumerate(val_loader):
            target_gpu = target.cuda()

            # compute output
            output = model(img.cuda())
            ce = nll_loss(output, target_gpu)
            dice = dice_loss(output.exp(), target_gpu, ignore_index=-1)
            dice *= dice_weight

            # Non-Adjacency loss
            nonadjloss = adjacencyLayer(output.exp())
            nonadjloss = gt_graph * nonadjloss
            nonadjloss = nonadjloss.sum() / nb_conn_ab_max

            # measure dice metric
            indices = output.data.max(dim=1)[1].cpu().numpy()
            metrics_res = eval_metrics(indices, target.numpy())
            dice_sim, iou_metric = metrics_res
            logger.metrics['ce'].update(ce.data.item(), img.size(0))
            logger.metrics['dice'].update(dice.data.item(), img.size(0))
            logger.metrics['dice_metric'].update(dice_sim, img.size(0))
            logger.metrics['iou_metric'].update(iou_metric, img.size(0))
            logger.metrics['nonadjloss'].update(nonadjloss, img.size(0))

            if i % args.print_freq == 0:
                logger.print_metrics(epoch, i, len(val_loader), 'test')
                logger.write_val_metrics(epoch + (i / len(val_loader)),
                                         'validation.csv')

        logger.write_avg_metrics(epoch, 'average_validation.csv')


if __name__ == '__main__':
    main()
