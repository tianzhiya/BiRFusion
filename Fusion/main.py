import os

import numpy as np
import torch

from Fusion.dataset import Fusion_dataset
from Fusion.utils import R_loss, P_loss, Fusionloss

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import random
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lrs
from torch.utils.data import DataLoader
from net.net import net


def seed_torch(seed=123456789):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def train():
    model.train()
    loss_print = 0
    for iteration, batch in enumerate(training_data_loader):

        vis, ir, file1, file2 = batch[0], batch[1], batch[2], batch[3]
        image_vis_ycrcb = RGB2YCrCb(vis)
        visY = image_vis_ycrcb[:, :1]

        vis = vis.cuda()
        ir = ir.cuda()
        L1, R1, X1 = model(visY)
        L2, R2, X2 = model(ir)
        R = torch.max(R1, R2)
        L = torch.max(L1, L2)
        fusion_imageY = R * L
        loss2 = R_loss(L1, R1, vis, X1)
        loss3 = P_loss(vis, X1)

        loss4 = R_loss(L2, R2, ir, X2)
        loss5 = P_loss(ir, X2)

        fusionLosss = Fusionloss()

        loss_fusion = fusionLosss(
            visY, ir, fusion_imageY
        )

        loss = loss2 * 1 + loss3 * 500 + loss4 * 1 + loss5 * 500 + loss_fusion[0]


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_print = loss_print + loss.item()
        if iteration % 10 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Learning rate: lr={}.".format(epoch,
                                                                                        iteration,
                                                                                        len(training_data_loader),
                                                                                        loss_print,
                                                                                        optimizer.param_groups[0][
                                                                                            'lr']))
            loss_print = 0


def checkpoint(epoch):
    model_out_path = opt.save_folder + "epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DECFusion')
    parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
    parser.add_argument('--nEpochs', type=int, default=400, help='number of epochs to train for')
    parser.add_argument('--snapshots', type=int, default=20, help='Snapshots')
    parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
    parser.add_argument('--decay', type=int, default='100', help='learning rate decay type')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
    parser.add_argument('--seed', type=int, default=123456789, help='random seed to use. Default=123')
    parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
    parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
    parser.add_argument('--output_folder', default='results/', help='Location to save checkpoint models')
    opt = parser.parse_args()

    seed_torch()
    cudnn.benchmark = True

    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    train_set = Fusion_dataset('train')
    training_data_loader = DataLoader(
        dataset=train_set,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    model = net().cuda()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

    milestones = []
    for i in range(1, opt.nEpochs + 1):
        if i % opt.decay == 0:
            milestones.append(i)

    scheduler = lrs.MultiStepLR(optimizer, milestones, opt.gamma)

    score_best = 0

    for epoch in range(opt.start_iter, opt.nEpochs + 1):
        train()
        scheduler.step()
        if epoch % opt.snapshots == 0:
            checkpoint(epoch)
