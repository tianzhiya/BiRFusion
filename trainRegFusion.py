import os
from distutils.file_util import move_file

import torch
from torch import nn

from Fusion.utils import R_loss, P_loss, Fusionloss
from Module.model import BiRGenerator
from Tools.saver import Saver

from dataset.RegDataset import RegData

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


import argparse


class TrainRegFusionOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # data loader related
        self.parser.add_argument('--dataroot', type=str, default='./dataset/train/MSRS',
                                 help='path of data')
        self.parser.add_argument('--phase', type=str, default='train', help='phase for dataloading')
        self.parser.add_argument('--batch_size', type=int, default=1, help='batch size')
        self.parser.add_argument('--nThreads', type=int, default=8, help='# of threads for data loader')

        # ouptput related
        self.parser.add_argument('--name', type=str, default='FS_MSRS', help='folder name to save outputs')
        self.parser.add_argument('--display_dir', type=str, default='./logs', help='path for saving display results')
        self.parser.add_argument('--result_dir', type=str, default='./results',
                                 help='path for saving result images and models')
        self.parser.add_argument('--display_freq', type=int, default=50, help='freq (iteration) of display')
        self.parser.add_argument('--img_save_freq', type=int, default=500, help='freq (epoch) of saving images')
        self.parser.add_argument('--model_save_freq', type=int, default=50, help='freq (epoch) of saving models')
        self.parser.add_argument('--no_display_img', action='store_true', help='specified if no dispaly')

        # training related
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='type of learn rate decay')
        self.parser.add_argument('--n_ep', type=int, default=1, help='number of epochs')  # 400 * d_iter
        self.parser.add_argument('--n_ep_decay', type=int, default=1600,
                                 help='epoch start decay learning rate, set -1 if no decay')  # 200 * d_iter
        self.parser.add_argument('--resume', type=str, default='./checkpoint/MSRS.pth',
                                 help='specified the dir of saved models for resume the training')
        self.parser.add_argument('--gpu', type=int, default=0, help='gpu')
        self.parser.add_argument('--stage', type=str, default='FS', help='reg&fus (RF) or fus&seg (FS)')

        # segmentation related
        self.parser.add_argument('--dataroot_val', type=str, default='./dataset/test/MSRS/',
                                 help="data for segmentation validation")

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt


def jointTrain():
    pass


def trainRegFusion(opts):
    print("===> Loading datasets")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join('results', 'checkpoint', 'RegFusion.pth')
    model = BiRGenerator()
    model.resume(model_path)
    model = model.cuda()
    model.eval()

    from Fusion.net.net import net
    net = net().cuda()
    net.load_state_dict(torch.load('./Fusion/weights/epoch_180.pth'))

    dataset = RegData(opts)
    saver = Saver(opts)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.nThreads)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for ep in range(0, opts.n_ep):

        for it, (image_ir, image_vi, image_ir_warp, image_vi_warp, deformation) in enumerate(train_loader):
            # input data
            image_ir = image_ir.cuda(opts.gpu).detach()
            image_vi = image_vi.cuda(opts.gpu).detach()
            image_ir_warp = image_ir_warp.cuda(opts.gpu).detach()
            image_vi_warp = image_vi_warp.cuda(opts.gpu).detach()
            deformation = deformation.cuda(opts.gpu).detach()
            if len(image_ir.shape) > 4:
                image_ir = image_ir.squeeze(1)
                image_vi = image_vi.squeeze(1)
                image_ir_warp = image_ir_warp.squeeze(1)
                image_vi_warp = image_vi_warp.squeeze(1)
                deformation = deformation.squeeze(1)

            model.image_ir_RGB = image_ir
            model.image_vi_RGB = image_vi
            model.image_ir_warp_RGB = image_ir_warp
            model.image_vi_warp_RGB = image_vi_warp
            model.disp = deformation

            model.DM_opt.zero_grad()
            model.train()
            model.BiDirectGeneration()

            loss_Reg_T = trainAdvReg(image_ir, image_vi, model)

            lossFusion = trainDecFusion(image_vi, model, net)

            loss = loss_Reg_T + lossFusion

            print(f"lossreg: {loss_Reg_T.detach().item():.6f}, lossFusion: {lossFusion.detach().item():.6f}")

            optimizer.zero_grad()
            loss.backward(retain_graph=True)  # 只调用一次
            optimizer.step()

            nn.utils.clip_grad_norm_(model.enCodeDeCode.parameters(), 5)
            model.DM_opt.step()

    saver.write_model(ep, opts.n_ep, model)
    torch.save(net.state_dict(), './Fusion/weights/epoch.pth')


def trainDecFusion(image_vi, model, net):
    visY = model.image_vi_Reg_Y
    vis = image_vi
    vis = vis.cuda()
    ir = model.image_ir_Y
    ir = ir.cuda()
    L1, R1, X1 = net(visY)
    L2, R2, X2 = net(ir)
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
    lossFusion = loss2 * 1 + loss3 * 500 + loss4 * 1 + loss5 * 500 + loss_fusion[0]
    return lossFusion


def trainAdvReg(image_ir, image_vi, model):
    # ==== 判别器训练 ====
    for _ in range(2):
        model.trainDiscriminator(image_ir, image_vi)
    D_fake = model.dual_discrim(model.image_ir_Reg_RGB.detach(), model.image_vi_RGB)
    loss_adv = -torch.mean(D_fake)
    loss_reg_img = model.imgloss(model.image_ir_warp_RGB, model.image_ir_warp_fake_RGB,
                                 model.goodmask) + model.imgloss(
        model.image_ir_Reg_RGB, model.image_ir_RGB, model.goodmask * model.goodmask_inverse) + \
                   model.imgloss(model.image_vi_warp_RGB, model.image_vi_warp_fake_RGB,
                                 model.goodmask) + model.imgloss(
        model.image_vi_Reg_RGB, model.image_vi_RGB, model.goodmask * model.goodmask_inverse)
    loss_reg_field = model.weightfiledloss(model.image_ir_warp_RGB, model.image_vi_warp_fake_RGB,
                                           model.deformation_1['vis2ir'], model.disp.permute(0, 3, 1, 2)) + \
                     model.weightfiledloss(model.image_vi_warp_RGB, model.image_ir_warp_fake_RGB,
                                           model.deformation_2['ir2vis'], model.disp.permute(0, 3, 1, 2))
    loss_border_re = 0.1 * model.border_suppression(model.image_ir_Reg_RGB,
                                                    model.goodmask_inverse) + 0.1 * model.border_suppression(
        model.image_vi_Reg_RGB, model.goodmask_inverse) + \
                     model.border_suppression(model.image_ir_warp_fake_RGB,
                                              model.goodmask) + model.border_suppression(
        model.image_vi_warp_fake_RGB, model.goodmask)
    loss_Reg_T = loss_reg_img * 10 + loss_reg_field + loss_border_re + loss_adv
    return loss_Reg_T


pass

if __name__ == '__main__':
    parser = TrainRegFusionOptions()
    opts = parser.parse()
    trainRegFusion(opts)
