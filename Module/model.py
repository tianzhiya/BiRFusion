import cv2

from Module.RetinexFusion import RetinexFusion
from Module.losses import *
import sys
import os
from Tools.utils import RGB2YCrCb, YCbCr2RGB

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from Module.modules import EncodeDecode, SpatialTransformer, get_scheduler, gaussian_weights_init

import torch
import torch.nn as nn
from torch.autograd import Variable




class Discriminator(nn.Module):
    def __init__(self, in_channels=6, base_channels=64):
        super(Discriminator, self).__init__()

        # 红外图1通道 + 可见光图1通道，所以输入是2通道
        self.model = nn.Sequential(
            # 输入 2通道，输出 64通道
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 64 -> 128
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 128 -> 256
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 256 -> 512
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 输出一个特征图，代表真假（PatchGAN思想）
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, image_ir, image_vi):
        # 拼接红外和可见光图
        x = torch.cat([image_ir, image_vi], dim=1)  # dim=1是通道方向
        out = self.model(x)
        return out


class D_VI(nn.Module):
    def __init__(self):
        super(D_VI, self).__init__()
        fliter = [6, 16, 32, 64, 128]
        kernel_size = 3
        stride = 2
        self.l1 = ConvLayer_dis(fliter[0], fliter[1], kernel_size, stride, use_relu=True)
        self.l2 = ConvLayer_dis(fliter[1], fliter[2], kernel_size, stride, use_relu=True)
        self.l3 = ConvLayer_dis(fliter[2], fliter[3], kernel_size, stride, use_relu=True)
        self.l4 = ConvLayer_dis(fliter[3], fliter[4], kernel_size, stride, use_relu=True)

        self.tanh = nn.Tanh()

    def forward(self, ir, vis):
        x = torch.cat((ir, vis), dim=1)
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = out.contiguous().view(out.size()[0], -1)
        linear = nn.Linear(out.size()[1], 1).cuda()
        out = self.tanh(linear(out))

        return out.squeeze()


class D_IR(nn.Module):
    def __init__(self):
        super(D_IR, self).__init__()
        fliter = [1, 16, 32, 64, 128]
        kernel_size = 3
        stride = 2
        self.l1 = ConvLayer_dis(fliter[0], fliter[1], kernel_size, stride, use_relu=True)
        self.l2 = ConvLayer_dis(fliter[1], fliter[2], kernel_size, stride, use_relu=True)
        self.l3 = ConvLayer_dis(fliter[2], fliter[3], kernel_size, stride, use_relu=True)
        self.l4 = ConvLayer_dis(fliter[3], fliter[4], kernel_size, stride, use_relu=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = out.view(out.size()[0], -1)
        linear = nn.Linear(out.size()[1], 1).cuda()
        out = self.tanh(linear(out))

        return out.squeeze()


class ConvLayer_dis(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_relu=True):
        super(ConvLayer_dis, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.use_relu = use_relu
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.conv2d(x)
        if self.use_relu is True:
            out = self.LeakyReLU(out)
        return out


class BiRGenerator(nn.Module):
    def __init__(self, opts=None):
        super(BiRGenerator, self).__init__()

        # parameters
        lr = 0.001
        # encoders
        self.enCodeDeCode = EncodeDecode()
        self.resume_flag = False
        self.ST = SpatialTransformer(256, 256, True)

        self.RetinexFusion = RetinexFusion()

        self.dual_discrim = Discriminator().cuda()
        self.optimizerD_dual = torch.optim.Adam(self.dual_discrim.parameters(), lr=1e-4, betas=(0.5, 0.999))

        # optimizers
        self.DM_opt = torch.optim.Adam(
            self.enCodeDeCode.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.00001)
        self.gradientloss = gradientloss()
        self.ncc_loss = ncc_loss()
        self.ssim_loss = ssimloss
        self.weights_sim = [1, 1, 0.2]
        self.weights_ssim1 = [0.3, 0.7]
        self.weights_ssim2 = [0.7, 0.3]

        self.deformation_1 = {}
        self.deformation_2 = {}
        self.border_mask = torch.zeros([1, 1, 256, 256])
        self.border_mask[:, :, 10:-10, 10:-10] = 1
        self.AP = nn.AvgPool2d(5, stride=1, padding=2)
        self.initialize()

    def initialize(self):
        self.enCodeDeCode.apply(gaussian_weights_init)

    def set_scheduler(self, opts, last_ep=0):
        self.DM_sch = get_scheduler(self.DM_opt, opts, last_ep)

    def setgpu(self, gpu):
        self.gpu = gpu
        self.enCodeDeCode.cuda(self.gpu)

    def test_forward(self, image_ir, image_vi):
        deformation = self.enCodeDeCode(image_ir, image_vi)
        image_ir_Reg = self.ST(image_ir, deformation['ir2vis'])
        image_fusion = self.FN(image_ir_Reg, image_vi)
        return image_fusion

    def generate_mask(self):
        flow = self.ST.grid + self.disp
        goodmask = torch.logical_and(flow >= -1, flow <= 1)
        if self.border_mask.device != goodmask.device:
            self.border_mask = self.border_mask.to(goodmask.device)
        self.goodmask = torch.logical_and(goodmask[..., 0], goodmask[..., 1]).unsqueeze(1) * 1.0
        for i in range(2):
            self.goodmask = (self.AP(self.goodmask) > 0.3).float()

        flow = self.ST.grid - self.disp
        goodmask = F.grid_sample(self.goodmask, flow)
        self.goodmask_inverse = goodmask

    def forward(self, ir, vi):
        disp = self.enCodeDeCode(ir, vi)['ir2vis']
        ir_reg = self.ST(ir, disp)
        vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(vi)
        fu = self.FN(ir_reg[:, 0:1], vi_Y)
        fu = YCbCr2RGB(fu, vi_Cb, vi_Cr)
        return fu

    def registration_forward(self, ir, vi):
        disp = self.enCodeDeCode(ir, vi)['ir2vis']
        ir_reg = self.ST(ir, disp)
        return ir_reg

    def RGB2YCrCb(self, input_im):
        im_flat = input_im.transpose(1, 3).transpose(
            1, 2).reshape(-1, 3)  # (nhw,c)
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

    def YCrCb2RGB(self, input_im):
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

    def fusion_forward(self, ir, vi):
        viYCrCb = self.RGB2YCrCb(vi)
        vi_Y = viYCrCb[:, 0:1, :, :]
        ir_yL, ir_yR, ir_yInput, vi_reg_yL, vi_reg_yR, ivi_reg_yInput, fu = self.RetinexFusion(ir[:, 0:1], vi_Y)

        fu = fu.clamp(0, 1)  # 确保亮度有效
        cr = viYCrCb[:, 1:2, :, :].clamp(0, 1)  # 保证颜色通道范围
        cb = viYCrCb[:, 2:, :, :].clamp(0, 1)

        fusionResult_ycrcb = torch.cat(
            (fu, cr,
             cb),
            dim=1,
        )

        fusionResult_RGB = self.YCrCb2RGB(fusionResult_ycrcb)
        ones = torch.ones_like(fusionResult_RGB)
        zeros = torch.zeros_like(fusionResult_RGB)
        fusionResult_RGB = torch.where(fusionResult_RGB > ones, ones, fusionResult_RGB)
        fusionResult_RGB = torch.where(
            fusionResult_RGB < zeros, zeros, fusionResult_RGB)
        fused_image = fusionResult_RGB.cpu().numpy()
        fused_image = fused_image.transpose((0, 2, 3, 1))
        fused_image = (fused_image - np.min(fused_image)) / (
                np.max(fused_image) - np.min(fused_image)
        )
        fused_image = np.uint8(255.0 * fused_image)

        return fused_image

    def normalize_tensor_to_uint8(self, tensor):

        tensor = tensor.squeeze()  # 去掉 batch 或 channel 维度
        min_val = tensor.min()
        max_val = tensor.max()
        norm_tensor = (tensor - min_val) / (max_val - min_val + 1e-8)  # 归一化到 [0,1]
        tensor_255 = (norm_tensor * 255.0).clamp(0, 255).byte()  # 转为 uint8
        return tensor_255.cpu().numpy()

    def imgSave(self, img, filename):
        img = img.squeeze().detach().cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # 避免除零
        img = (img * 255).astype(np.uint8)  # 转换到 0-255 范围
        cv2.imwrite(filename, img)

    def BiDirectGeneration(self):
        b = self.image_ir_warp_RGB.shape[0]
        ir_stack = torch.cat([self.image_ir_warp_RGB, self.image_ir_RGB])
        vi_stack = torch.cat([self.image_vi_RGB, self.image_vi_warp_RGB])
        deformation = self.enCodeDeCode(ir_stack, vi_stack, type='bi')

        self.deformation_1['vis2ir'], self.deformation_2['vis2ir'] = deformation['vis2ir'][0:b, ...], deformation[
                                                                                                          'vis2ir'][b:,
                                                                                                      ...]
        self.deformation_1['ir2vis'], self.deformation_2['ir2vis'] = deformation['ir2vis'][0:b, ...], deformation[
                                                                                                          'ir2vis'][b:,
                                                                                                      ...]
        img_stack = torch.cat([ir_stack, vi_stack])
        disp_stack = torch.cat([deformation['ir2vis'], deformation['vis2ir']])
        img_warp_stack = self.ST(img_stack, disp_stack)

        self.image_ir_Reg_RGB, self.image_ir_warp_fake_RGB, self.image_vi_warp_fake_RGB, self.image_vi_Reg_RGB = torch.split(
            img_warp_stack, b, dim=0)

        self.image_vi_Y, self.image_vi_Cb, self.image_vi_Cr = RGB2YCrCb(self.image_vi_RGB)
        self.image_vi_Reg_Y, self.image_vi_Reg_Cb, self.image_vi_Reg_Cr = RGB2YCrCb(self.image_vi_Reg_RGB)
        self.image_ir_Y = self.image_ir_RGB[:, 0:1, ...]
        self.image_ir_Reg_Y = self.image_ir_Reg_RGB[:, 0:1, ...]

        ir_stack_reg_Y = torch.cat([self.image_ir_Y, self.image_ir_Reg_Y])
        vi_stack_reg_Y = torch.cat([self.image_vi_Reg_Y, self.image_vi_Y])

        self.ir_stack_reg_Y = ir_stack_reg_Y
        self.vi_stack_reg_Y = vi_stack_reg_Y

        # fusion_img = self.FN(ir_stack_reg_Y, vi_stack_reg_Y)
        self.imgSave(self.image_ir_Y, f"../temp/image_ir_Y.png")
        self.imgSave(self.image_vi_Reg_Y,
                     f"../temp/image_vi_Reg_Y.png")

        self.generate_mask()
        self.image_display = torch.cat((self.image_ir_RGB[0:1, 0:1], self.image_ir_warp_RGB[0:1, 0:1],
                                        self.image_ir_Reg_RGB[0:1, 0:1],
                                        (self.image_vi_RGB - self.image_vi_warp_RGB)[0:1].abs().mean(dim=1,
                                                                                                     keepdim=True),

                                        self.image_vi_Y[0:1], RGB2YCrCb(self.image_vi_warp_RGB[0:1])[0],
                                        self.image_vi_Reg_Y[0:1],
                                        (self.image_vi_RGB - self.image_vi_Reg_RGB)[0:1].abs().mean(dim=1,
                                                                                                    keepdim=True),
                                        ), dim=0).detach()

    def train_forward_FS(self):
        self.image_vi_Y, self.image_vi_Cb, self.image_vi_Cr = RGB2YCrCb(self.image_vi_RGB)
        self.image_ir_Y = self.image_ir_RGB[:, 0:1, ...]

        fusion_img = self.FN(self.image_ir_Y, self.image_vi_Y)
        self.image_fusion = fusion_img
        self.fused_image_RGB = YCbCr2RGB(self.image_fusion, self.image_vi_Cb, self.image_vi_Cr)



    def trainReg(self, image_ir, image_vi, image_ir_warp, image_vi_warp, disp):
        self.image_ir_RGB = image_ir
        self.image_vi_RGB = image_vi
        self.image_ir_warp_RGB = image_ir_warp
        self.image_vi_warp_RGB = image_vi_warp
        self.disp = disp

        self.DM_opt.zero_grad()
        self.BiDirectGeneration()

        # ==== 判别器训练 ====
        for _ in range(2):
            self.trainDiscriminator(image_ir, image_vi)

        self.backward_RF()
        nn.utils.clip_grad_norm_(self.enCodeDeCode.parameters(), 5)
        self.DM_opt.step()

    def trainDiscriminator(self, image_ir, image_vi):
        gamma_ = 10
        all_d_loss = 0
        for _ in range(2):
            # 真实样本对：原红外 + 可见光
            D_real = self.dual_discrim(image_ir.detach(), image_vi)
            D_real_loss = - torch.mean(D_real)
            # 伪造样本对：生成的配准红外 + 可见光
            D_fake = self.dual_discrim(self.image_ir_Reg_RGB.detach(), image_vi)
            D_fake_loss = torch.mean(D_fake)

            # 梯度惩罚
            alpha = torch.rand(image_ir.size(0), 1, 1, 1).cuda().expand_as(image_ir)
            interpolated = Variable(alpha * image_ir.data + (1 - alpha) * self.image_ir_Reg_RGB.data,
                                    requires_grad=True)
            D_interpolated = self.dual_discrim(interpolated, image_vi)
            grad = torch.autograd.grad(outputs=D_interpolated,
                                       inputs=interpolated,
                                       grad_outputs=torch.ones_like(D_interpolated),
                                       create_graph=True,
                                       retain_graph=True,
                                       only_inputs=True)[0]
            grad = grad.view(grad.size(0), -1)
            grad_norm = torch.sqrt(torch.sum(grad ** 2, dim=1) + 1e-12)
            grad_penalty = torch.mean((grad_norm - 1) ** 2)

            # 判别器loss
            D_loss = D_real_loss + D_fake_loss + gamma_ * grad_penalty
            all_d_loss += D_loss.item()

            self.D_loss = D_loss

            # 更新判别器
            self.optimizerD_dual.zero_grad()
            D_loss.backward(retain_graph=True)
            self.optimizerD_dual.step()

    def imgloss(self, src, tgt, mask=1, weights=[0.1, 0.9]):
        return weights[0] * (l1loss(src, tgt, mask) + l2loss(src, tgt, mask)) + weights[1] * self.gradientloss(src, tgt,
                                                                                                               mask)

    def weightfiledloss(self, ref, tgt, disp, disp_gt):
        ref = (ref - ref.mean(dim=[-1, -2], keepdim=True)) / (ref.std(dim=[-1, -2], keepdim=True) + 1e-5)
        tgt = (tgt - tgt.mean(dim=[-1, -2], keepdim=True)) / (tgt.std(dim=[-1, -2], keepdim=True) + 1e-5)
        g_ref = KF.spatial_gradient(ref, order=2).mean(dim=1).abs().sum(dim=1).detach().unsqueeze(1)
        g_tgt = KF.spatial_gradient(tgt, order=2).mean(dim=1).abs().sum(dim=1).detach().unsqueeze(1)
        w = (((g_ref + g_tgt)) * 2 + 1) * self.border_mask
        return (w * (1000 * (disp - disp_gt).abs().clamp(min=1e-2).pow(2))).mean()

    def border_suppression(self, img, mask):
        return (img * (1 - mask)).mean()




    def calDecLoss(self, L1, R1, X1, L2, R2, X2, vis, ir):
        R = torch.max(R1, R2)
        L = torch.max(L1, L2)
        fusion_imageY = R * L
        loss2 = self.R_loss(L1, R1, vis, X1)
        loss3 = self.P_loss(vis, X1)

        loss4 = self.R_loss(L2, R2, ir, X2)
        loss5 = self.P_loss(ir, X2)

        fusionLosss = Fusionloss()

        loss_fusion = fusionLosss(
            self.vi_stack_reg_Y, self.ir_stack_reg_Y, fusion_imageY
        )

        loss = loss2 * 1 + loss3 * 500 + loss4 * 1 + loss5 * 500 + loss_fusion[0]

        return loss

    def R_loss(self, L1, R1, im1, X1):
        max_rgb1, _ = torch.max(im1, 1)
        max_rgb1 = max_rgb1.unsqueeze(1)
        loss1 = torch.nn.MSELoss()(L1 * R1, X1) + torch.nn.MSELoss()(R1, X1 / L1.detach())
        loss2 = torch.nn.MSELoss()(L1, max_rgb1) + self.tv_loss(L1)
        return loss1 + loss2

    def P_loss(self, im1, X1):
        loss = torch.nn.MSELoss()(im1, X1)
        return loss

    def gradient(self, img):
        height = img.size(2)
        width = img.size(3)
        gradient_h = (img[:, :, 2:, :] - img[:, :, :height - 2, :]).abs()
        gradient_w = (img[:, :, :, 2:] - img[:, :, :, :width - 2]).abs()
        return gradient_h, gradient_w

    def tv_loss(slef, illumination):
        gradient_illu_h, gradient_illu_w = slef.gradient(illumination)
        loss_h = gradient_illu_h
        loss_w = gradient_illu_w
        loss = loss_h.mean() + loss_w.mean()
        return loss

    def backward_RF(self):
        D_fake = self.dual_discrim(self.image_ir_Reg_RGB.detach(), self.image_vi_RGB)
        loss_adv = -torch.mean(D_fake)

        loss_reg_img = self.imgloss(self.image_ir_warp_RGB, self.image_ir_warp_fake_RGB, self.goodmask) + self.imgloss(
            self.image_ir_Reg_RGB, self.image_ir_RGB, self.goodmask * self.goodmask_inverse) + \
                       self.imgloss(self.image_vi_warp_RGB, self.image_vi_warp_fake_RGB, self.goodmask) + self.imgloss(
            self.image_vi_Reg_RGB, self.image_vi_RGB, self.goodmask * self.goodmask_inverse)
        loss_reg_field = self.weightfiledloss(self.image_ir_warp_RGB, self.image_vi_warp_fake_RGB,
                                              self.deformation_1['vis2ir'], self.disp.permute(0, 3, 1, 2)) + \
                         self.weightfiledloss(self.image_vi_warp_RGB, self.image_ir_warp_fake_RGB,
                                              self.deformation_2['ir2vis'], self.disp.permute(0, 3, 1, 2))
        loss_smooth = smoothloss(self.deformation_1['vis2ir']) + smoothloss(self.deformation_1['ir2vis']) + \
                      smoothloss(self.deformation_2['vis2ir']) + smoothloss(self.deformation_2['ir2vis'])

        loss_border_re = 0.1 * self.border_suppression(self.image_ir_Reg_RGB,
                                                       self.goodmask_inverse) + 0.1 * self.border_suppression(
            self.image_vi_Reg_RGB, self.goodmask_inverse) + \
                         self.border_suppression(self.image_ir_warp_fake_RGB, self.goodmask) + self.border_suppression(
            self.image_vi_warp_fake_RGB, self.goodmask)

        assert not loss_reg_img is None, 'loss_reg_img is None'
        assert not loss_reg_field is None, 'loss_reg_filed is None'
        assert not loss_smooth is None, 'loss_smooth is None'

        loss_total = loss_reg_img * 10 + loss_reg_field + loss_border_re + loss_adv

        (loss_total).backward()

        self.loss_reg_img = loss_reg_img
        self.loss_reg_field = loss_reg_field

        self.loss_smooth = loss_smooth

        self.loss_total = loss_total


    def update_lr(self):
        self.DM_sch.step()

    def resume(self, model_dir, train=True):
        self.resume_flag = True
        checkpoint = torch.load(model_dir)
        # weight
        try:
            self.enCodeDeCode.load_state_dict({k: v for k, v in checkpoint['DM'].items() if k in self.enCodeDeCode.state_dict()})
        except:
            pass
        try:
            self.FN.load_state_dict({k: v for k, v in checkpoint['FN'].items() if k in self.FN.state_dict()})
        except:
            pass

        if train:

            self.DM_opt.param_groups[0]['initial_lr'] = 0.001

        return checkpoint['ep'], checkpoint['total_it']

    def save(self, filename, ep, total_it):
        state = {
            'DM': self.enCodeDeCode.state_dict(),
            'DM_opt': self.DM_opt.state_dict(),
            'ep': ep,
            'total_it': total_it
        }
        torch.save(state, filename)
        return

    def normalize_image(self, x):
        return x[:, 0:1, :, :]


class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_vis, image_ir, generate_img):
        image_y = image_vis[:, :1, :, :]
        x_in_max = torch.max(image_y, image_ir)
        loss_in = F.l1_loss(x_in_max, generate_img)
        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.max(y_grad, ir_grad)
        loss_grad = F.l1_loss(x_grad_joint, generate_img_grad)
        loss_total = loss_in + 10 * loss_grad
        return loss_total, loss_in, loss_grad


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)
