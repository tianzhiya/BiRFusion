import torch
import argparse

from Fusion.net.net import net

from Module.model import BiRGenerator

from time import time
from tqdm import tqdm

import cv2
import kornia.utils as KU

import os
import numpy as np
from PIL import Image

from dataset.RegDataset import TestData

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='Fusion',
                    help='Reg for only image registration, Fusion for only image fusion, Reg&Fusion for image registration and fusion')
parser.add_argument('--dataset_name', type=str, default='MSRS', help='MSRS or M3FD')


def imsave(img, filename):
    # 如果是 tensor，则转换为 numpy
    if isinstance(img, torch.Tensor):
        img = img.squeeze().detach().cpu()
        img = KU.tensor_to_image(img) * 255.
    else:
        img = np.squeeze(img) * 255.

    img = img.astype(np.uint8)
    cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def save_fused_images(fused_image, name, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Step 1: Tensor to NumPy
    if isinstance(fused_image, torch.Tensor):
        fused_image = fused_image.detach().cpu().numpy()

    # Step 2: Remove batch dim → shape (C, H, W)
    if fused_image.ndim == 4:
        fused_image = fused_image[0]

    # Step 3: (C, H, W) → (H, W, C)
    if fused_image.shape[0] == 3:
        fused_image = np.transpose(fused_image, (1, 2, 0))

    # Step 4: Normalize to uint8 if needed
    if fused_image.dtype != np.uint8:
        fused_image = np.clip(fused_image, 0, 1) * 255
        fused_image = fused_image.astype(np.uint8)

    # Step 5: Convert BGR → RGB (only if needed)
    # Remove this line if your network outputs already RGB

    # Step 6: Save image
    image = Image.fromarray(fused_image)
    save_path = os.path.join(save_dir, name if name.endswith('.png') else name + '.png')
    image.save(save_path)

    print(f"[✓] Fusion image saved to {save_path}")


def RGB2YCrCb(input_im):
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


def fuse(ir, vi_tensor):
    ir = ir[:, 0:1]
    viYCrCb = RGB2YCrCb(vi_tensor)
    visY = viYCrCb[:, :1]
    VisL, VisR, VisInputY = net(visY)
    IrL, IrR, IrInput = net(ir)
    Lmax = torch.max(VisL, IrL)
    Rmax = torch.max(VisR, IrR)
    FuseY = Lmax * Rmax
    return FuseY


def getFuse_image(fuseY, vi_tensor):
    viYCrCb = RGB2YCrCb(vi_tensor)
    fusionImage_ycrcb = torch.cat(
        (fuseY, viYCrCb[:, 1:2, :,
                :], viYCrCb[:, 2:, :, :]),
        dim=1,
    )
    fusionResult_RGB = YCrCb2RGB(fusionImage_ycrcb)
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


if __name__ == '__main__':
    opts = parser.parse_args()
    img_path = os.path.join('./dataset/test', opts.dataset_name)
    # if opts.mode == 'Fusion':
    #     ir_path = os.path.join(img_path, 'ir')
    # else:
    ir_path = os.path.join(img_path, 'irWarp')
    vi_path = os.path.join(img_path, 'vi')
    model_path = os.path.join('checkpoint', 'RegFusion.pth')

    save_dir = os.path.join('./results', opts.mode, opts.dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    model = BiRGenerator()
    model.resume(model_path)
    model = model.cuda()
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net().to(device)
    net.load_state_dict(torch.load('./Fusion/weights/epoch_180.pth'))
    net.eval()

    test_dataloader = TestData(ir_path, vi_path)
    p_bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))

    for idx, [ir, vi, name] in p_bar:
        vi_tensor = vi.to(device)
        ir_tensor = ir.to(device)
        start = time()
        with torch.no_grad():
            # 设定不同的保存目录
            reg_save_dir = os.path.join(save_dir, 'registration')  # Registration结果保存目录
            fusion_save_dir = os.path.join(save_dir, 'registrationfusion')  # Fusion结果保存目录
            other_save_dir = os.path.join(save_dir, 'fusion')  # 其他结果保存目录

            # 确保目录存在
            os.makedirs(reg_save_dir, exist_ok=True)
            os.makedirs(fusion_save_dir, exist_ok=True)
            os.makedirs(other_save_dir, exist_ok=True)

            # 分别保存结果
            results_reg = model.registration_forward(ir_tensor, vi_tensor)
            imsave(results_reg, os.path.join(reg_save_dir, name))

            fuseY = fuse(results_reg, vi_tensor)
            fused_image = getFuse_image(fuseY, vi_tensor)

            save_fused_images(fused_image, name, fusion_save_dir)
            # fusionImage = model.fusion_forward(ir_tensor, vi_tensor)

            #
            # results_other = model.forward(ir_tensor, vi_tensor)
            # imsave(results_other, os.path.join(other_save_dir, name))

        end = time()

        if opts.mode == 'Reg':
            p_bar.set_description(f'registering {name} | time : {str(round(end - start, 4))}')
        elif opts.mode == 'Fusion':
            p_bar.set_description(f'fusing {name} | time : {str(round(end - start, 4))}')
        else:
            p_bar.set_description(f'registering and fusing {name} | time : {str(round(end - start, 4))}')
