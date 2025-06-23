import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

def deform_image_tensor(image_tensor, dx=10, dy=5):
    _, _, H, W = image_tensor.shape

    # 创建标准网格
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W),
        indexing='ij'
    )
    grid = torch.stack((grid_x, grid_y), dim=-1)  # [H, W, 2]

    # 归一化的位移
    norm_dx = dx / (W / 2)
    norm_dy = dy / (H / 2)
    disp = torch.zeros_like(grid)
    disp[..., 0] = norm_dx
    disp[..., 1] = norm_dy

    sampling_grid = (grid + disp).unsqueeze(0)  # [1, H, W, 2]
    deformed_tensor = F.grid_sample(image_tensor, sampling_grid, align_corners=True)

    return deformed_tensor, disp

def process_images_in_folder(input_dir, output_image_dir, output_disp_dir, dx=10, dy=5):
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_disp_dir, exist_ok=True)

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))]

    for fname in image_files:
        input_path = os.path.join(input_dir, fname)
        print(f"Processing: {input_path}")

        # 读取并转为 Tensor
        image = cv2.imread(input_path)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # 变形处理
        deformed_tensor, disp = deform_image_tensor(image_tensor, dx=dx, dy=dy)

        # 保存变形图像
        deformed_np = (deformed_tensor.squeeze().permute(1, 2, 0).detach().numpy() * 255).astype(np.uint8)
        output_img_path = os.path.join(output_image_dir, fname)
        cv2.imwrite(output_img_path, deformed_np)

        # 保存视差图
        # 保存视差图（只保存 .disp 文件）
        disp_np = disp.numpy()  # [H, W, 2]
        disp_txt_path = os.path.join(output_disp_dir, os.path.splitext(fname)[0] + '.disp')
        np.savetxt(disp_txt_path, disp_np.reshape(-1, 2), fmt="%.6f")

    print("全部处理完成！")

# ==== 示例调用 ====
input_folder = './RoadSence/ir'
output_img_folder = './RoadSence/irWarp'
output_disp_folder = './RoadSence/irDisp'

process_images_in_folder(input_folder, output_img_folder, output_disp_folder, dx=15, dy=8)
