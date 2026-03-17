import torch
import torch.nn.functional as F
from PIL import Image
import os
import utils
from skimage import img_as_ubyte
from collections import OrderedDict
from natsort import natsorted
from glob import glob
import cv2
import argparse
from model.SUNet import SUNet_model
import math
from tqdm import tqdm
import yaml
import gc
import numpy as np
import time
from datetime import datetime

# Read configuration file
with open('training.yaml', 'r') as config:
    opt = yaml.safe_load(config)

# Command line parameter configuration
parser = argparse.ArgumentParser(description='Demo Image Restoration')
parser.add_argument('--input_dir', default='G:\SUNet_Project\SUNet-main\datasets\Denoising_DIV2K\test\input', type=str)
parser.add_argument('--window_size', default=8, type=int)
parser.add_argument('--size', default=256, type=int)
parser.add_argument('--stride', default=128, type=int)
parser.add_argument('--result_dir', default='G:\SUNet_Project\SUNet-main\datasets\Denoising_DIV2K\test\target',
                    type=str)
parser.add_argument('--weights', default='G:\SUNet_Project\SUNet-main\pretrain-model\model_bestPSNR.pth', type=str)
parser.add_argument('--log_file', default='Image_Processing_Information_Summary.txt', type=str, help='Statistics document save path')

args = parser.parse_args()
def overlapped_square(timg, kernel=256, stride=128):
    patch_images = []
    b, c, h, w = timg.size()
    pad_h = (kernel - h % kernel) % kernel
    pad_w = (kernel - w % kernel) % kernel
    img = F.pad(timg, (pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2), mode='reflect')
    mask = F.pad(torch.ones_like(timg[:, :1, :, :]),
                 (pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2),
                 mode='constant', value=0)

    h_pad, w_pad = img.shape[2], img.shape[3]

    patches = img.unfold(2, kernel, stride).unfold(3, kernel, stride)
    nh, nw = patches.shape[2], patches.shape[3]
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, c, kernel, kernel)

    coords = []
    for i in range(nh):
        for j in range(nw):
            top = i * stride
            left = j * stride
            bottom = top + kernel
            right = left + kernel
            coords.append((top, left, bottom, right))

    for each in patches:
        patch_images.append(each.unsqueeze(0))

    return patch_images, mask, h_pad, w_pad, coords, h, w

def reconstruct_patches(patches, coords, h_pad, w_pad, kernel=256, device='cuda'):
    b = 1
    c = patches[0].shape[1]
    restored = torch.zeros((b, c, h_pad, w_pad), dtype=torch.float32, device=device)
    weight_sum = torch.zeros((b, c, h_pad, w_pad), dtype=torch.float32, device=device)

    sigma = kernel / 7.0
    gw_1d = cv2.getGaussianKernel(kernel, sigma)
    gw_2d = np.outer(gw_1d, gw_1d)
    gw_2d = gw_2d / gw_2d.max()

    gw = torch.from_numpy(gw_2d).float().to(device)
    gw = gw.unsqueeze(0).unsqueeze(0).repeat(1, c, 1, 1)

    for idx, patch in enumerate(patches):
        top, left, bottom, right = coords[idx]
        restored[:, :, top:bottom, left:right] += patch * gw
        weight_sum[:, :, top:bottom, left:right] += gw
    restored = restored / weight_sum.clamp(min=1e-6)
    return restored
def infer_patch_tta(model, patch):
    out = model(patch)
    out_h = model(torch.flip(patch, dims=[3]))
    out = out + torch.flip(out_h, dims=[3])
    out_v = model(torch.flip(patch, dims=[2]))
    out = out + torch.flip(out_v, dims=[2])
    return out / 3.0


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def load_checkpoint(model, weights, device):
    checkpoint = torch.load(weights, map_location=device)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    model.eval()
    model.half() if torch.cuda.is_available() else model.float()
    return model

def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
inp_dir = args.input_dir
out_dir = args.result_dir
os.makedirs(out_dir, exist_ok=True)

files = natsorted(glob(os.path.join(inp_dir, '*.jpg')) + glob(os.path.join(inp_dir, '*.png')))
if len(files) == 0:
    raise Exception(f"No images found in {inp_dir}")

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    gpu_usage = 1
    cpu_usage = 0
    torch.backends.cudnn.benchmark = True
    print(f'GPU is available, using {torch.cuda.get_device_name(0)} for accelerated processing')
else:
    device = torch.device('cpu')
    gpu_usage = 0
    cpu_usage = 1
    print('GPU is not available, automatically switching to CPU processing')

model = SUNet_model(opt).to(device)
model = load_checkpoint(model, args.weights, device)

log_data = []
total_start_time = time.time()
extra_data_usage = 0
psnr_list = []
ssim_list = []

print(f'Found {len(files)} images, start denoising...')
stride = args.stride
model_img = args.size

for file_ in tqdm(files):
    img_start_time = time.time()

    img = Image.open(file_).convert('RGB')
    img_original = np.array(img).astype(np.uint8)
    img_np = np.array(img).astype(np.float32) / 255.0
    input_ = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        patches, mask, h_pad, w_pad, coords, orig_h, orig_w = overlapped_square(
            input_, kernel=model_img, stride=stride
        )
        output_patches = []
        for p in patches:
            if torch.cuda.is_available():
                p = p.half()
            res = infer_patch_tta(model, p)
            output_patches.append(res.float())
        restored = reconstruct_patches(output_patches, coords, h_pad, w_pad, kernel=model_img, device=device)
        pad_h = (h_pad - orig_h) // 2
        pad_w = (w_pad - orig_w) // 2
        restored = restored[:, :, pad_h:pad_h+orig_h, pad_w:pad_w+orig_w]
        restored = torch.clamp(restored, 0, 1)

    restored = restored.permute(0, 2, 3, 1).squeeze().cpu().numpy()
    restored = img_as_ubyte(restored)
    filename = os.path.splitext(os.path.basename(file_))[0]
    save_path = os.path.join(out_dir, f'{filename}.png')
    save_img(save_path, restored)
    img_runtime = round(time.time() - img_start_time, 4)
    psnr = calculate_psnr(img_original, restored)
    ssim = calculate_ssim(img_original, restored)
    psnr_list.append(round(psnr, 2))
    ssim_list.append(round(ssim, 3))

    log_data.append({
        'filename': f'{filename}.png',
        'runtime': img_runtime,
        'cpu_usage': cpu_usage,
        'gpu_usage': gpu_usage,
        'extra_data': extra_data_usage
    })
    del input_, restored, output_patches, patches
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

total_processing_time = round(time.time() - total_start_time, 4)
success_count = len(log_data)
average_runtime = round(total_processing_time / success_count, 4) if success_count else 0.0
global_average_psnr = round(np.mean(psnr_list), 2) if psnr_list else 0.0
global_average_ssim = round(np.mean(ssim_list), 3) if ssim_list else 0.0

generation_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
with open(args.log_file, 'w', encoding='utf-8') as f:
    f.write('===== readme =====\n')
    f.write(f'Generation Time: {generation_time}\n')
    f.write(f'Default GPU Usage: {gpu_usage} | Default Extra Training Data Usage: {extra_data_usage}\n')

    for entry in log_data:
        f.write(f'=== {entry["filename"]} ===\n')
        f.write(f'runtime per image [s] : {entry["runtime"]}\n')
        f.write(f'CPU[{entry["cpu_usage"]}] / GPU[{entry["gpu_usage"]}] : 1\n')
        f.write(f'Extra Data [1] / No Extra Data [0] : {entry["extra_data"]}\n')

    f.write('===== Statistical Information =====\n')
    f.write(f'Successfully Processed Images: {success_count}\n')
    f.write(f'Total Processing Time: {total_processing_time} seconds\n')
    f.write(f'Average Single Image Processing Time: {average_runtime} seconds\n')
    f.write(f'Global Average PSNR: {global_average_psnr}\n')
    f.write(f'Global Average SSIM: {global_average_ssim}\n')

print(f'Denoising completed! Results saved to: {out_dir}')
print(f'Statistical log saved to: {args.log_file}')
print(
    f'Key Statistics: Total Time={total_processing_time}s, Avg Runtime={average_runtime}s, Avg PSNR={global_average_psnr}, Avg SSIM={global_average_ssim}')