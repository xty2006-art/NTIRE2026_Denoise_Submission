SUNet Optimized Inference Pipeline

NTIRE 2026 Image Denoising Challenge (noise level = 50)

This project is an optimized inference implementation for image denoising task in NTIRE 2026. It is built upon the SUNet model architecture, with comprehensive engineering improvements on inference strategy, memory efficiency, image reconstruction, reproducibility, and evaluation metrics.
Method Overview
Our solution is based on the SUNet (Swin Transformer UNet) denoising backbone proposed by Fan et al. We do not modify the original model structure, but redesign the entire inference pipeline to achieve better denoising quality, faster speed, lower GPU memory consumption, and artifact-free reconstruction for high-resolution test images.
Major Optimizations (Our Contributions)
1.	Overlapped patch inference with Gaussian-weighted fusion
1.	Replace simple fold average with Gaussian-weighted patch reconstruction to completely eliminate stitching artifacts.
2.	Use reflection padding instead of zero-padding to reduce edge artifacts.
2.	Test-Time Augmentation (TTA)
1.	Add horizontal/vertical flip augmentation during inference.
2.	Effectively improve PSNR/SSIM by about 0.2–0.5 dB.
3.	Mixed precision inference (FP16)
1.	Reduce GPU memory usage by nearly 50%.
2.	Speed up inference without losing denoising quality.
4.	Explicit memory management
1.	Manual tensor deletion, GPU cache clearing, and garbage collection.
2.	Stable for batch processing of large datasets without memory leak.
5.	Automatic evaluation & logging
1.	Built-in PSNR/SSIM calculation for each image.
2.	Automatically generate a complete log including runtime, average metrics, and hardware usage for reproducibility.
6.	Robust engineering implementation
1.	Support arbitrary input resolution.
2.	Automatic GPU/CPU detection.
3.	Command-line configurable paths and parameters.
4.	Clear progress bar and real-time feedback.
Environment
ca-certificates	2026.2.25	
certifi	2026.2.25	
charset-normalizer	3.4.4	
colorama	0.4.6	
einops	0.8.1	
filelock	3.16.1	
fsspec	2025.3.0	
huggingface-hub	0.36.2	
idna	3.11	
imageio	2.35.1	
jinja2	3.1.6	
joblib	1.4.2	
lazy-loader	0.4	
libffi	3.5.2	
liblzma	5.8.2	
liblzma-devel	5.8.2	
libsqlite	3.51.2	
libzlib	1.3.1	
markupsafe	2.1.5	
mpmath	1.3.0	
natsort	8.4.0	
networkx	3.1	
numpy	1.24.4	
nw	0.0.5	
opencv-python	4.13.0.92	
openssl	3.6.1	
packaging	26.0	
pillow	10.4.0	
pip	25.0.1	
protobuf	5.29.6	
python	3.8.20	
pytorch-msssim	1.0.0	
pywavelets	1.4.1	
pyyaml	6.0.3	
requests	2.32.4	
safetensors	0.5.3	
scikit-image	0.21.0	
scipy	1.10.1	
setuptools	75.3.0	
sympy	1.13.3	
tensorboardx	2.6.2.2	
thop	0.1.1-2209072238	
tifffile	2023.7.10	
timm	0.9.16	
tk	8.6.13	
torch	2.1.0+cu121	
torchaudio	2.1.0+cu121	
torchvision	0.16.0+cu121	
tqdm	4.67.3	
typing-extensions	4.13.2	
ucrt	10.0.26100.0	
urllib3	2.2.3	
vc	14.3	
vc14-runtime	14.44.35208	
vcomp14	14.44.35208	
warmup-scheduler	0.3	
wheel	0.45.1	
xz	5.8.2	
xz-tools	5.8.2	
How to Run
python demo_any_resolution \
    --input_dir SUNet_Project/SUNet-main/datasets/Denoising_DIV2K/test/input \
    --result_dir SUNet_Project/SUNet-main/datasets/Denoising_DIV2K/test/target \
    --weights pretrain-model/model_bestPSNR.pth
Citation
Original SUNet Model
@inproceedings{fan2022sunet,
  title={SUNet: swin transformer UNet for image denoising},
  author={Fan, Chi-Mao and Liu, Tsung-Jung and Liu, Kuan-Hsien},
  booktitle={2022 IEEE International Symposium on Circuits and Systems (ISCAS)},
  pages={2333--2337},
  year={2022},
  organization={IEEE}
}
Official repository: https://github.com/FanChiMao/SUNet
Acknowledgement
We thank the authors of SUNet for their excellent work and open-source contribution. Our optimized inference pipeline fully respects the original model and is designed for better performance and reproducibility in the NTIRE 2026 challenge.
Contact
For reproducibility issues or questions about the optimized code, please contact the team leader via email:3980209359@qq.com

