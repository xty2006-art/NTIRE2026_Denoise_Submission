clc
clear
close all
%% Path setting
% Check the image format in the folder is correct (png, jpg...)!
% Denoise image path (end with '\'):
pathD = 'G:\SUNet_Project\SUNet-main\datasets\Denoising_DIV2K\test\target';
img_denoise_list = dir(strcat(pathD,'*.png'));

% Noise image path (end with '\'):  
pathN  = 'G:\SUNet_Project\SUNet-main\datasets\Denoising_DIV2K\test\input';
img_noise_list = dir(strcat(pathN,'*.png'));

% Clean image path (end with '\'): 
pathGT = 'G:\SUNet_Project\SUNet-main\datasets\Denoising_DIV2K\test\GT';
img_GT_list = dir(strcat(pathGT,'*.png')); 
%% Evaluation
noise_psnr = 0;
denoise_psnr = 0;
noise_ssim = 0;
denoise_ssim = 0;

img_num = length(img_GT_list);%����v�H�`�ƶq


for j = 1:img_num %�v�@Ū���v�H
	GT_name = img_GT_list(j).name;% �v�H�W
    GT = imread(strcat(pathGT,GT_name));
    GT = im2single(GT);
    GT_g = rgb2gray(GT);
        
    Noise_name = img_noise_list(j).name;% �v�H�W
    Noise = imread(strcat(pathN,Noise_name));
    Noise = im2single(Noise);
    Noise_g = rgb2gray(Noise);
        
    Denoise_name = img_denoise_list(j).name;% �v�H�W
    Denoise = imread(strcat(pathD,Denoise_name));
    Denoise = im2single(Denoise);
    Denoise_g = rgb2gray(Denoise);
    
    [d_psnr, d_snr] = psnr(Denoise, GT); 
    [n_psnr, n_snr] = psnr(Noise, GT); 
    d_ssim = ssim(Denoise_g, GT_g);
    n_ssim = ssim(Noise_g, GT_g);
    
    fprintf('GT: %s\n' ,GT_name);
    fprintf('Noise: %s\n' ,Noise_name);
    fprintf('Denoise: %s\n' ,Denoise_name);
    fprintf('\n')
    fprintf('  Noise PSNR  = %0.4f dB ---------- %2d/%2d\n', n_psnr, j, img_num);
    fprintf('  Noise SSIM  = %0.4f     ---------- %2d/%2d\n', n_ssim, j, img_num);
    fprintf('Denoise PSNR* = %0.4f dB ---------- %2d/%2d\n', d_psnr, j, img_num);
    fprintf('Denoise SSIM* = %0.4f     ---------- %2d/%2d\n', d_ssim, j, img_num);
    fprintf('----------------------------------------------\n')
    noise_psnr = noise_psnr + n_psnr;
    denoise_psnr = denoise_psnr + d_psnr;
    noise_ssim = noise_ssim + n_ssim;
    denoise_ssim = denoise_ssim + d_ssim;
end       
Total_noise_PSNR = noise_psnr/img_num;
Total_denoise_PSNR = denoise_psnr/img_num;
Total_noise_SSIM = noise_ssim/img_num;
Total_denoise_SSIM = denoise_ssim/img_num;
fprintf('                     Finsh!                   \n');
fprintf('  Average noise PSNR  = %0.4f dB\n', Total_noise_PSNR);
fprintf('  Average noise SSIM  = %0.4f\n', Total_noise_SSIM);
fprintf('Average denoise PSNR* = %0.4f dB\n', Total_denoise_PSNR);
fprintf('Average denoise SSIM* = %0.4f\n', Total_denoise_SSIM);
fprintf('----------------------------------------------\n')
