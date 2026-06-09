"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import sys
import time
import numpy as np
import torch

from models.cycle_gan_model import CycleGANModel
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torchvision.transforms as transforms


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")  # 使用写入模式打开文件

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # 立即刷新到文件

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        if not self.log.closed:
            self.log.close()
# 新增函数：计算信息熵
def calculate_entropy(image_np):
    """计算图像的信息熵"""
    if len(image_np.shape) > 2:  # 彩色图像
        entropy = 0
        for ch in range(image_np.shape[2]):
            channel = image_np[:, :, ch]
            hist = np.histogram(channel, bins=256, range=(0, 255))[0]
            hist = hist / hist.sum()
            entropy -= np.sum(hist * np.log2(hist + 1e-7))
        entropy /= image_np.shape[2]
    else:  # 灰度图像
        hist = np.histogram(image_np, bins=256, range=(0, 255))[0]
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
    return entropy

# 新增函数：计算结构指数（平均梯度）
def calculate_structure_index(image_np):
    """计算图像的结构指数（平均梯度）"""
    if len(image_np.shape) > 2:  # 彩色图像
        structure = 0
        for ch in range(image_np.shape[2]):
            channel = image_np[:, :, ch].astype(np.float32)
            gx, gy = np.gradient(channel)
            structure += np.mean(np.sqrt(gx**2 + gy**2))
        structure /= image_np.shape[2]
    else:  # 灰度图像
        channel = image_np.astype(np.float32)
        gx, gy = np.gradient(channel)
        structure = np.mean(np.sqrt(gx**2 + gy**2))
    return structure
if __name__ == '__main__':

    try:
        opt = TrainOptions().parse()  # get training options
        # 创建日志目录
        log_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'training_logs.txt')
    # 创建Logger实例并重定向sys.stdout
        logger = Logger(log_file)
        sys.stdout = logger
        opt.continue_train = True;
        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        dataset_size = len(dataset)  # get the number of images in the dataset.
        print('The number of training images = %d' % dataset_size)
        model = create_model(opt)  # create a model given opt.model and other options
        model.setup(opt)  # regular setup: load and print networks; create schedulers
        visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
        total_iters = 0
        # 添加加密性能统计变量
        encrypt_times = []

        # the total number of training iterations
        for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()  # timer for data loading per iteration
            epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
            visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch
            model.update_learning_rate()  # update learning rates in the beginning of every epoch.
            if epoch % 5 == 0:
                # 自适应调整混沌因子
                current_uaci = model.calculate_uaci(model.fake_B_pre_encrypt, model.fake_B_encrypted)
                if current_uaci < 30.0:
                    model.chaos_factor = min(0.95, model.chaos_factor + 0.05)
                elif current_uaci > 36.0:  # 放宽上限，避免过早抑制
                    model.chaos_factor = max(0.7, model.chaos_factor - 0.02)
            for i, data in enumerate(dataset):  # inner loop within one epoch
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                total_iters += opt.batch_size
                epoch_iter += opt.batch_size
            # 记录加密开始时间
                encrypt_start_time = time.time()
                model.set_input(data)  # unpack data from dataset and apply preprocessing
                model.optimize_parameters()  # calculate loss functions, get gradients, update network weights
            # 记录加密时间
                encrypt_time = time.time() - encrypt_start_time
                encrypt_times.append(encrypt_time)
                if i % opt.display_freq == 0:
                    print(
                        f"Identity Loss A: {model.loss_identity_A.item():.4f}, "
                        f"Identity Loss B: {model.loss_identity_B.item():.4f}, "
                        f"SSIM Loss A: {model.loss_ssim_A.item():.4f}, "
                        f"SSIM Loss B: {model.loss_ssim_B.item():.4f}, ")
                if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                    save_result = total_iters % opt.update_html_freq == 0
                    model.compute_visuals()
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
                    visuals = model.get_current_visuals()
                    if 'real_A' in visuals and 'fake_B_encrypted' in visuals:
                        to_pil = transforms.ToPILImage()
                    # 原始A域图像
                        real_A_tensor = visuals['real_A'].detach().cpu()
                        real_A_img = to_pil(real_A_tensor[0])
                        real_A_np = np.array(real_A_img)

                    # 加密后的fake_B_encrypted
                        fake_B_encrypted_tensor = visuals['fake_B_encrypted'].detach().cpu()
                        fake_B_encrypted_img = to_pil(fake_B_encrypted_tensor[0])
                        fake_B_encrypted_np = np.array(fake_B_encrypted_img)
                    # 解密后的rec_A
                        rec_A_tensor = visuals['rec_A'].detach().cpu()
                        rec_A_img = to_pil(rec_A_tensor[0])
                        rec_A_np = np.array(rec_A_img)
                    # 计算SSIM (A域原图 vs 加密后的fake_B)
                        if real_A_np.shape == fake_B_encrypted_np.shape:
                            ssim_val = ssim(
                                real_A_np,
                                fake_B_encrypted_np,
                                channel_axis=-1 if len(real_A_np.shape) > 2 else None,
                                data_range=255
                            )
                            ssim_rval = ssim(
                                real_A_np,
                                rec_A_np,
                                channel_axis=-1 if len(real_A_np.shape) > 2 else None,
                                data_range=255
                            )

                        # 计算信息熵
                            entropy_real_A = calculate_entropy(real_A_np)
                            entropy_fake_B = calculate_entropy(fake_B_encrypted_np)

                        # 计算结构指数
                            si_real_A = calculate_structure_index(real_A_np)
                            si_fake_B = calculate_structure_index(fake_B_encrypted_np)

                        # 计算平均加密时间
                            avg_encrypt_time = np.mean(encrypt_times) if encrypt_times else 0
                            # 归一化图像
                            real_A_float = real_A_np / 255.0
                            fake_B_encrypted_float = fake_B_encrypted_np / 255.0
                            rec_A_float = rec_A_np / 255.0

                            # 计算 NPCR 和 UACI（统一尺度 [0,1]）
                            npcr_cross, uaci_cross = model.calculate_npcr_uaci_float(real_A_float, fake_B_encrypted_float)

                        # 计算NPCR和UACI
                            #npcr_real_A, uaci_real_A = model.calculate_npcr_uaci(real_A_np / 255.0, real_A_np / 255.0)
                            uaci_final = float(torch.abs(
                                torch.from_numpy(fake_B_encrypted_np / 255.0) -
                                torch.from_numpy(real_A_np / 255.0)
                            ).mean()) * 100
                            uaci_val = model.calculate_uaci(
                                visuals['real_A'].detach().cpu(),
                                visuals['fake_B_encrypted'].detach().cpu()
                            )
                            npcr_val = model.calculate_npcr(
                                visuals['real_A'].detach().cpu(),
                                visuals['fake_B_encrypted'].detach().cpu()
                            )
                            rec_psnr = model.calculate_rec_psnr()
                        # 打印结果
                            print("\n" + "=" * 60)
                            print("混沌加密性能分析:")
                            print(f"平均加密时间: {avg_encrypt_time:.6f} 秒/图像")
                            print(f"SSIM (A域原图 vs 加密后的fake_B): {ssim_val:.4f}")
                            print(f"SSIM (A域原图 vs 解密后的fake_B): {ssim_rval:.4f}")
                            print(f"信息熵 - A域原图: {entropy_real_A:.4f}")
                            print(f"信息熵 - 加密后的fake_B: {entropy_fake_B:.4f}")
                            print(f"结构指数 - A域原图: {si_real_A:.4f}")
                            print(f"结构指数 - 加密后的fake_B: {si_fake_B:.4f}")
                            print("NPCR/UACI分析:")
                            print(f"A域原图vs加密后fake_B - NPCR: {npcr_cross:.4f}%, UACI: {uaci_final:.2f}%")
                            print("恢复图像质量分析:")
                            print(f"PSNR (原图 vs 恢复图rec_A): {rec_psnr:.2f} dB")
                            print("=" * 60 + "\n")

                        # 重置加密时间记录
                            encrypt_times = []
                        else:
                            print(f"[Warning] real_A 和 fake_B_encrypted 尺寸不匹配: {real_A_np.shape} vs {fake_B_encrypted_np.shape}")
                if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    if opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

                if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)

                iter_data_time = time.time()
            if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks('latest')
                model.save_networks(epoch)

            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
    except Exception as e:
        # 捕获异常并打印
        print(f"训练过程中发生错误: {str(e)}")
        import traceback

        traceback.print_exc()
    finally:
        # 确保日志文件正确关闭
        if 'logger' in locals():
            logger.close()
        sys.stdout = sys.__stdout__  # 恢复原始标准输出
