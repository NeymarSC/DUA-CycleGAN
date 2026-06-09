import torch
import itertools
import matplotlib.pyplot as plt
from skimage.exposure import equalize_hist

from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from pytorch_msssim import ssim
import torch.nn.functional as F
import numpy as np
import cv2
import lpips
from torchvision.utils import make_grid
# ==============================================================================
#  CONFIDENTIAL IMPLEMENTATION DETAILS
# ==============================================================================
#  The following section implements the core intellectual property of the
#  DUA-CycleGAN architecture, specifically the Adaptive Lesion Feature
#  Enhancement mechanism and the Dynamic Weighted UACI optimization.
#
#  Due to laboratory confidentiality agreements and pending patent review,
#  the specific parameter configurations, weight calculation formulas, and
#  architectural connections are omitted here.
#
#  The placeholder below maintains the structural integrity of the model
#  for reproducibility of the experimental environment, but the functional
#  logic has been masked.
#
#  For collaboration or academic verification requests, please contact the
#  corresponding author.
# ==============================================================================
class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_rec', type=float, default=20.0, help='weight for reconstruction loss')
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'fake_B_encrypted', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']
        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            # 新增：初始化混沌序列缓存
            self.chaos_cache = {}  # 缓存混沌序列，加速训练
            self.current_uaci=0.0
            self.current_npcr = 0.0
            self.current_psnr = 0.0

    def calculate_lpips(self, img1, img2):
        """
        计算两张图像之间的LPIPS值
        Args:
            img1, img2: 输入图像，范围[-1, 1]
        Returns:
            LPIPS值
        """
        # 确保输入范围正确
        img1 = torch.clamp(img1, -1, 1)
        img2 = torch.clamp(img2, -1, 1)

        # 计算LPIPS
        with torch.no_grad():
            lpips_value = self.lpips_model(img1, img2).mean().item()

        return lpips_value


    def chaotic_encrypt(self, image_tensor):
        # [CONFIDENTIAL]: Core attention mechanism implementation hidden.
        # Placeholder for variance-weighted pooling layers.
        return self.chaotic_model(image_tensor)


    def adjust_encryption_strength(self):

        # [CONFIDENTIAL]: Core attention mechanism implementation hidden.
        # Placeholder for variance-weighted pooling layers.
        return self.chaotic_model(image_tensor)


    def calculate_npcr(self, img1, img2):
        """高精度NPCR计算（支持GPU）"""
        # 转换为整数避免浮点误差
        img1_int = (img1 * 255).byte()
        img2_int = (img2 * 255).byte()

        # 计算差异像素比例
        diff_pixels = torch.sum(img1_int != img2_int).float()
        total_pixels = img1_int.numel()
        npcr = (diff_pixels / total_pixels) * 100

        return npcr.item()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # [CONFIDENTIAL]: Core attention mechanism implementation hidden.
        # Placeholder for variance-weighted pooling layers.


        # 实时UACI监控
        if self.isTrain:
            self.adjust_encryption_strength()
            self.current_uaci =  self.calculate_uaci(self.real_A,self.fake_B_encrypted)
            self.current_npcr = self.calculate_npcr(self.real_A, self.fake_B_encrypted)
            self.current_psnr = self.calculate_current_psnr(self.real_A, self.rec_A)
            print(f"[NPCR] {self.current_npcr:.2f}% | [UACI] {self.current_uaci:.2f}% | [PSNR] {self.current_psnr:.2f} dB | [LPIPS A→B] {self.current_lpips_A2B:.4f} | "
              f"[LPIPS B→A] {self.current_lpips_B2A:.4f}")
    def calculate_current_psnr(self, img1, img2):
        img1 = (img1 + 1) * 127.5
        img2 = (img2 + 1) * 127.5
        img1 = torch.clamp(img1, 0, 255)
        img2 = torch.clamp(img2, 0, 255)
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        psnr = 10 * torch.log10((255 ** 2) / mse)
        return psnr.item()

    def calculate_uaci(self, img1, img2):
        """统一的UACI计算方法，与uaci_final一致"""
        # 确保输入在[0,1]范围内
        img1 = torch.clamp((img1 + 1) / 2, 0, 1)  # [-1,1] -> [0,1]
        img2 = torch.clamp((img2 + 1) / 2, 0, 1)

        # 转换为numpy数组进行计算（与uaci_final一致的方法）
        img1_np = img1.detach().cpu().numpy()
        img2_np = img2.detach().cpu().numpy()

        # 使用与uaci_final相同的计算方法
        uaci = float(np.abs(img1_np - img2_np).mean()) * 100
        return uaci

    def calculate_npcr_fast(self, img1, img2):
        """快速计算 NPCR（基于 PyTorch，支持 GPU）"""
        img1 = torch.clamp(img1, 0, 1)
        img2 = torch.clamp(img2, 0, 1)

        # 转换为 [0, 255] 的整数
        img1_int = (img1 * 255).round().long()
        img2_int = (img2 * 255).round().long()

        # 计算不同像素的数量
        diff = (img1_int != img2_int).float()
        npcr = diff.mean() * 100

        return npcr.item()
    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        # [CONFIDENTIAL]: Core attention mechanism implementation hidden.
        # Placeholder for variance-weighted pooling layers.
        pass

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    def identity_loss(self, real_image, same_class_image):
        """Identity loss for CycleGAN"""
        return torch.mean(torch.abs(real_image - same_class_image))

    def calculate_npcr_uaci(self, img1, img2):
        """计算两张图像之间的NPCR和UACI"""
        # 确保输入是[0,255]范围的整数
        img1_int = (img1 * 127.5 + 127.5).astype(np.uint8)
        img2_int = (img2 * 127.5 + 127.5).astype(np.uint8)

        # 确保图像形状相同
        if img1_int.shape != img2_int.shape:
            raise ValueError("Images must have the same dimensions")

        # 计算NPCR
        diff_pixels = np.sum(img1_int != img2_int)
        total_pixels = np.prod(img1_int.shape)
        npcr = (diff_pixels / total_pixels) * 100

        # 计算UACI
        abs_diff = np.abs(img1_int.astype(np.float32) - img2_int.astype(np.float32))
        uaci = np.mean(abs_diff / 255.0) * 100

        return npcr, uaci

    def calculate_psnr(self, img1, img2, max_val=255.0):
        """
        计算两张图像之间的PSNR(峰值信噪比)

        Args:
            img1, img2: 需要比较的两张图像(numpy数组，值范围0-255)
            max_val: 图像的最大像素值(默认为255)

        Returns:
            PSNR值(单位: dB)
        """
        # 将图像转换为浮点类型
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        # 计算均方误差(MSE)
        mse = np.mean((img1 - img2) ** 2)

        # 避免除以零
        if mse == 0:
            return float('inf')  # 图像完全相同，PSNR无穷大

        # 计算PSNR
        psnr = 10 * np.log10((max_val ** 2) / mse)
        return psnr

    def calculate_rec_psnr(self):
        real_A = (self.real_A + 1) * 127.5  # [-1,1] -> [0,255]
        rec_A = (self.rec_A + 1) * 127.5

        # 添加边界保护
        real_A = torch.clamp(real_A, 0, 255)
        rec_A = torch.clamp(rec_A, 0, 255)

        mse = torch.mean((real_A - rec_A) ** 2)
        return 10 * torch.log10(255 ** 2 / (mse + 1e-10))

    def ssim_loss(self, img1, img2):
        """Compute SSIM loss between two images."""
        # Ensure the images are of the same shape, for example, (batch_size, channels, height, width)
        assert img1.size() == img2.size(), f"Shape mismatch: {img1.size()} vs {img2.size()}"

        # Flatten to make them 1D vectors, and compute cosine similarity
        return 1 - F.cosine_similarity(img1.view(img1.size(0), -1), img2.view(img2.size(0), -1), dim=1).mean()

    def calculate_npcr_uaci_float(self, img1, img2):
        """计算 [0,1] 浮点图像之间的 NPCR 和 UACI"""
        img1 = np.clip(img1, 0, 1)
        img2 = np.clip(img2, 0, 1)

        # 将图像量化为 256 级（模拟 uint8）
        img1_int = (img1 * 255).round().astype(np.uint8)
        img2_int = (img2 * 255).round().astype(np.uint8)

        # NPCR
        diff_pixels = np.sum(img1_int != img2_int)
        total_pixels = np.prod(img1_int.shape)
        npcr = (diff_pixels / total_pixels) * 100

        # UACI
        abs_diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
        uaci = np.mean(abs_diff) * 100

        return npcr, uaci

    def calculate_psnr_float(self, img1, img2):
        """计算 [0,1] 浮点图像之间的 PSNR"""
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        psnr = 10 * np.log10(1.0 / mse)  # max_val = 1.0
        return psnr