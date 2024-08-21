import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
from CycleGAN_networks import * 

import sys
import matplotlib.pyplot as plt 

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from torchsummary import summary
# from models import *
from processing_utilities import *
from MRILoader import MRImgLoader, ImageTransform, RandomCrop
import gc
# import wandb

def config():
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    print(torch.cuda.is_available())
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")
    parser.add_argument("--decay_epoch", type=int, default=1000, help="epoch from which to start lr decay") # 1000
    parser.add_argument("--model_name", type=str, default="test", help="name of the dataset")
    parser.add_argument("--data_name", type=str, default="data_example", help="name of the dataset")
    parser.add_argument("--n_depth", type=int, default=3, help="number of discirminator depth")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--img_height", type=int, default=512, help="size of image height")
    parser.add_argument("--img_width", type=int, default=512, help="size of image width")
    parser.add_argument("--half_height", type=int, default=256, help="size of center crops")
    parser.add_argument("--half_width", type=int, default=256, help="size of center crops")
    parser.add_argument("--local_height", type=int, default=128, help="size of random sampling")
    parser.add_argument("--local_width", type=int, default=128, help="size of random sampling")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model checkpoints")
    parser.add_argument("--lambda_adv", type=float, default=0.1, help="cycle loss weight")
    parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
    parser.add_argument("--lambda_id", type=float, default=2.0, help="identity loss weight")
    opt = parser.parse_args()

    return opt

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CycleGAN_Trainer():
    def __init__(self):
        self.opt = config() 
        self.makedirs()
        seed_everything(42)

        # Losses
        self.criterion_GAN = torch.nn.MSELoss().cuda() # qq1
        self.criterion_cyc = torch.nn.L1Loss().cuda()
        self.criterion_idt = torch.nn.L1Loss().cuda()

        # shapes
        self.input_shape = (self.opt.channels, self.opt.img_height, self.opt.img_width)
        self.halfcrop_shape = (self.opt.channels, self.opt.half_height, self.opt.half_width)
        self.localcrop_shape = (self.opt.channels, self.opt.local_height, self.opt.local_width)

        self.set_generator(self.opt)
        self.set_discriminator(self.opt)
        self.network_init(self.opt)

        # buffer & cropper
        self.fake_epi_buffer_512 = ReplayBuffer()
        self.fake_epi_buffer_256 = ReplayBuffer()
        self.fake_epi_buffer_192 = ReplayBuffer()
        self.fake_rou_buffer_512 = ReplayBuffer()
        self.fake_rou_buffer_256 = ReplayBuffer()
        self.fake_rou_buffer_192 = ReplayBuffer()

        self.halfcropper = transforms.RandomCrop(size=(self.opt.half_height, self.opt.half_width))
        self.localcropper = transforms.RandomCrop(size=(self.opt.local_height, self.opt.local_width))
        # self.halfcropper = RandomCrop(in_size=opt.img_height, tar_size=opt.center_height) # 512 -> 256
        # self.localcropper = RandomCrop(in_size=opt.half_height, tar_size=opt.half_width, margin=0) # 256 -> 192

        # dataset loader
        epi_dataset = MRImgLoader(path =f"./data/{self.opt.data_name}/train_patch/", sequence="low_dose", transform=ImageTransform(), batch_size=self.opt.batch_size)
        rou_dataset = MRImgLoader(path =f"./data/{self.opt.data_name}/train_patch/", sequence="mid_dose", transform=ImageTransform(), batch_size=self.opt.batch_size)
        
        self.n_sample = len(rou_dataset)
        self.epi_loader = DataLoader(dataset=epi_dataset, batch_size=self.opt.batch_size, shuffle=True, num_workers=4)
        self.rou_loader = DataLoader(dataset=rou_dataset, batch_size=self.opt.batch_size, shuffle=True, num_workers=4)

    def makedirs(self):
        os.makedirs("train_save/%s" % self.opt.model_name, exist_ok=True)

    def set_generator(self, opt):
        self.G_epi2rou = BlurPooling_generator(self.input_shape).cuda() # qq2
        self.G_rou2epi = BlurPooling_generator(self.input_shape).cuda()

        self.optimizer_G_epi2rou = torch.optim.Adam(self.G_epi2rou.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        self.optimizer_G_rou2epi = torch.optim.Adam(self.G_rou2epi.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        
        self.lr_scheduler_G_epi2rou = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_G_epi2rou, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
        )
        self.lr_scheduler_G_rou2epi = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_G_rou2epi, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
        )

        return 0

    def set_discriminator(self, opt):
        if opt.n_depth == 1:
            self.Dis512 = True # qq3 : 나중에 192 -> 128로 변경 후 성능 확인 
            self.Dis256 = False
            self.Dis192  = False
            
        elif opt.n_depth == 2:
            self.Dis512 = True
            self.Dis256 = True
            self.Dis192  = False
            
        elif opt.n_depth == 3:
            self.Dis512 = True
            self.Dis256 = True
            self.Dis192 = True

        if self.Dis512:
            self.D_epi_512 = PatchGANDiscriminator_BlurPooling(self.input_shape).cuda() # qq4
            self.D_rou_512 = PatchGANDiscriminator_BlurPooling(self.input_shape).cuda()

            self.ones_512 = Variable(torch.cuda.FloatTensor(np.ones((opt.batch_size, *self.D_epi_512.output_shape))), requires_grad=False)
            self.zeros_512 = Variable(torch.cuda.FloatTensor(np.zeros((opt.batch_size, *self.D_epi_512.output_shape))), requires_grad=False)

            self.optimizer_D_epi_512 = torch.optim.Adam(self.D_epi_512.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
            self.optimizer_D_rou_512 = torch.optim.Adam(self.D_rou_512.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

            self.lr_scheduler_D_EPI_512 = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer_D_epi_512, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
            )
            self.lr_scheduler_D_ROU_512 = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer_D_rou_512, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
            )

        if self.Dis256:
            self.D_epi_256 = PatchGANDiscriminator_BlurPooling(self.halfcrop_shape).cuda() 
            self.D_rou_256 = PatchGANDiscriminator_BlurPooling(self.halfcrop_shape).cuda()

            self.ones_256 = Variable(torch.cuda.FloatTensor(np.ones((opt.batch_size, *self.D_epi_256.output_shape))), requires_grad=False)
            self.zeros_256 = Variable(torch.cuda.FloatTensor(np.zeros((opt.batch_size, *self.D_epi_256.output_shape))), requires_grad=False)

            self.optimizer_D_epi_256 = torch.optim.Adam(self.D_epi_256.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
            self.optimizer_D_rou_256 = torch.optim.Adam(self.D_rou_256.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

            self.lr_scheduler_D_EPI_256 = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer_D_epi_256, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
            )
            self.lr_scheduler_D_ROU_256 = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer_D_rou_256, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
            )

        if self.Dis192:
            self.D_epi_192 = PatchGANDiscriminator_BlurPooling(self.localcrop_shape).cuda()
            self.D_rou_192 = PatchGANDiscriminator_BlurPooling(self.localcrop_shape).cuda()

            self.ones_192 = Variable(torch.cuda.FloatTensor(np.ones((opt.batch_size, *self.D_epi_192.output_shape))), requires_grad=False)
            self.zeros_192 = Variable(torch.cuda.FloatTensor(np.zeros((opt.batch_size, *self.D_epi_192.output_shape))), requires_grad=False)

            self.optimizer_D_epi_192 = torch.optim.Adam(self.D_epi_192.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
            self.optimizer_D_rou_192 = torch.optim.Adam(self.D_rou_192.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

            self.lr_scheduler_D_EPI_192 = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer_D_epi_192, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
            )
            self.lr_scheduler_D_ROU_192 = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer_D_rou_192, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
            )

        return 0

    def network_init(self, opt):
        if opt.epoch != 0:
            # Load pretrained models
            print(f'\n=============={opt.epoch} 번째 학습부터 시작됩니다.=============\n')
            self.G_epi2rou.load_state_dict(torch.load("./train_save/%s/G_epi2rou_%d.pth" % (opt.model_name, opt.epoch)))
            self.G_rou2epi.load_state_dict(torch.load("./train_save/%s/G_rou2epi_%d.pth" % (opt.model_name, opt.epoch)))

            self.D_epi_512.load_state_dict(torch.load("./train_save/%s/D_epi_512_%d.pth" % (opt.model_name, opt.epoch)))
            self.D_rou_512.load_state_dict(torch.load("./train_save/%s/D_rou_512_%d.pth" % (opt.model_name, opt.epoch)))

            self.D_epi_256.load_state_dict(torch.load("./train_save/%s/D_epi_256_%d.pth" % (opt.model_name, opt.epoch)))
            self.D_rou_256.load_state_dict(torch.load("./train_save/%s/D_rou_256_%d.pth" % (opt.model_name, opt.epoch)))

            self.D_epi_192.load_state_dict(torch.load("./train_save/%s/D_epi_192_%d.pth" % (opt.model_name, opt.epoch)))
            self.D_rou_192.load_state_dict(torch.load("./train_save/%s/D_rou_192_%d.pth" % (opt.model_name, opt.epoch)))

        else:
            print(f'\n==============첫 번째 학습부터 시작됩니다.=============\n')

        return 0

    def train(self):
        opt = self.opt

        for epoch in range(opt.epoch+1, opt.n_epochs+1): # 1부터 ~ 
            for i, (real_epi, real_rou) in enumerate(zip(self.epi_loader, self.rou_loader)):
                torch.cuda.empty_cache()

                real_epi = real_epi.cuda().float()
                real_rou = real_rou.cuda().float()

                self.G_epi2rou.train()
                self.G_rou2epi.train()

                # fake image generation
                fake_epi = self.G_rou2epi(real_rou)
                fake_rou = self.G_epi2rou(real_epi)

                recon_epi = self.G_rou2epi(fake_rou) # epi -> fake -> recon
                recon_rou = self.G_epi2rou(fake_epi) # rou -> fake -> recon

                ### buffer 
                fake_epi_ = self.fake_epi_buffer_512.push_and_pop(fake_epi)
                fake_rou_ = self.fake_rou_buffer_512.push_and_pop(fake_rou)

                # 256 crop
                real_epi_half = self.halfcropper(real_epi)
                real_rou_half = self.halfcropper(real_rou)
                fake_epi_half = self.halfcropper(fake_epi)
                fake_rou_half = self.halfcropper(fake_rou)

                fake_epi_half_ = self.fake_epi_buffer_256.push_and_pop(fake_epi_half)
                fake_rou_half_ = self.fake_rou_buffer_256.push_and_pop(fake_rou_half)

                # 192 crop
                real_epi_local = self.localcropper(real_epi_half)
                real_rou_local = self.localcropper(real_rou_half)
                fake_epi_local = self.localcropper(fake_epi_half)
                fake_rou_local = self.localcropper(fake_rou_half)

                fake_epi_local_ = self.fake_epi_buffer_192.push_and_pop(fake_epi_local)
                fake_rou_local_ = self.fake_rou_buffer_192.push_and_pop(fake_rou_local)

                # -----------------------
                #  Train Discriminator
                # -----------------------
                if self.Dis512:
                    # -------------------------
                    #  Train Discriminator EPI
                    # -------------------------

                    self.D_epi_512.train()
                    self.D_rou_512.train()

                    self.optimizer_D_epi_512.zero_grad()
                    loss_real = self.criterion_GAN(self.D_epi_512(real_epi), self.ones_512)
                    loss_fake = self.criterion_GAN(self.D_epi_512(fake_epi_.detach()), self.zeros_512)

                    loss_D_epi = (loss_real + loss_fake) * 0.5

                    loss_D_epi.backward()
                    self.optimizer_D_epi_512.step()

                    # -------------------------
                    #  Train Discriminator ROU
                    # -------------------------

                    self.optimizer_D_rou_512.zero_grad()
                    loss_real = self.criterion_GAN(self.D_rou_512(real_rou), self.ones_512)
                    loss_fake = self.criterion_GAN(self.D_rou_512(fake_rou_.detach()), self.zeros_512)

                    loss_D_rou = (loss_real + loss_fake) * 0.5

                    loss_D_rou.backward()
                    self.optimizer_D_rou_512.step()

                    loss_D_512 = (loss_D_epi + loss_D_rou)

                if self.Dis256:
                    # -------------------------
                    #  Train Discriminator EPI
                    # -------------------------

                    self.D_epi_256.train()
                    self.D_rou_256.train()

                    self.optimizer_D_epi_256.zero_grad()
                    loss_real = self.criterion_GAN(self.D_epi_256(real_epi_half), self.ones_256)
                    loss_fake = self.criterion_GAN(self.D_epi_256(fake_epi_half_.detach()), self.zeros_256)

                    loss_D_epi = (loss_real + loss_fake) * 0.5

                    loss_D_epi.backward()
                    self.optimizer_D_epi_256.step()

                    # -------------------------
                    #  Train Discriminator ROU
                    # -------------------------

                    self.optimizer_D_rou_256.zero_grad()
                    loss_real = self.criterion_GAN(self.D_rou_256(real_rou_half), self.ones_256)
                    loss_fake = self.criterion_GAN(self.D_rou_256(fake_rou_half_.detach()), self.zeros_256)

                    loss_D_rou = (loss_real + loss_fake) * 0.5

                    loss_D_rou.backward()
                    self.optimizer_D_rou_256.step()

                    loss_D_256 = (loss_D_epi + loss_D_rou)

                if self.Dis192:
                    # -------------------------
                    #  Train Discriminator EPI
                    # -------------------------

                    self.D_epi_192.train()
                    self.D_rou_192.train()

                    self.optimizer_D_epi_192.zero_grad()
                    loss_real = self.criterion_GAN(self.D_epi_192(real_epi_local), self.ones_192)
                    loss_fake = self.criterion_GAN(self.D_epi_192(fake_epi_local_.detach()), self.zeros_192)

                    loss_D_epi = (loss_real + loss_fake) * 0.5

                    loss_D_epi.backward()
                    self.optimizer_D_epi_192.step()

                    # -------------------------
                    #  Train Discriminator ROU
                    # -------------------------

                    self.optimizer_D_rou_192.zero_grad()
                    loss_real = self.criterion_GAN(self.D_rou_192(real_rou_local), self.ones_192)
                    loss_fake = self.criterion_GAN(self.D_rou_192(fake_rou_local_.detach()), self.zeros_192)

                    loss_D_rou = (loss_real + loss_fake) * 0.5

                    loss_D_rou.backward()
                    self.optimizer_D_rou_192.step()

                    loss_D_192 = (loss_D_epi + loss_D_rou)

                # ------------------
                #  Train Generators
                # ------------------
                del fake_epi 
                del fake_rou
                del recon_epi
                del recon_rou 
                gc.collect()

                self.optimizer_G_epi2rou.zero_grad()
                self.optimizer_G_rou2epi.zero_grad()

                # addddddddddddddddddddddddddd
                # fake image generation
                fake_epi = self.G_rou2epi(real_rou)
                fake_rou = self.G_epi2rou(real_epi)

                recon_epi = self.G_rou2epi(fake_rou) # epi -> fake -> recon
                recon_rou = self.G_epi2rou(fake_epi) # rou -> fake -> recon

                # 256 crop
                fake_epi_half = self.halfcropper(fake_epi)
                fake_rou_half = self.halfcropper(fake_rou)

                # 192 crop
                fake_epi_local = self.localcropper(fake_epi_half)
                fake_rou_local = self.localcropper(fake_rou_half)

                # Identity loss
                loss_idt_epi = self.criterion_idt(self.G_rou2epi(real_epi), real_epi)
                loss_idt_rou = self.criterion_idt(self.G_epi2rou(real_rou), real_rou)

                loss_idt = (loss_idt_epi + loss_idt_rou).item()

                # Cycle consistency loss 
                loss_cyc_epi = self.criterion_cyc(recon_epi, real_epi)
                loss_cyc_rou = self.criterion_cyc(recon_rou, real_rou)

                loss_cyc = (loss_cyc_epi + loss_cyc_rou).item()

                # Adversarial loss
                if opt.n_depth == 3:
                    loss_GAN_AB_512 = self.criterion_GAN(self.D_rou_512(fake_rou), self.ones_512)
                    loss_GAN_AB_256 = self.criterion_GAN(self.D_rou_256(fake_rou_half), self.ones_256)
                    loss_GAN_AB_192 = self.criterion_GAN(self.D_rou_192(fake_rou_local), self.ones_192)
                    loss_GAN_AB = (loss_GAN_AB_512 + loss_GAN_AB_256 + loss_GAN_AB_192) / 3.0

                    # total G_epi2rou loss 
                    loss_G_epi2rou = (opt.lambda_adv * loss_GAN_AB) + (opt.lambda_cyc * loss_cyc) + (opt.lambda_id * loss_idt)
                    loss_G_epi2rou.backward()
                    self.optimizer_G_epi2rou.step() 

                    loss_GAN_BA_512 = self.criterion_GAN(self.D_epi_512(fake_epi), self.ones_512)
                    loss_GAN_BA_256 = self.criterion_GAN(self.D_epi_256(fake_epi_half), self.ones_256)
                    loss_GAN_BA_192 = self.criterion_GAN(self.D_epi_192(fake_epi_local), self.ones_192)
                    loss_GAN_BA = (loss_GAN_BA_512 + loss_GAN_BA_256 + loss_GAN_BA_192) / 3.0
                    
                    # total G_rou2epi loss
                    loss_G_rou2epi = (opt.lambda_adv * loss_GAN_BA) + (opt.lambda_cyc * loss_cyc) + (opt.lambda_id * loss_idt)
                    loss_G_rou2epi.backward()
                    self.optimizer_G_rou2epi.step()

                else:
                    pass 

                # print log
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [D: %f] [G_epi2rou loss: %f, G_rou2epi loss: %f]"
                    % (
                        epoch,
                        opt.n_epochs,
                        i+1,
                        self.n_sample / opt.batch_size,
                        # loss_D_512.item(),
                        # loss_D_256.item(),
                        loss_D_192.item(),
                        loss_G_epi2rou.item(),
                        loss_G_rou2epi.item(),
                    )
                )

            if epoch > opt.decay_epoch and epoch < opt.decay_epoch + 6000:
                self.lr_adjust(epoch)

            if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
                self.save_model(epoch, opt)

        return 0

    def lr_adjust(self, epoch):
        if epoch >= self.opt.decay_epoch + 4000: # qq 6
            self.lr_scheduler_G_epi2rou.step()
            self.lr_scheduler_G_rou2epi.step()
            self.lr_scheduler_D_EPI_512.step()
            self.lr_scheduler_D_ROU_512.step()
            self.lr_scheduler_D_EPI_256.step()
            self.lr_scheduler_D_ROU_256.step()
            self.lr_scheduler_D_EPI_192.step()
            self.lr_scheduler_D_ROU_192.step()
        
        elif epoch >= self.opt.decay_epoch + 2000:
            self.lr_scheduler_G_epi2rou.step()
            self.lr_scheduler_G_rou2epi.step()
            self.lr_scheduler_D_EPI_512.step()
            self.lr_scheduler_D_ROU_512.step()
            self.lr_scheduler_D_EPI_256.step()
            self.lr_scheduler_D_ROU_256.step()

        else: 
            self.lr_scheduler_G_epi2rou.step() 
            self.lr_scheduler_G_rou2epi.step()
            self.lr_scheduler_D_EPI_512.step()
            self.lr_scheduler_D_ROU_512.step()

    def save_model(self, epoch, opt):
        torch.save(self.G_epi2rou.state_dict(), "train_save/%s/G_epi2rou_%d.pth" % (opt.model_name, epoch))
        torch.save(self.G_rou2epi.state_dict(), "train_save/%s/G_rou2epi_%d.pth" % (opt.model_name, epoch))
        torch.save(self.D_epi_512.state_dict(), "train_save/%s/D_epi_512_%d.pth" % (opt.model_name, epoch))
        torch.save(self.D_rou_512.state_dict(), "train_save/%s/D_rou_512_%d.pth" % (opt.model_name, epoch))
        torch.save(self.D_epi_256.state_dict(), "train_save/%s/D_epi_256_%d.pth" % (opt.model_name, epoch))
        torch.save(self.D_rou_256.state_dict(), "train_save/%s/D_rou_256_%d.pth" % (opt.model_name, epoch))
        torch.save(self.D_epi_192.state_dict(), "train_save/%s/D_epi_192_%d.pth" % (opt.model_name, epoch))
        torch.save(self.D_rou_192.state_dict(), "train_save/%s/D_rou_192_%d.pth" % (opt.model_name, epoch))

if __name__ == '__main__':
    instance = CycleGAN_Trainer()
    instance.train()
        


        
    





            
        
