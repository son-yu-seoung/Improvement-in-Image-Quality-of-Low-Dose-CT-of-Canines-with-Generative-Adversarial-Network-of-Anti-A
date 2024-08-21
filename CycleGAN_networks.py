from time import sleep
from numpy.random import poisson
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.models import vgg19
# from spectral_normalization import SpectralNorm
from torch.nn.utils import spectral_norm
from cbam_attention import CBAM
# from pytorch_msssim import ms_ssim, ssim
import antialiased_cnns

def weights_init_normal(m):
    classname = m.__class__.__name__
    
    # if classname.find("Conv") != -1:
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02) # m = ConvBlock 등 
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.BatchNorm2d): #!= -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class ESA(nn.Module):
     def __init__(self, n_feats):
         super(ESA, self).__init__()
         f = n_feats // 4
         
         self.conv1 = nn.Conv2d(n_feats, f, kernel_size=1)         
         self.conv_f = nn.Conv2d(f, f, kernel_size=1)
         self.conv_max = nn.Conv2d(f, f, kernel_size=3, padding=1)
         self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)
         self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1)
         self.conv3_ = nn.Conv2d(f, f, kernel_size=3, padding=1)
         self.conv4 = nn.Conv2d(f, n_feats, kernel_size=1)
         self.sigmoid = nn.Sigmoid()
         self.relu = nn.ReLU(inplace=True)
  
     def forward(self, x, f):
         c1_ = (self.conv1(f))
         c1 = self.conv2(c1_)
         v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
         v_range = self.relu(self.conv_max(v_max))
         c3 = self.relu(self.conv3(v_range))
         c3 = self.conv3_(c3)
         c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners =False) 
         cf = self.conv_f(c1_)
         c4 = self.conv4(c3+cf)
         m = self.sigmoid(c4)
         
         return x * m


class ESABlurPool(nn.Module):
     def __init__(self, n_feats):
         super(ESABlurPool, self).__init__()
         f = n_feats // 4
         
         self.conv1 = nn.Conv2d(n_feats, f, kernel_size=1)         
         self.conv_f = nn.Conv2d(f, f, kernel_size=1)
         self.conv_max = nn.Conv2d(f, f, kernel_size=3, padding=1)
         self.conv21 = nn.Sequential(
            nn.Conv2d(f, f, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            antialiased_cnns.BlurPool(f, stride=2))
         
         self.conv22 = nn.Sequential(
             nn.Conv2d(f, f, kernel_size=3, stride=1, padding=1),
             nn.ReLU(inplace=True),
             antialiased_cnns.BlurPool(f, stride=2))
         
         self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1)
         self.conv3_ = nn.Conv2d(f, f, kernel_size=3, padding=1)
         self.conv4 = nn.Conv2d(f, n_feats, kernel_size=1)
         self.sigmoid = nn.Sigmoid()
         self.relu = nn.ReLU(inplace=True)
  
     def forward(self, x, f):
         c1_ = self.conv1(f)
         c1 = self.conv21(c1_)
         v_max = self.conv22(c1)
         v_range = self.relu(self.conv_max(v_max))
         c3 = self.relu(self.conv3(v_range))
         c3 = self.conv3_(c3)
         c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners =False) 
         cf = self.conv_f(c1_)
         c4 = self.conv4(c3+cf)
         m = self.sigmoid(c4)
         
         return x * m


class SEBlock(nn.Module):
    def __init__(self, in_channels, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1,1)) # (height=1, width=1)
        self.excitation = nn.Sequential(
            spectral_norm(nn.Linear(in_channels, in_channels // r)),
            nn.ReLU(),
            spectral_norm(nn.Linear(in_channels // r, in_channels)),
            nn.Sigmoid()
        )
        self._init_weight()


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        copy = x # [4, 32, 64, 64]
        x = self.squeeze(x) # [4, 32, 1, 1]
        x = x.view(x.size(0), -1) # [4, 32] # view는 원소의 개수의 유지가 가장 중요하다.
        x = self.excitation(x) # [4, 32]
        x = x.view(x.size(0), x.size(1), 1, 1) # [4, 32, 1, 1]
        return x * copy


class CABlock(nn.Module):
    def __init__(self, in_channels, r=16):
        super().__init__()
        self.squeeze = spectral_norm(nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0, bias=False))
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, x): # [4, 32, 64, 64]
        copy = x
        acti = F.sigmoid(self.squeeze(x)) # [4, 1, 64, 64]

        return acti * copy # [4, 32, 64, 64]


class FABlock(nn.Module):
    def __init__(self, in_channels = 32, r=16):
        super().__init__()
        self.nFilters = in_channels
        # default
        # self.squeeze_excite_block = SEBlock(self.nFilters)
        # ESA
        self.squeeze_excite_block = ESA(n_feats=self.nFilters)
        # ESA with BlurPool
        # self.squeeze_excite_block = ESABlurPool(n_feats=self.nFilters)
        
        self.channel_squeeze = CABlock(self.nFilters)
        self.skip_add = nn.quantized.FloatFunctional()
        
        self.conv_block = nn.Sequential(
            spectral_norm(nn.Conv2d(self.nFilters, self.nFilters, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.3),
            spectral_norm(nn.Conv2d(self.nFilters, self.nFilters, kernel_size=3, stride=1, padding=1))
            )

        self.final = nn.Sequential(
            spectral_norm(nn.Conv2d(self.nFilters*4, self.nFilters, kernel_size=1, stride=1, padding=0, bias=False)),
            nn.LeakyReLU(0.3),
        )
        for p in self.conv_block:
            if isinstance(p, nn.Conv2d):
                nn.init.xavier_uniform_(p.weight.data, 1.)

        for p in self.final:
            if isinstance(p, nn.Conv2d):
                nn.init.xavier_uniform_(p.weight.data, 1.)
        

    def Conv_SE_CE_block(self, x):
        conv = self.conv_block(x)
        # SE = self.squeeze_excite_block(conv)
        SE = self.squeeze_excite_block(conv, conv)
        CE = self.channel_squeeze(conv)

        return self.skip_add.add(SE, CE)

    def forward(self, x):
        identity = x

        conv1 = self.Conv_SE_CE_block(x)
        resd1 = self.skip_add.add(conv1, x) # +

        conv2 = self.Conv_SE_CE_block(resd1)
        resd2 = self.skip_add.add(conv2, resd1)

        conv3 = self.Conv_SE_CE_block(resd2)
        resd3 = self.skip_add.add(conv3, resd2)

        conv4 = self.Conv_SE_CE_block(resd3)

        aggregated = torch.cat((conv1, conv2, conv3, conv4), dim=1)
        final = self.final(aggregated)

        return final + identity


class BlurPooling_generator(nn.Module):
    def __init__(self, input_size):
        super(BlurPooling_generator, self).__init__()
        self.nFilters = 32
        self.input_size = input_size
        self.n_channel = int(input_size[0])
        self.n_hight = int(input_size[1])
        self.n_width = int(input_size[2])
        self.ratio = 4
        
        self.featurea_aggregation = FABlock(self.nFilters)
        self.pixel_unshuffle = nn.PixelUnshuffle(self.ratio)
        self.pixel_shuffle = nn.PixelShuffle(self.ratio)
        self.global_average_pooling = nn.AdaptiveAvgPool2d((int(self.n_hight/self.ratio), int(self.n_width/self.ratio)))
        self.channel_unpooling = nn.Conv2d(self.nFilters, self.nFilters * (self.ratio ** 2), kernel_size=3, stride=1, padding=1)
        
        self.encoding = nn.Sequential(
            nn.Conv2d((self.ratio ** 2), self.nFilters, kernel_size=9, stride=1, padding=4),
            nn.InstanceNorm2d(self.nFilters),
            nn.LeakyReLU(0.3),
        )
        
        self.blurpool_encoding_1 = nn.Sequential(
            spectral_norm(nn.Conv2d(1, self.nFilters, 3, stride=1, padding=1)),
            nn.InstanceNorm2d(self.nFilters),
            nn.LeakyReLU(0.2, inplace=True),
            antialiased_cnns.BlurPool(self.nFilters, stride=2)
        )
        
        self.blurpool_encoding_2 = nn.Sequential(
            spectral_norm(nn.Conv2d(self.nFilters, self.nFilters, 3, stride=1, padding=1)),
            nn.InstanceNorm2d(self.nFilters),
            nn.LeakyReLU(0.2, inplace=True),
            antialiased_cnns.BlurPool(self.nFilters, stride=2)
        )
                
        self.upsampling = nn.Sequential(
            spectral_norm(nn.Conv2d(self.nFilters, self.nFilters*4, 3, stride=1, padding=1)),
            nn.InstanceNorm2d(self.nFilters),
            nn.LeakyReLU(0.2, inplace=True),
            nn.PixelShuffle(upscale_factor=2)
        )
            
        self.decoding = nn.Sequential(
            nn.Conv2d(self.nFilters, self.nFilters, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(self.nFilters*4),
            nn.LeakyReLU(0.3),
            )

        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(self.nFilters, 1, kernel_size=7, stride=1, padding=0),
            nn.Sigmoid(),
            )

        for p in self.encoding:
            if isinstance(p, nn.Conv2d):
                nn.init.xavier_uniform_(p.weight.data, 1.)
        for p in self.decoding:
            if isinstance(p, nn.Conv2d):
                nn.init.xavier_uniform_(p.weight.data, 1.)
        for p in self.final:
            if isinstance(p, nn.Conv2d):
                nn.init.xavier_uniform_(p.weight.data, 1.)
        nn.init.xavier_uniform_(self.channel_unpooling.weight.data, 1.)
        
    def forward(self, x):
        x = self.blurpool_encoding_1(x)
        encoded = self.blurpool_encoding_2(x)
        
        x = self.featurea_aggregation(encoded)
        x = self.featurea_aggregation(x)
        
        x = x + encoded

        x = self.upsampling(x)
        x = self.upsampling(x)
        
        x = self.decoding(x)
        output = self.final(x)
        
        return output


class PatchGANDiscriminator_BlurPooling(nn.Module):
    def __init__(self, input_shape):
        super(PatchGANDiscriminator_BlurPooling, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            if normalize:
                layers = [spectral_norm(nn.Conv2d(in_filters, out_filters, 3, stride=1, padding=1))]
                layers.append(nn.BatchNorm2d(out_filters))

            else:
                layers = [spectral_norm(nn.Conv2d(in_filters, out_filters, 3, stride=1, padding=1))]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(antialiased_cnns.BlurPool(out_filters, stride=2))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)

