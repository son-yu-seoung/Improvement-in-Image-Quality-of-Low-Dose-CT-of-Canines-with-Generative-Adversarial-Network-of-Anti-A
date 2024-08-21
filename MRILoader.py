import os
import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from natsort import natsorted
import csv
from torchvision import transforms
from PIL import Image
import random
import matplotlib.pyplot as plt


class ImageTransform():
    def __init__(self,):
        self.data_transform = transforms.Compose([
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomVerticalFlip(p=0.5),
                            transforms.ToTensor()])

    def __call__(self, img):
        return self.data_transform(img)
    
class MRImgLoader(Dataset):
    def __init__(self, path, sequence, transform, batch_size = 1):
        self.base_path = path
        self.transform = transform
        self.target_path = os.path.join(self.base_path, sequence)
        self.batch_size = batch_size
        self.make_list()
                
    def __getitem__(self, index): 
        img = np.load(os.path.join(self.target_path, self.files[index]))
        # img = Image.fromarray(img[0])
        img = np.expand_dims(img, axis=0)
        # img = self.transform(img)
        # print('test2', img.shape)
        return img # (img - 0.5) * 2
        
    def __len__(self):
        return self.total_samples
            
    def make_list(self):
        self.files = os.listdir(self.target_path)
        self.files = natsorted(self.files)
        self.nFiles = len(self.files)
        self.n_batches = int(self.nFiles / self.batch_size)
        self.total_samples = int(self.n_batches * self.batch_size)


class RandomCrop():
    def __init__(self, in_size, tar_size, margin=50):
        self.in_size = in_size
        self.tar_size = tar_size
        self.margin = margin
        self.map_size = in_size - (margin * 2) # 462

    def __call__(self, img): # img type : tensor.Float
        img = img.cpu().detach().numpy()

        x = random.randint(0, self.map_size-self.tar_size) 
        y = random.randint(0, self.map_size-self.tar_size)
        # print()
        # print(f'y : {y}, x : {x}')
        # print(f'y : {y+self.margin}~{y+self.margin+self.tar_size}, x : {x+self.margin}~{x+self.margin+self.tar_size}')
        # print()

        crop_img = img[0,0][y+self.margin:y+self.margin+self.tar_size, x+self.margin:x+self.margin+self.tar_size]
        # test
        # print(crop_img.shape)
        # plt.figure(figsize=(12,8))

        # plt.subplot(1, 2, 1)
        # plt.imshow(img[0, 0], cmap='gray')

        # plt.subplot(1, 2, 2)
        # plt.imshow(crop_img, cmap='gray')

        # plt.show()

        crop_img = np.expand_dims(crop_img, axis=0)
        crop_img = np.expand_dims(crop_img, axis=0)

        # 범위 1 정도 오차 확인  (crop_img shape가 256, 192 잘 나온다.)
        # 실제 imshow로 crop 상태 확인 
         
        return torch.Tensor(crop_img).cuda().float()
    






# batch_size = 4
# EPI_dataset = MRImgLoader(path ="./data/EPIMix/train_patch/T1/", sequence="EPI", transform=ImageTransform(), batch_size=batch_size)
# Routine_dataset = MRImgLoader(path ="./data/EPIMix/train_patch/T1/", sequence="Routine", transform=ImageTransform(), batch_size=batch_size)
# EPI_loader = DataLoader(dataset=EPI_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
# Routine_loader = DataLoader(dataset=Routine_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# for iter_, (EPI, Routine) in enumerate(zip(EPI_loader, Routine_loader)):
#     print (EPI.size())
#     print (Routine.size())
#     print (iter_)
    
#     plt.imsave('./images/epi_{}.png'.format(0), EPI[0, 0, :, :], cmap=cm.gray)
#     plt.imsave('./images/epi_{}.png'.format(1), EPI[1, 0, :, :], cmap=cm.gray)
#     plt.imsave('./images/epi_{}.png'.format(2), EPI[2, 0, :, :], cmap=cm.gray)
#     plt.imsave('./images/epi_{}.png'.format(3), EPI[3, 0, :, :], cmap=cm.gray)

#     input("stop")



# dataiter = iter(dataloader)
# data = dataiter.next()
# x, y = data
# x = x.view(-1, 1, patch_size, patch_size)
# y = y.view(-1, 1, patch_size, patch_size)
# print (x.shape,y.shape)
