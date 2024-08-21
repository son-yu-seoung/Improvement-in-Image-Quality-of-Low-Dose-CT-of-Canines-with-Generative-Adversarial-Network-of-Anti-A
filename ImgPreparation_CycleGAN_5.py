import os
import cv2
import numpy as np
import SimpleITK as itk
from glob import glob
import random
from natsort import natsorted
import pydicom
import matplotlib.pyplot as plt

class DataLoader():
    def __init__(self, basedir, data_name, img_res=(384, 384)):
        self.basedir = basedir
        self.data_name = data_name
        self.img_res = img_res
        self.n_batches = 480
        self.current_path = os.getcwd()

        random.seed(42)
    
    def imread(self, path, img_res):
        dicom = pydicom.read_file(path) # error
        bit = dicom.BitsStored
        bit_range = 2**bit
        source = dicom.pixel_array
        source[source > bit_range] = 0
        # source = source.astype('uint16')

        if self.img_res != source.shape:
            source = cv2.resize(source, dsize=img_res)
        
        return source
        
    def min_max_norm(self, image): # -2048 ~ 3000
        image += 2048

        max_value = 5200
        min_value = 0

        image = np.clip(image, min_value, max_value) 
        image = (image - min_value) / (max_value - min_value)

        return image

    def train_dcm2npy(self):
        cases = os.listdir(self.basedir)
        files = []
        for case in cases:
            file = case.split(".")
            if len(file) == 1:
                files.append(file[0])
        
        files = natsorted(files)

        ###
        images = os.listdir('./data/512')
        images = natsorted(images)
        ###

        count_low = 0
        count_mid = 0
        
        for case in files:
            low_dose_path = os.path.join(self.basedir, case, "low_dose")
            mid_dose_path = os.path.join(self.basedir, case, 'mid_dose')
            
            low_doses = natsorted(os.listdir(low_dose_path)) 
            mid_doses = natsorted(os.listdir(mid_dose_path))

            n_slice = min(len(low_doses), len(mid_doses))

            for i in range(0, n_slice, 4): # 0, 4, 8 : 0 + (0~3)
                # idx = random.randint(i, i+3)

                try:
                    img_low = self.imread(os.path.join(self.basedir, case, "low_dose", low_doses[i]), img_res=self.img_res)
                    img_mid = self.imread(os.path.join(self.basedir, case, "mid_dose", mid_doses[i]), img_res=self.img_res)
                
                except:
                    print('except : ', i)
                    img_low = self.imread(os.path.join(self.basedir, case, "low_dose", low_doses[i]), img_res=self.img_res)
                    img_mid = self.imread(os.path.join(self.basedir, case, "mid_dose", mid_doses[i]), img_res=self.img_res)

                norm_img_low = self.min_max_norm(img_low)
                norm_img_mid = self.min_max_norm(img_mid)

                
                
                np.save('./data/{}/train_patch/low_dose/low_{}.npy'.format(self.data_name, count_low), norm_img_low)
                np.save('./data/{}/train_patch/mid_dose/mid_{}.npy'.format(self.data_name, count_mid), norm_img_mid)
                count_low += 1
                count_mid += 1

    def validation_dcm2npy(self):
        cases = os.listdir(self.basedir)
        files = []
        for case in cases:
            file = case.split(".")
            if len(file) == 1:
                files.append(file[0])
        
        files = natsorted(files)

        ###
        images = os.listdir('./data/512')
        images = natsorted(images)
        ###
        
        for case in files:
            low_dose_path = os.path.join(self.basedir, case, "low_dose")
            mid_dose_path = os.path.join(self.basedir, case, 'mid_dose')
            
            low_doses = natsorted(os.listdir(low_dose_path)) 
            mid_doses = natsorted(os.listdir(mid_dose_path))

            count_low = 0
            count_mid = 0

            n_slice = min(len(low_doses), len(mid_doses))

            for i in range(0, n_slice, 1): # for imgs in low_doses: 
                img = self.imread(os.path.join(self.basedir, case, "low_dose", low_doses[i]), img_res=self.img_res)
                norm_img = self.min_max_norm(img)

                # # visualization
                # plt.figure(figsize=(12,7))
                # plt.subplot(1, 2, 1)
                # plt.xlabel('before')
                # plt.imshow(img, cmap='gray')

                # plt.subplot(1, 2, 2)
                # plt.xlabel('after')
                # plt.imshow(norm_img, cmap='gray')

                # plt.show()

                np.save('./data/{}/valid_patch/{}/low_dose/low_{}.npy'.format(self.data_name, case, count_low), norm_img)
                count_low += 1
            
            for i in range(0, n_slice, 1): # for imgs in mid_doses:
                img = self.imread(os.path.join(self.basedir, case, "mid_dose", mid_doses[i]), img_res=self.img_res)
                norm_img = self.min_max_norm(img)

                # # visualization
                # plt.figure(figsize=(12,7))
                # plt.subplot(1, 2, 1)
                # plt.xlabel('before')
                # plt.imshow(img, cmap='gray')

                # plt.subplot(1, 2, 2)
                # plt.xlabel('after')
                # plt.imshow(norm_img, cmap='gray')

                # plt.show()
               
                np.save('./data/{}/valid_patch/{}/mid_dose/mid_{}.npy'.format(self.data_name, case, count_mid), norm_img)
                count_mid += 1


data_loaer = DataLoader(basedir="./data/data_example/train", data_name='data_example', img_res=(512, 512))
data_loaer.train_dcm2npy()

data_loaer = DataLoader(basedir="./data/data_example/validation", data_name='data_example', img_res=(512, 512))
data_loaer.validation_dcm2npy()

