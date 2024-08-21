import os
import numpy as np
import natsort 		# For natural sorting
import cv2
import random
import SimpleITK as itk
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable
import torch
import numpy as np

from torchvision.utils import save_image


def slice_array(X, start=None, stop=None):
    if hasattr(start, '__len__'):
        return X[start]
    else:
        return X[start:stop]

def samplewise_zero_center_and_std_norm(X):
    mean = np.mean(X, axis=(1, 2, 3))
    stddev = np.std(X, axis=(1, 2, 3))

    nbatch = X.shape[0]
    for i in range(nbatch):
        X[i] = (X[i] - mean[i]) / stddev[i]

    return X
def data_clipping(X, min, max):
    return np.clip(X, int(min), int(max))


def max_value_norm(X, max_=2048, min_=-1024):
    
    X = np.clip(X, min_, max_)
    max_value = np.max(X, axis=(1, 2, 3))

    # make intensitiy range as [-1, 1]
    nbatch = X.shape[0]
    for i in range(nbatch):
        X[i] = (X[i]/max_value[i] - 0.5) * 2.0

    return X

def reverse_max_value_norm(source, target):
    max_value = np.max(source, axis=(0, 1, 2))

    return ((target / 2.0) + 0.5) * max_value
        

# def ct_tanh_norm(X, max_=2048, min_=-1024):
    
#     X = np.clip(X, -1024, 2048)

#     max_value = np.max(X, axis=(1, 2, 3))
#     min_value = np.full(max_value.shape, -1024)
#     # min_value = np.min(X, axis=(1, 2, 3))
#     for i in range(X.shape[0]):
#         X[i] = ((X[i] - min_value[i]) / (max_value[i] - min_value[i]))*2 - 1.0

#     return X, min_value, max_value
     
def ct_tanh_norm(img, min_=-1024, max_=2048):
    img = np.clip(img, -1024, 2048)
    max_value = np.max(img)
    min_value = np.min(img)

    img = ((img - min_value) / (max_value - min_value)) * 2.0 - 1.0

    return img, min_value, max_value

# rescale [-1, 1] to [-1024, 2048]
def tanh2intensity(img, min_value, max_value):

    img = (((img + 1.0)*0.5) * (max_value - min_value)) + min_value

    return torch.clip(img, -1024, 2048)
    # return np.clip(img, -1024, 2048)


def reverse_ct_tanh_norm(X, min_value, max_value):
    
    for i in range(X.shape[0]):
        X = (((X + 1.0)*0.5) * (max_value - min_value)) + min_value
        
    X = np.clip(X, -1024, 2048)

    return X

def cropping(img, width, height):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    
    return img[y:y+height, x:x+width]

# def randomcrop(img, n, width, height):
#     temp = []
#     for i in range(n):
#         temp.append(cropping(img, width, height))

#     return torch.Tensor(temp)
#     # return np.array(temp)

def randomcrop(img, n, width, height):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = random.randint(0, img.shape[1] - (width * n))
    y = random.randint(0, img.shape[0] - (height* n))
    cropped = img[y:y+(height*n), x:x+(width*n)]
    cropped = cropped.reshape(-1, height, width)
    
    index = random.sample(list(np.arange(n*n)), 5)
    
    return cropped[index]


def _convert_to_patchs(img):
    Images = []
    for i in range(img.shape[0]):
        image  = np.squeeze(img[i])
        Images.append(cropping(image, 64, 64))
        Images.append(cropping(image, 64, 64))
        Images.append(cropping(image, 64, 64))
        Images.append(cropping(image, 64, 64))
        Images.append(cropping(image, 64, 64))

    temp = np.asarray(Images, dtype=np.float32)
    temp = np.expand_dims(temp, axis=3)

    return temp




def HU_intensity_norm(X, max_=2048, min_=-1024):

    X = np.clip(X, min_, max_)
    X = X.astype(np.float32)
    
    max_value = np.max(X, axis=(1, 2, 3))
    min_value = np.min(X, axis=(1, 2, 3))
    for i in range(X.shape[0]):
        X[i] = (X[i] - min_value[i]) / (max_value[i] - min_value[i])

    return X

def add_noise(X, max_=2048, min_=-1024):
    
    X = X.astype(np.float32)
    noise_factor = 0.5
    x_train_noisy = X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape) 
    return np.clip(x_train_noisy, -1.0, 1.0)

def revers_HU_intensity_norm(X, Y, max_=2048, min_=-1024):

    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    
    max_value = np.max(X)
    min_value = np.min(X)
    img = Y * (max_value - min_value) + min_value
        
    return np.clip(img, min_, max_)
    
    
def fixed_value_norm2(image):
    MIN_BOUND = -2048.0
    MAX_BOUND = 1500.0
    
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

def fixed_value_centering(image):
    PIXEL_MEAN = 0.5

    image = image - PIXEL_MEAN
    return image

def decentering_with_fixed_value(image):
    PIXEL_MEAN = 0.5

    image = image + PIXEL_MEAN
    return image
    
def fixed_value_norm(X):
    mean = 1024.0
    maxvalue = 4000.0

    return (X + mean)/maxvalue

def _random_blur(X, sigma_max = 5.0):
    for i in range(len(X)):
        if bool(random.getrandbits(1)):
            # Random sigma
            sigma = random.uniform(0., sigma_max)
            X[i] = ndimage.filters.gaussian_filter(X[i], sigma)
    return X

def shuffle(*arrs):
    """ shuffle.
    Shuffle given arrays at unison, along first axis.
    Arguments:
        *arrs: Each array to shuffle at unison.
    Returns:
        Tuple of shuffled arrays.
    """
    arrs = list(arrs)
    for i, arr in enumerate(arrs):
        assert len(arrs[0]) == len(arrs[i])
        arrs[i] = np.array(arr)
    p = np.random.permutation(len(arrs[0]))
    return tuple(arr[p] for arr in arrs)

def load_imagedata(filenames, folder_path, output_size):
    nImage = len(filenames)
    Images = []
    for i in range(nImage):
        path = os.path.join(folder_path, filenames[i])
        ImgData = itk.ReadImage(path)
        ImgArray = itk.GetArrayFromImage(ImgData)   
        source = np.asarray(ImgArray, dtype=np.float32)
        Images.append(source)

    temp = np.asarray(Images, dtype=np.float32)
    temp = np.squeeze(temp)
    temp = np.expand_dims(temp, axis=3)
    
    return temp


def img_read(path):
    ImgData = itk.ReadImage(path)
    ImgArray = itk.GetArrayFromImage(ImgData)   
    
    return np.asarray(ImgArray, dtype=np.float32)


def AdjustPixelRange(pData, Lower, Upper):
    # version 2 
    range_ratio = (Upper - Lower) / 255.0

    img_adjusted = (pData - Lower)/range_ratio
    # return torch.clip(img_adjusted, 0, 255)

    img_adjusted = img_adjusted.clip(0, 255)
    return img_adjusted


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=4096):
    L = val_range
    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
    C1 = (0.01 * L)**2
    C2 = (0.03 * L)**2
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)
    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=4096, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)
        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))
    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)
    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2
    pow1 = mcs ** weights
    pow2 = mssim ** weights
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


class Structural_loss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(Structural_loss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, img1, img2):
        MS_SSIM = msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)
        return 1 - MS_SSIM



class ReplayBuffer: 
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data): # fake_A [4, 1, 256, 256]
        to_return = []
        for element in data.data: # element [1, 256, 256] 
            element = torch.unsqueeze(element, 0) # [1, 1, 256, 256] ?, slice별로 나누기 위해서 Buffer?

            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
