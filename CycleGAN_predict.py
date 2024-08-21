import os
import torch
import argparse
from CycleGAN_networks import *
from torch.utils.data import DataLoader
import pydicom
from natsort import natsorted
import cv2
import matplotlib.pyplot as plt

def reverse_normalization(source, modality): 
    max_value = 5200
    min_value = 0
    # source = source * 2600 + 2600
    source = source * (max_value - min_value) + min_value
    source -= 2048
    # source += 35
    return source


def reverse_min_max_norm(img, min_value, max_value):
    img = (img * (max_value - min_value)) + min_value
    return img

def min_max_norm(image, min_value, max_value):
    image = (image - min_value) / (max_value - min_value)
    return image


def write_dicom(origin_dir, save_path, images, fakeimage, predict_type, name=None, idx=0):
    fakeimage = np.asarray(fakeimage, dtype='int16')
    ds = pydicom.dcmread(os.path.join(origin_dir, images[0]), force=True)

    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    
    if predict_type == 'train':
        ds = pydicom.dcmread(os.path.join(origin_dir, images[0]))
        ds.PixelData = fakeimage.tobytes()
        saving_path = save_path + f'/{name}_{idx}'
        ds.save_as(saving_path, '.dcm')
        print(saving_path)

    elif predict_type == 'validation':
        for i in range(20):
            ds = pydicom.dcmread(os.path.join(origin_dir, images[i]))
            ds.PixelData = fakeimage[i].tobytes()
            saving_path = save_path + '/fake+{}'.format(i)
            ds.save_as(saving_path + '.dcm')   

def imread(path, img_res):
    dicom = pydicom.read_file(path)
    
    min_value = dicom.SmallestImagePixelValue
    max_value = dicom.LargestImagePixelValue

    bit = dicom.BitsStored
    bit_range = 2**bit
    source = dicom.pixel_array
    source[source > bit_range] = 0
    source = source.astype('uint16')
    
    if source.shape == img_res:
        return source, min_value, max_value
    else:
        source = cv2.resize(source, dsize=img_res, interpolation=cv2.INTER_AREA)
        return source, min_value, max_value

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']="0"
    print(torch.cuda.is_available())
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=5000)
    parser.add_argument("--model_name", type=str, default='test')
    parser.add_argument("--data_name", type=str, default="data_example", help="name of the dataset")
    parser.add_argument('--predict_type', type=str, default='validation')
    parser.add_argument("--img_height", type=int, default=512, help="size of image height")
    parser.add_argument("--img_width", type=int, default=512, help="size of image width")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    opt = parser.parse_args()

    cuda = torch.cuda.is_available()
    input_shape = (opt.channels, opt.img_height, opt.img_width) # (1, 256, 256)

    G_AB = BlurPooling_generator(input_shape)
    G_BA = BlurPooling_generator(input_shape)

    if cuda:
        G_AB = G_AB.cuda()
        G_BA = G_BA.cuda()
    
    if opt.epoch != 0:
        print(f'\n=============={opt.epoch} 번째 모델의 예측을 시작합니다.=============\n')
        G_AB.load_state_dict(torch.load("./train_save/%s/G_epi2rou_%d.pth" % (opt.model_name, opt.epoch), map_location='cpu'))
        G_BA.load_state_dict(torch.load("./train_save/%s/G_rou2epi_%d.pth" % (opt.model_name, opt.epoch), map_location='cpu'))

    if opt.predict_type == 'validation':
        G_AB.eval()
        G_BA.eval()

        n_patients = ['example_patient2']
        
        for n_patient in n_patients:
            Fake_EPI = list()
            Fake_Routine = list()

            save_path_Routine = f'./predict/{opt.model_name}/{n_patient}/fake_mid_{opt.epoch}'
            save_path_EPI = f'./predict/{opt.model_name}/{n_patient}/fake_low_{opt.epoch}'

            if os.path.exists(save_path_Routine) is False:
                os.makedirs(save_path_Routine)
            
            if os.path.exists(save_path_EPI) is False:
                os.makedirs(save_path_EPI)
    
            dcm_512_path = f'./data/{opt.data_name}/validation/{n_patient}/mid_dose' # Directory Path

            speeder_path = f"./data/{opt.data_name}/valid_patch/{n_patient}/low_dose"
            routine_path = f"./data/{opt.data_name}/valid_patch/{n_patient}/mid_dose"

            Speeders = os.listdir(speeder_path)
            Speeders = natsorted(Speeders)

            Routines = os.listdir(routine_path)
            Routines = natsorted(Routines)
            
            # G_AB EPI2Routine
            for img in Speeders:
                x = np.load(os.path.join(speeder_path, img)) 
                x = np.expand_dims(x, axis=0)
                x = np.expand_dims(x, axis=0)
                
                if cuda:
                    x = torch.from_numpy(x).to('cuda', dtype=torch.float32)
                else:
                    x = torch.from_numpy(x).cpu()
                    x = x.float()
                y = G_AB(x)

                y = y.view(512, 512)
                y = reverse_normalization(y, 'Routine_T1')
                
                Fake_Routine.append(y.cpu().detach().numpy())
            
            Fake_Routine = np.asarray(Fake_Routine, dtype='int16')

            images = os.listdir(dcm_512_path)
            images = natsorted(images)
            print('확인', os.path.join(dcm_512_path, images[0]))
            for i in range(Fake_Routine.shape[0]):
                ds = pydicom.dcmread(os.path.join(dcm_512_path, images[0]))
                ds.PixelData = Fake_Routine[i:i+1].tobytes()
                saving_path = save_path_Routine + '/fake_{}'.format(i)
                ds.save_as(saving_path + '.dcm')

            # # G_BA Routine2EPI
            # for img in Routines:
            #     x = np.load(os.path.join(routine_path, img))
            #     x = np.expand_dims(x, axis=0)
            #     x = np.expand_dims(x, axis=0)

            #     if cuda:
            #         x = torch.from_numpy(x).to('cuda', dtype=torch.float32)
            #     else:
            #         x = torch.from_numpy(x).cpu()
            #         x = x.float()

            #     y = G_BA(x)

            #     # y = torch.nn.functional.interpolate(y, size=(256, 256), mode='bicubic', align_corners=False)
            #     y = y.view(512, 512)

            #     y = reverse_normalization(y, 'EPI_T1')
                
            #     Fake_EPI.append(y.cpu().detach().numpy())
            
            # Fake_EPI = np.asarray(Fake_EPI, dtype='int16')\

            # images = os.listdir(dcm_512_path)
            # images = natsorted(images)

            # for i in range(Fake_EPI.shape[0]):
            #     ds = pydicom.dcmread(os.path.join(dcm_512_path, images[0]))
            #     ds.PixelData = Fake_EPI[i:i+1].tobytes()
            #     saving_path = save_path_EPI + '/fake_{}'.format(i)
            #     ds.save_as(saving_path + '.dcm')

