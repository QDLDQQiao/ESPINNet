import numpy as np
import torch
from PIL import Image
from func import image_roi
from scipy.ndimage import zoom, uniform_filter


def stack_TemplateWindow(img, N_w):
        """generate image stack for each window size

        Args:
            img (2D/3D numpy array): original image or image stack [image number, y, x]
            img_Nw: multi resolution image stack, [image number * (2+self.Nw_list+1), y, x]
        """
        if len(img.shape) == 2:
            img = img[np.newaxis, :, :]
        
        axis_Nw = np.arange(-N_w, N_w + 1)
        img_stack = []
        for x in axis_Nw:
            for y in axis_Nw:
                img_stack.append(
                    np.roll(np.roll(img, x, axis=-1), y, axis=-2))
        img_stack = np.moveaxis(np.array(img_stack), [0, 1], [-1, -2])

        return np.moveaxis(np.reshape(img_stack, (img_stack.shape[0], img_stack.shape[1], img_stack.shape[2]*img_stack.shape[3])), -1, 0)
    
    
class Dataset_exp(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch inference'

    def __init__(self, ref_list, img_list, M_roi, N_scan, dark='None', flat='None', upsampling=1, norm_method='max', data_augument_method='interp'):
        'Initialization'
        # the ref_list is a list of list
        self.ref_list = ref_list
        self.img_list = img_list
        self.M_roi = M_roi
        self.N_scan = N_scan

        # dark background for all the images
        self.dark_f = dark
        # flat for all the images
        self.flat_f = flat
        # upsample factor for the raw image
        self.upsampling = upsampling
        # most of time, use max; but if there's hot spot, use mean
        self.norm_method = norm_method
        # data interpolation mode, interpolation [interp] or use nearby subwindow stacking [stack]
        self.data_augument_method = data_augument_method

        if self.dark_f == 'None':
            self.dark = 0
        else:
            self.dark = np.array(Image.open(self.dark_f)).astype(np.float32)

        if self.flat_f == 'None':
            self.flat = 1
        elif self.flat_f == 'avg':
            pass
        else:
            self.flat = np.array(Image.open(self.flat_f)).astype(np.float32)

        # get the first sample image as starting point for energy correction
        self.img_ini = np.mean(np.array([np.array(Image.open(f)).astype(np.float32) for f in self.img_list[0]]), axis=0)
        print(self.img_ini.shape, 'done')
        self.img_ini = self.img_ini - self.dark
        # self.img_ini = (self.img_ini - self.dark) / (self.flat - self.dark)


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.img_list)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        item_img_list = self.img_list[index]
        
        item_ref_list = self.ref_list[index]
        # if the img number is less than required N_scan, use interpolation to get the data
        if self.N_scan > len(item_img_list):
            if self.data_augument_method == "interp":
                # do the interpolation
                print('zoom...', self.N_scan, len(item_img_list))
                img = np.array([np.array(Image.open(item_img_list[k])).astype(np.float32) for k in range(len(item_img_list))])
                img = zoom(img, (self.N_scan / img.shape[0], 1, 1), order=1)
                
                ref = np.array([np.array(Image.open(item_ref_list[k])).astype(np.float32) for k in range(len(item_ref_list))])
                ref = zoom(ref, (self.N_scan / ref.shape[0], 1, 1), order=1)
                    
            elif self.data_augument_method == 'stack':
                num_w = int(np.sqrt(self.N_scan/len(item_img_list))/2)
                # print(num_w)
                
                img = np.array([np.array(Image.open(item_img_list[k])).astype(np.float32) for k in range(len(item_img_list))])
                
                img = stack_TemplateWindow(img, num_w)
                
                if self.N_scan <= img.shape[0]:
                    img = img[0:self.N_scan, :, :]
                else:
                    img = zoom(img, (self.N_scan / img.shape[0], 1, 1), order=1)
                
                
                ref = np.array([np.array(Image.open(item_ref_list[k])).astype(np.float32) for k in range(len(item_ref_list))])
                
                ref = stack_TemplateWindow(ref, num_w)
                
                if self.N_scan <= ref.shape[0]:
                    ref = ref[0:self.N_scan, :, :]
                else:
                    ref = zoom(ref, (self.N_scan / ref.shape[0], 1, 1), order=1)
        else:
            img = np.array([np.array(Image.open(item_img_list[k])).astype(np.float32) for k in range(self.N_scan)])
            # print(img.shape)
            
            ref = np.array([np.array(Image.open(item_ref_list[k])).astype(np.float32) for k in range(self.N_scan)])
            # print(ref.shape)
        # remove dark background
        img = img - self.dark
        ref = ref - self.dark

        # background correction
        if self.flat_f == 'avg':
            flat = uniform_filter(np.mean(ref, axis=0), size=3)
        else:
            flat = self.flat
        img = img / (flat - self.dark)
        ref = ref / (flat - self.dark)

        img = image_roi(img, self.M_roi).astype(np.float32)
        ref = image_roi(ref, self.M_roi).astype(np.float32)

        # upsample the raw data if self.upsampling is not 1
        if self.upsampling == 1:
            if self.norm_method == 'max':
                img_stack = np.concatenate((ref, img), axis=0)/np.amax(ref)
            elif self.norm_method == 'mean':
                img_stack = np.concatenate((ref, img), axis=0)/np.mean(ref)
        else:
            print('data shape before upsampling', ref.shape)
            ref = zoom(ref, (1, self.upsampling, self.upsampling), order=1)
            img = zoom(img, (1, self.upsampling, self.upsampling), order=1)
            print('data shape after upsampling', ref.shape)
            if self.norm_method == 'max':
                img_stack = np.concatenate((ref, img), axis=0)/np.amax(ref)
            elif self.norm_method == 'mean':
                img_stack = np.concatenate((ref, img), axis=0)/np.mean(ref)
        
        return img_stack, self.img_list[index]