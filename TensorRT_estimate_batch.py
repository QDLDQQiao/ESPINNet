import torch
import math
import numpy as np
import PIL
import PIL.Image
import os
import json
import h5py
import time
from glob import glob
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import scipy.constants as sc
import scipy.ndimage as snd
import scipy.interpolate as sfit
import argparse

import multiprocessing as ms
import concurrent.futures

from data_prepare import Dataset_exp
from DS_network import Network

from func import prColor, save_img

import torch_tensorrt as torchtrt
# Only this extra line of code is required to use oneDNN Graph
# torch.jit.enable_onednn_fusion(True)


def stack_image_align(image_stack, offset_image_stack):
    '''
        here's a function to do the alignment of two images.
        the offset_image is shifted relatively to image to find the best position
        and return the shifted back offset_image
        input:
            image:              the first image
            offset_image:       the second image
        output:
            pos:                best shift postion to maximize the correlation
            image_back:         the image after alignment
    '''
    # from skimage.feature import register_translation
    from skimage.registration import phase_cross_correlation
    from scipy.ndimage import fourier_shift
    # roi = lambda x: x[0:100][0:100]
    # shift, error, diffphase = register_translation(image, offset_image, 10)

    shift, error, diffphase = phase_cross_correlation(image_stack[0], offset_image_stack[0], upsample_factor=10)

    print('shift dist: {}, alignment error: {} and phase difference: {}'.format(shift, error, diffphase))
    # image_back = image_shift(offset_image, shift[0], shift[1])
    image_back = np.array([fourier_shift(np.fft.fftn(img), shift) for img in offset_image_stack])
    image_back = np.real(np.fft.ifftn(image_back, axes=(-1, -2)))
    return shift, image_back

def write_h5(result_path, file_name, data_dict):
    ''' this function is used to save the variables in *args to hdf5 file
        args are in format: {'name': data}
    '''
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    with h5py.File(os.path.join(result_path, file_name+'.hdf5'), 'w') as f:
        for key_name in data_dict:
            f.create_dataset(key_name, data=data_dict[key_name], compression="gzip", compression_opts=9)
    print('result hdf5 file : {} saved'.format(file_name+'.hdf5'))

def write_npz(result_path, file_name, data_dict):
    ''' this function is used to save the variables in *args to hdf5 file
        args are in format: {'name': data}
    '''

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    # np.savez(os.path.join(result_path, file_name+'.npz'), **data_dict)
    np.savez_compressed(os.path.join(result_path, file_name+'.npz'), **data_dict)
    prColor('result npz file : {} saved'.format(file_name+'.npz'), 'green')


class args_setting:
    def __init__(self, dict_para):

        self.device = dict_para['device']
        self.OutputFolder = dict_para['OutputFolder']
        self.DataFolder = dict_para['DataFolder']
        self.TestFolder = dict_para['TestFolder']
        self.num_workers = dict_para['num_workers']
        self.batch_size = dict_para['batch_size']
        self.epoch = dict_para['epoch']
        self.lr = dict_para['lr']
        self.lv_chs = dict_para['lv_chs']
        self.output_level = dict_para['output_level']
        self.batch_norm = dict_para['batch_norm']
        self.corr = dict_para['corr']
        self.corr_activation = dict_para['corr_activation']
        self.search_range = dict_para['search_range']
        self.with_T = dict_para['with_T']
        self.with_D = dict_para['with_D']
        self.phaseC = dict_para['phaseC']
        self.fp16 = dict_para['fp16']
        self.with_refiner = dict_para['with_refiner']
        self.upsample_mode = dict_para['upsample_mode']
        self.load_all_data = dict_para['load_all_data']
        self.load_check_point = dict_para['load_check_point']
        self.device_type = torch.device(self.device)
        self.num_levels = len(self.lv_chs)

class Create_obj:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)

def estimate(input_stack, model, args):

    intWidth = input_stack[0].shape[-1]
    intHeight = input_stack[0].shape[-2]

    intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
    intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))
    
    input_stack = torch.nn.functional.interpolate(
                            input=input_stack,
                            size=(intPreprocessedHeight, intPreprocessedWidth),
                            mode='bilinear',
                            align_corners=True)

    with torch.no_grad():
        # Transfer to GPU
        if args.with_T:
            input_stack = input_stack.to(args.device_type)
            with torch.cuda.amp.autocast(enabled=args.fp16):
                flow_predict, T_predict, D_predict = model(input_stack)

            flow_predict = flow_predict.cpu().detach().numpy().astype(np.float32)
            T_predict = T_predict.cpu().detach().numpy().astype(np.float32)
            T_predict = np.minimum(T_predict, 1.5)

            D_predict = D_predict.cpu().detach().numpy().astype(np.float32)
            D_predict = np.minimum(D_predict, 1.5)

            flow_predict[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
            flow_predict[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

            return flow_predict[:, :, :, :], T_predict[:, :, :, :], D_predict, None
            
        else:
            input_stack = input_stack.to(args.device_type)
            with torch.cuda.amp.autocast(enabled=args.fp16):
                flow_predict = model(input_stack)
            
            flow_predict = flow_predict.cpu().detach().numpy().astype(np.float32)
            
            flow_predict[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
            flow_predict[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

            return flow_predict[:, :, :, :], None, None, None

def make_print_to_file(result_folder):
    import sys
    import os
    import sys
    import datetime

    class Logger(object):
        def __init__(self, filename='Default.log', path="./") -> None:
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), 'a', encoding='utf8')
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        
        def flush(self):
            pass
    
    filename = datetime.datetime.now().strftime('day'+'%Y_%m_%d_%H_%M_%S')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    sys.stdout = Logger(filename + '.log', path=result_folder)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print(torch.cuda.current_device())
    # parameters for the model prediction
    trained_model_path = './trained_model/training_model_001500.pt'
    trained_setting_path = './trained_model/setting_001500.json'

    device = 'cuda'
    batch = 1
    M_roi = 2048
    align = False
    energy_norm = False
    # data normalization method, max or mean
    norm_method = 'mean'
    # remove noise using the first reference image or not
    remove_noise = False
    dark = 'None'
    flat = 'None'

    # how many scans within single dataset
    N_datascan_num = 20
    # use N_scans images for the calculation, if N_scans < N_scan_model, data interpolation will be performed, otherwise, just select the first N_scan_model within the N_scans
    N_scans = 20
    # use N_scan_model for the calculation, this is fixed for the model
    N_scan_model = 20

    ref_path = './test_data/refs/'
    img_path = './test_data/sample_in/'
    
    result_folder = './test_data/ESPINNet_N{}/'.format(N_scans)

    ref_list = sorted(glob(os.path.join(ref_path, "*.tif")), key=lambda f: int(os.path.basename(f).split('.')[0].split('_')[-1]))
    img_list = sorted(glob(os.path.join(img_path, "*.tif")), key=lambda f: int(os.path.basename(f).split('.')[0].split('_')[-1]))

    ref_list = [ref_list[0:N_scans]] * 16
    img_list = [img_list[0:N_scans]] * 16

    make_print_to_file(result_folder)

    print('To be processed data number: {}\nRef data number: {}'.format(len(img_list), len(ref_list)))
      
    input_setting = {'model_path': trained_model_path,
                    'setting_path': trained_setting_path,
                    'device': device,
                    'result_folder': result_folder,
                    'ref_path': ref_path,
                    'img_path': img_path,
                    'N_scans': N_scans,
                    'N_scan_model': N_scan_model,
                    'N_datascan_num': N_datascan_num,
                    'dark': dark,
                    'flat': flat,
                    'align': align,
                    'batch': batch,
                    'energy_norm': energy_norm,
                    'remove_noise': remove_noise,
                    'M_roi':    M_roi,
                    }

    input_args = Create_obj(input_setting)

    for key, value in input_setting.items():
        print('{}: {}'.format(key, value))

    if not os.path.exists(input_setting['result_folder']):
        os.makedirs(input_setting['result_folder'])
    # use_cuda = torch.cuda.is_available()

    torch.backends.cudnn.benchmark = False
    # load the setting from the trained model

    with open(trained_setting_path) as f:
        setting_dict = json.load(f)

    setting_dict['device'] = device
    # save input parameters for inference
    with open(os.path.join(result_folder, 'setting.json'), 'w') as f:
        f.write(json.dumps(input_setting, indent=4))
        
    with open(os.path.join(result_folder, 'model_para.json'), 'w') as f:
        f.write(json.dumps(setting_dict, indent=4))

    # load model ----------------------------------------------
    torch.set_grad_enabled(False)
    torch.manual_seed(42)
    args = args_setting(setting_dict)
    
    # args.device_type = 'cuda:1'
    
    model = Network(args).to(args.device_type)
    checkpoint = torch.load(trained_model_path, map_location=args.device_type)
    print('load checkpoint from file: {}'.format(trained_model_path))

    model.load_state_dict(checkpoint['model_state_dict'])

    epoch = checkpoint['epoch']
    training_losses = checkpoint['training_loss']
    validation_losses = checkpoint['validation_loss']
    # from torch.nn.parallel import DistributedDataParallel as DDP
    # model = DDP(model)

    model.eval()

    predict_dataset = Dataset_exp(ref_list, img_list, M_roi=input_args.M_roi, N_scan=input_args.N_scan_model, align=input_args.align, energy_norm=input_args.energy_norm, dark=input_args.dark, flat=input_args.flat)
    predict_loader = torch.utils.data.DataLoader(
        predict_dataset, batch_size=input_args.batch, shuffle=False, num_workers=8, pin_memory=True)
    print('data loader initialization')
    dpc_x_recon = []
    dpc_y_recon = []
    T_recon = []
    D_recon = []
    int_phi_recon = []
    angle_recon = []

    if input_args.remove_noise:
        ref_dataset = Dataset_exp([ref_list[0]], [ref_list[0]], M_roi=input_args.M_roi, N_scan=input_args.N_scan_model, norm_method=norm_method)

        ref_stack = torch.from_numpy(np.expand_dims(ref_dataset[0][0], axis=0))
        print(ref_stack.shape)
        flow_stack_ref, T_stack_ref, D_stack_ref, phase_stack_ref = estimate(ref_stack, model, args)
        print('calculated the ref noise')

    # Tracing the model with example input
    # example_forward_input = [torch.rand(1, 40, input_args.M_roi, input_args.M_roi).to(args.device_type)]
    # traced_model = torch.jit.trace(model, example_forward_input).to(args.device_type)
    # Invoking torch.jit.freeze
    # traced_model = torch.jit.freeze(traced_model)

    from ctypes import cdll, c_char_p
    import torch.backends.cudnn as cudnn
    libcudart = cdll.LoadLibrary('libcudart.so')
    libcudart.cudaGetErrorString.restype = c_char_p
    def cudaSetDevice(device_idx):
        ret = libcudart.cudaSetDevice(device_idx)
        if ret != 0:
            error_string = libcudart.cudaGetErrorString(ret)
            raise RuntimeError("cudaSetDevice: " + error_string)
    
    cudaSetDevice(0)
    inputs = [torch.randn((1, 40, input_args.M_roi, input_args.M_roi)).to(args.device_type)]
    compile_settings = {
            "inputs": inputs,
            "enabled_precisions": {torch.float16},
            # "workspace_size": 2000000000,
            # "truncate_long_and_double": True,
        }
    
    trt_ts_module = torchtrt.compile(model, **compile_settings)

    for batch_id, (img_stack, img_list_batch)in enumerate(predict_loader):
        
        img_stack = img_stack.to(args.device_type)
        
        tic = time.time()
        flow_stack, T_stack, D_stack, phase_stack = estimate(img_stack, trt_ts_module, args)
        toc = time.time()
        print('prediction time: {:.6f}s, percentage: {:.2f}%'.format(toc - tic, 100*(batch_id+1)/len(predict_loader)))

        # save the estimation for each image pair
        for kk in range(flow_stack.shape[0]):
            img_prefix = os.path.basename(img_list_batch[0][kk].split('.')[0])
            Flow = flow_stack[kk, :, :, :]
            T = T_stack[kk, 0, :, :]
            D = D_stack[kk, 0, :, :]
            
            if input_args.remove_noise:
                DPC_x = Flow[0, :, :] - flow_stack_ref[0, 0, :, :]
                DPC_y = Flow[1, :, :] - flow_stack_ref[0, 1, :, :]
            else:
                DPC_x = Flow[0, :, :]
                DPC_y = Flow[1, :, :]
            
            