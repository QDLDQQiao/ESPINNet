import torch
import math
import numpy as np
import os
import json
import h5py
import time
from glob import glob
from matplotlib import pyplot as plt

import scipy.constants as sc

from scipy.ndimage import zoom

from data_prepare import Dataset_exp
from DS_network import Network

from func import frankotchellappa, prColor, save_img

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
    shift, error, diffphase = phase_cross_correlation(image_stack[0], offset_image_stack[0], upsample_factor=10)

    print('shift dist: {}, alignment error: {} and phase difference: {}'.format(shift, error, diffphase))
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
    # input_stack = torch.FloatTensor(np.ascontiguousarray(input_stack[:, :, :].astype(np.float32)))
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
                # flow_predict, T_predict, flow_list, T_list = model(img_stack)
                flow_predict, T_predict, D_predict = model(input_stack)

            flow_predict = flow_predict.cpu().detach().numpy().astype(np.float32)
            T_predict = T_predict.cpu().detach().numpy().astype(np.float32)
            T_predict = np.minimum(T_predict, 1.5)

            D_predict = D_predict.cpu().detach().numpy().astype(np.float32)
            D_predict = np.minimum(D_predict, 1.5)

            flow_predict[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
            flow_predict[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

            phase_predict = []
            for kk in range(flow_predict.shape[0]):
                phase_predict.append(frankotchellappa(flow_predict[kk, 0, :, :], flow_predict[kk, 1, :, :]))
            phase_predict = np.array(phase_predict)

            return flow_predict[:, :, :, :], T_predict[:, :, :, :], D_predict, phase_predict
            
        else:
            input_stack = input_stack.to(args.device_type)
            with torch.cuda.amp.autocast(enabled=args.fp16):
                flow_predict = model(input_stack)
            
            flow_predict = flow_predict.cpu().detach().numpy().astype(np.float32)
            
            flow_predict[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
            flow_predict[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

            phase_predict = []
            for kk in range(flow_predict.shape[0]):
                phase_predict.append(frankotchellappa(flow_predict[kk, 0, :, :], flow_predict[kk, 1, :, :]))
            phase_predict = np.array(phase_predict)

            return flow_predict[:, :, :, :], None, None, phase_predict

def plot_figure(file_path, img, p_x=1, cbar_label='', show_fig=False):
    extent_data = np.array([
                    -img.shape[1] / 2 * p_x,
                    img.shape[1] / 2 * p_x,
                    -img.shape[0] / 2 * p_x,
                    img.shape[0] / 2 * p_x
                ])
    # Spectral
    c_map = 'gray'
    plt.figure()
    # plt.imshow(img, cmap=cm.get_cmap('gist_gray'), interpolation='bilinear',
    #         extent=extent_data*1e6)
    plt.imshow(img,
            interpolation='bilinear',
            extent=extent_data, cmap=c_map)
    if p_x == 1:
        plt.xlabel('x (pixel)', fontsize=22)
        plt.ylabel('y (pixel)', fontsize=22)
    else:
        plt.xlabel('x (m)', fontsize=22)
        plt.ylabel('y (m)', fontsize=22)
    cbar = plt.colorbar()
    cbar.set_label(cbar_label, rotation=90, fontsize=20)
    plt.tight_layout()
    plt.savefig(file_path,
                dpi=150,
                transparent=True)
    if not show_fig:
        plt.close()

def save_results(result_folder, filename, data, save_fig=False):

    write_h5(result_folder, filename, data)
    if save_fig:
        save_img((2, 3), img_data=[[data['displace_x'], data['T'], data['phase']],
                                    [data['displace_y'], data['D'], data['phase']]], data_title=[
                                                ['dx', 'T', 'phi'],['dy', 'D', 'phi']
                                            ], file_path=os.path.join(result_folder, filename+'.png'), fig_size=(9, 6))

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
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # parameters for the model prediction
    trained_model_path = './trained_model/training_model_001500.pt'
    trained_setting_path = './trained_model/setting_001500.json'

    device = 'cuda'
    
    # area size needs to be int-times of 64, otherwise the output size is different
    M_roi = 1024

    # data normalization method, max or mean
    norm_method='mean'
    # upsampling fator for the raw data
    upsampling_factor = 1

    # experimental parameters
    para_energy = 20e3
    para_px = 0.65e-6
    para_distance = 812e-3

    # how many scans within single dataset
    N_datascan_num = 20
    # use N_scans images for the calculation, if N_scans < N_scan_model, data interpolation will be performed, otherwise, just select the first N_scan_model within the N_scans
    N_scans = 10
    # scan_mode, 1 for linear, n for 2D scan
    scan_mode = 1

    # use N_scan_model for the calculation, this is fixed for the model
    N_scan_model = 20

    ref_path = './test_data/refs/'
    img_path = './test_data/sample_in/'
    result_folder = os.path.join('./result/', 'ESPINNet')

    dark = 'None'
    flat = 'None'
    # flat = 'D:/exp_data/2023/20230424_13HB/IC2_20keV/unguangshan_200ms.tif'

    ref_list = sorted(glob(os.path.join(ref_path, "*.tif")), key=lambda f: int(os.path.basename(f).split('.')[0].split('_')[-1]))
    img_list = sorted(glob(os.path.join(img_path, "*.tif")), key=lambda f: int(os.path.basename(f).split('.')[0].split('_')[-1]))

    if N_scans > N_datascan_num:
        pass
    else:
        if scan_mode == 1:
            # if the scan mode is linear, select the first part
            ref_list = [ref_list[i:i+N_scans] for i in range(0, len(ref_list), N_datascan_num)]
            img_list = [img_list[i:i+N_scans] for i in range(0, len(img_list), N_datascan_num)]
        else:
            # if the scan mode is XY 2D square scan, select the 2D subpart with same step size
            num_2ndDim_steps = int(N_datascan_num/scan_mode)
            n_scan_1stDim = int(np.sqrt(N_scans))
            sub_id = []
            for ii in range(scan_mode):
                sub_id += [ii*scan_mode + jj for jj in range(n_scan_1stDim)]
            sub_id = sub_id[0:N_scans]
            print('Scan mode\n1st Dim: {}; 2nd Dim: {}'.format(scan_mode, num_2ndDim_steps))
            print('use the following XY scan: ', sub_id)
            ref_list = [ref_list[i*N_datascan_num:i*N_datascan_num+N_datascan_num] for i in range(0, len(ref_list), N_datascan_num)]
            
            img_list = [img_list[i*N_datascan_num:i*N_datascan_num+N_datascan_num] for i in range(0, len(ref_list), N_datascan_num)]

            for i in range(len(ref_list)):
                ref_list[i] = [ref_list[i][k] for k in sub_id]
                img_list[i] = [img_list[i][k] for k in sub_id]

    print(img_list)

    make_print_to_file(result_folder)

    para_exp = {
        'trained_model_path': trained_model_path,
        'trained_setting_path': trained_setting_path,
        'device': device,
        'result_folder': result_folder,
        'M_roi': M_roi,
        'norm_method': norm_method,
        'dark': dark,
        'flat': flat,
        'energy': para_energy,
        'distance': para_distance,
        'p_x': para_px,
        'N_datascan_num': N_datascan_num,
        'N_scans': N_scans,
        'N_scan_model': N_scan_model,
        'ref_path': ref_path,
        'img_path': img_path,
    }

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
                    'batch': 1,
                    'flat': flat,
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
    
    with open(os.path.join(result_folder, 'exp_para.json'), 'w') as f:
        f.write(json.dumps(para_exp, indent=4))

    # load model ----------------------------------------------
    torch.set_grad_enabled(False)
    torch.manual_seed(42)
    args = args_setting(setting_dict)
    model = Network(args).to(args.device_type)
    checkpoint = torch.load(trained_model_path, map_location=args.device_type)
    print('load checkpoint from file: {}'.format(trained_model_path))

    model.load_state_dict(checkpoint['model_state_dict'])

    epoch = checkpoint['epoch']
    training_losses = checkpoint['training_loss']
    validation_losses = checkpoint['validation_loss']

    model.eval()

    predict_dataset = Dataset_exp(ref_list, img_list, M_roi=input_args.M_roi, N_scan=input_args.N_scan_model, dark=input_args.dark, flat=input_args.flat, upsampling=upsampling_factor,norm_method=norm_method)
    predict_loader = torch.utils.data.DataLoader(
        predict_dataset, batch_size=input_args.batch, shuffle=False, num_workers=8, pin_memory=False)
    print('data loader initialization')


    for batch_id, (img_stack, img_list_batch)in enumerate(predict_loader):
        # print(img_list_batch)
        tic = time.time()
        flow_stack, T_stack, D_stack, phase_stack = estimate(img_stack, model, args)
        toc = time.time()
        print('prediction time: {:.6f}s, percentage: {:.2f}%'.format(toc - tic, 100*(batch_id+1)/len(predict_loader)))

        # save the estimation for each image pair
        for kk in range(flow_stack.shape[0]):
            img_prefix = os.path.basename(img_list_batch[0][kk].split('.')[0])
            Flow = flow_stack[kk, :, :, :]
            T = T_stack[kk, 0, :, :]
            D = D_stack[kk, 0, :, :]
            int_phi = phase_stack[kk, :, :]
            # ratio = img_warp / ref

            DPC_x = Flow[0, :, :]
            DPC_y = Flow[1, :, :]
            prColor('id for inference: {}'.format(batch_id), 'purple')

    # scale back the data 
    
    if upsampling_factor != 1:
        DPC_x = zoom(DPC_x, (1/upsampling_factor, 1/upsampling_factor), order=1) / upsampling_factor
        DPC_y = zoom(DPC_y, (1/upsampling_factor, 1/upsampling_factor), order=1) / upsampling_factor
        T = zoom(T, (1/upsampling_factor, 1/upsampling_factor), order=1)
        D = zoom(D, (1/upsampling_factor, 1/upsampling_factor), order=1)

    wavelength = sc.value('inverse meter-electron volt relationship') / para_energy
    phase = frankotchellappa(DPC_x * para_px / para_distance,
                            DPC_y * para_px / para_distance) * para_px * 2 * np.pi / wavelength

    plot_figure(os.path.join(result_folder, 'dpc_x_{}.png'.format(kk)), DPC_x, cbar_label='urad', show_fig=False)
    plot_figure(os.path.join(result_folder, 'dpc_y_{}.png'.format(kk)), DPC_y, cbar_label='urad', show_fig=False)
    plot_figure(os.path.join(result_folder, 'phase_{}.png'.format(kk)), phase, cbar_label='rad', show_fig=False)
    plot_figure(os.path.join(result_folder, 'T_{}.png'.format(kk)), T, cbar_label='(a.u.)', show_fig=False)
    plot_figure(os.path.join(result_folder, 'D_{}.png'.format(kk)), D, cbar_label='(a.u.)', show_fig=False)

    save_results(result_folder, 'Result_UltraSPINNet', {
                    'displace_x': DPC_x.astype(np.float32),
                    'displace_y': DPC_y.astype(np.float32),
                    'T': T.astype(np.float32),
                    'D': D.astype(np.float32),
                    'phase': phase.astype(np.float32)
                    }, False)