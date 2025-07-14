import os
import UMPA as u
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import scipy.constants as sc
from glob import glob
import time


def frankotchellappa(dpc_x, dpc_y):
    '''
        Frankt-Chellappa Algrotihm
        input:
            dpc_x:              the differential phase along x
            dpc_y:              the differential phase along y       
        output:
            phi:                phase calculated from the dpc
    '''
    fft2 = lambda x: np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))
    ifft2 = lambda x: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))
    fftshift = lambda x: np.fft.fftshift(x)
    # ifftshift = lambda x: np.fft.ifftshift(x)

    NN, MM = dpc_x.shape

    wx, wy = np.meshgrid(np.fft.fftfreq(MM) * 2 * np.pi,
                         np.fft.fftfreq(NN) * 2 * np.pi,
                         indexing='xy')
    wx = fftshift(wx)
    wy = fftshift(wy)
    numerator = -1j * wx * fft2(dpc_x) - 1j * wy * fft2(dpc_y)
    # here use the np.fmax method to eliminate the zero point of the division
    denominator = np.fmax((wx)**2 + (wy)**2, np.finfo(float).eps)

    div = numerator / denominator

    phi = np.real(ifft2(div))

    phi -= np.mean(np.real(phi))

    return phi

def image_roi(img, M):
    '''
        take out the interested area of the all data.
        input:
            img:            image data, 2D or 3D array
            M:              the interested array size
                            if M = 0, use the whole size of the data
        output:
            img_data:       the area of the data
    '''
    img_size = img.shape
    if not isinstance(M, list):
        if M == 0:
            return img
        elif len(img_size) == 2:
            if M > min(img_size):
                return img
            else:
                pos_0 = np.arange(M) - np.round(M / 2) + np.round(img_size[0] / 2)
                pos_0 = pos_0.astype('int')
                pos_1 = np.arange(M) - np.round(M / 2) + np.round(img_size[1] / 2)
                pos_1 = pos_1.astype('int')
                img_data = img[pos_0[0]:pos_0[-1] + 1, pos_1[0]:pos_1[-1] + 1]
        elif len(img_size) == 3:
            if M > min(img_size[1:]):
                return img
            else:
                pos_0 = np.arange(M) - np.round(M / 2) + np.round(img_size[1] / 2)
                pos_0 = pos_0.astype('int')
                pos_1 = np.arange(M) - np.round(M / 2) + np.round(img_size[2] / 2)
                pos_1 = pos_1.astype('int')
                img_data = np.zeros((img_size[0], M, M))
                for kk, pp in enumerate(img):
                    img_data[kk] = pp[pos_0[0]:pos_0[-1] + 1,
                                    pos_1[0]:pos_1[-1] + 1]
    elif isinstance(M, list):
        if len(img_size) == 2:
            pos_0 = np.arange(M[0]) - np.round(M[0] / 2) + np.round(img_size[0] / 2)
            pos_0 = pos_0.astype('int')
            pos_1 = np.arange(M[1]) - np.round(M[1] / 2) + np.round(img_size[1] / 2)
            pos_1 = pos_1.astype('int')
            img_data = img[pos_0[0]:pos_0[-1] + 1, pos_1[0]:pos_1[-1] + 1]
        elif len(img_size) == 3:
            
            pos_0 = np.arange(M[0]) - np.round(M[0] / 2) + np.round(img_size[1] / 2)
            pos_0 = pos_0.astype('int')
            pos_1 = np.arange(M[1]) - np.round(M[1] / 2) + np.round(img_size[2] / 2)
            pos_1 = pos_1.astype('int')
            img_data = np.zeros((img_size[0], M[0], M[1]))
            for kk, pp in enumerate(img):
                img_data[kk] = pp[pos_0[0]:pos_0[-1] + 1,
                                pos_1[0]:pos_1[-1] + 1]
    else:
        print('wrong M shape')

    return img_data

if __name__=='__main__':
    
    ref_path = './test_data/refs'
    img_path = './test_data/sample_in'

    save_results = True
    withDF = True  # use DF but N_window_size cannot be 0; otherwise use withDF=False

    M_roi = 2048

    N_image = 20
    N_window_size = 1

    ref_list = glob(os.path.join(ref_path, '*.tif'))
    img_list = glob(os.path.join(img_path, '*.tif'))

    ref = np.array([np.array(Image.open(ref_list[k])).astype(np.float32) for k in range(len(ref_list))])
    img = np.array([np.array(Image.open(img_list[k])).astype(np.float32) for k in range(len(img_list))])

    if M_roi > 0:
        ref = image_roi(ref, M_roi)[0:N_image,:,:]
        img = image_roi(img, M_roi)[0:N_image,:,:]
    else:
        ref = ref[0:N_image,:,:]
        img = img[0:N_image,:,:]

    print(ref.shape)

    t0 = time.time()

    if withDF:
        m = u.model.UMPAModelDF(img, ref, window_size=N_window_size, max_shift=10)
    else:
        m = u.model.UMPAModelNoDF(img, ref, window_size=N_window_size, max_shift=10)
    m.assign_coordinates = "ref"
    res = m.match(num_threads =128)

    
    ux , uy = -res["dx"], -res["dy"]
    if withDF:
        df = res['df']
    else:
        df = ux * 0
    t = res['T']

    t1 = time.time()
    print('processing time: {:.3f}s'.format(t1-t0))

    para_energy = 20e3
    para_px = 0.65e-6
    para_distance = 812e-3
    wavelength = sc.value('inverse meter-electron volt relationship') / para_energy

    phase = frankotchellappa(ux * para_px / para_distance,
                            uy * para_px / para_distance) * para_px * 2 * np.pi / wavelength

    cmap='gray'
    plt.figure(figsize=(6,10))
    plt.subplot(321)
    plt.imshow(ux, cmap=cmap)
    plt.title('dx')
    plt.colorbar()
    plt.clim([-10, 10])
    plt.subplot(322)
    plt.imshow(uy, cmap=cmap)
    plt.colorbar()
    plt.title('dy')
    plt.clim([-10, 10])
    plt.subplot(323)
    plt.imshow(df, cmap=cmap)
    plt.title('df')
    plt.colorbar()
    plt.subplot(324)
    plt.imshow(t, cmap=cmap)
    plt.title('T')
    plt.colorbar()
    plt.subplot(325)
    plt.imshow(phase, cmap=cmap)
    plt.title('phi')
    plt.colorbar()
    plt.clim([-30, 30])
    plt.savefig('UMPA.png')

    if save_results:
        save_folder = os.path.join(os.path.dirname(ref_path), 'UMPA_{}_nw{}_N{}'.format(withDF, N_window_size, N_image))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        plt.savefig(os.path.join(save_folder, 'UMPA.png'))

        np.savez(os.path.join(save_folder, 'recon.npz'), ux=ux, uy=uy, df=df, T=t)

        with open(os.path.join(save_folder, 'running_time.txt'), 'w') as f:
            f.write('running time: {:.3f}s'.format(t1-t0))
        
        print(save_folder)