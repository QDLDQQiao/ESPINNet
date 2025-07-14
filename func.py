import sys
import numpy as np
from matplotlib import pyplot as plt 
import h5py

def prColor(word, color_type):
    ''' function to print color text in terminal
        input:
            word:           word to print
            color_type:     which color
                            'red', 'green', 'yellow'
                            'light_purple', 'purple'
                            'cyan', 'light_gray'
                            'black'
    '''
    end_c = '\033[00m'
    if color_type == 'red':
        start_c = '\033[91m'
    elif color_type == 'green':
        start_c = '\033[92m'
    elif color_type == 'yellow':
        start_c = '\033[93m'
    elif color_type == 'light_purple':
        start_c = '\033[94m'
    elif color_type == 'purple':
        start_c = '\033[95m'
    elif color_type == 'cyan':
        start_c = '\033[96m'
    elif color_type == 'light_gray':
        start_c = '\033[97m'
    elif color_type == 'black':
        start_c = '\033[98m'
    else:
        print('color not right')
        sys.exit()

    print(start_c + str(word) + end_c)

def read_h5(filepath, dict_key):
    with h5py.File(filepath,'r') as f:
        # Get the data
        data = f[dict_key][:].astype(np.float32)
    return data

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


def save_img(img_num, img_data, data_title, file_path, fig_size=(18, 8)):

    cbar_shrink = 0.05
    impad = 0.1
    plt.figure(figsize=fig_size)
    fig, axs = plt.subplots(img_num[0], img_num[1], figsize=fig_size)
    for jj in range(img_num[0]):
        for kk in range(img_num[1]):
            
            im = axs[jj, kk].imshow(img_data[jj][kk], cmap='Spectral')
            axs[jj, kk].set_axis_off()
            plt.colorbar(im, ax=axs[jj, kk], fraction=cbar_shrink, pad=impad)
            axs[jj, kk].set_title(data_title[jj][kk])
    plt.tight_layout()
    plt.savefig(file_path, dpi=150)
    plt.close('all')
