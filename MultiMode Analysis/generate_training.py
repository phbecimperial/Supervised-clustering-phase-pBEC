"""
Generates Training data
"""

import random
import lzma
import os
from os.path import sep
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from scipy.ndimage import rotate, gaussian_filter, zoom, shift
from scipy.fft import fft2, fftshift
import cv2 as cv
from modes import mode_func
from astropy.convolution import convolve, Gaussian2DKernel, TrapezoidDisk2DKernel
from LightPipes import * 
from tqdm import tqdm
import torch
from glob import glob

def sigmoid(x, k, c):
    return 1/(1+np.exp(-(x - c) / k))

def quick_norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def noise_shift(im, scale):
    sh = im.shape
    yy, xx = np.indices(sh)
    t = np.random.normal(size = sh)
    dx = gaussian_filter(t, sh[0]/500*np.random.randint(10,100), order=(0,1))
    dy = gaussian_filter(t, sh[1]/500*np.random.randint(10,100), order=(1,0))
    dx *= scale/dx.max()
    dy *= scale/dy.max()

    xmap = (xx-dx).astype(np.float32)
    ymap = (yy-dy).astype(np.float32)
    warped = cv.remap(im, xmap, ymap ,cv.INTER_LINEAR)

    return warped


def generate_data(num, size, dim, mode_params, w0, noise=1, fringe_size=[0.2,0.5], 
                  wavelen=950*nm, spec_num = [0, 20], mult_las_split = 0.5, spec_rad = [1*um, 7*um], 
                  save = True, save_dir = None, mode_func = mode_func, sigm = None):
    images = []

    for i in tqdm(range(num)):
        beam = Begin(size=size, labda=wavelen, N=dim)
        beam1 = beam2 = beam
        comb, outputs = mode_func(mult_las_split, mode_params, len(mode_params))
        # comb = modes[np.random.randint(0, len(modes)-1)]
        # comb = [1]
        # outputs = []
        amps = 0.3 + np.random.random(len(comb))*0.7
        amps = amps/max(amps)
        shifts = np.random.random(2)*dim/16 - dim/32

        
        # w = np.random.random(1)*(max(w0) - min(w0)) + min(w0)
        for j, (mode, amp) in enumerate(zip(comb, amps)):
            w = np.random.random(1)*(max(w0) - min(w0)) + min(w0)
            if mode['diameter'] is not False:
                w = np.random.random(1)*(max(mode['diameter']) - min(mode['diameter'])) + min(mode['diameter'])

            addbeam = GaussBeam(beam, w0=w, n=mode['mode'][0], m=mode['mode'][1], LG=mode['LG'])

            to_squash = quick_norm(Intensity(addbeam))

            if not (mode['mode'][0] == mode['mode'][1] == 0):
                to_squash = zoom(to_squash, [np.random.randint(50,100)/100,1])
            
            to_squash = to_squash[:,int((to_squash.shape[1] - to_squash.shape[0])/2):int((to_squash.shape[1] + to_squash.shape[0])/2)]

            trap = TrapezoidDisk2DKernel(np.random.randint(1,10), np.random.randint(15, 1000)/100,)
            # trap = TrapezoidDisk2DKernel(10, 0.0000)
            # print(trap.shape)

            to_squash = convolve(to_squash, trap, boundary= None)

            # gsize = np.random.randint(30, 100)
            # gaus = Gaussian2DKernel(gsize, gsize, x_size = to_squash.shape[0], y_size = to_squash.shape[1])._array
            # to_squash = to_squash*gaus

            addbeam.field = zoom(to_squash, [dim/to_squash.shape[0],dim/to_squash.shape[1]])


            if mode['angle'] is not False:
                addbeam.field = rotate(np.absolute(addbeam.field), angle = mode['angle'] + np.random.randint(-30,30), reshape=False)
            else:
                addbeam.field = rotate(np.absolute(addbeam.field), angle = np.random.randint(0,360), reshape=False)

            addbeam.field = quick_norm(addbeam.field)
            addbeam = IntAttenuator(addbeam, amp)

            # addbeam.field = Intensity(addbeam)
            beam.field += addbeam.field
        # beam = Normal(beam)

        beam.field = np.roll(np.array(beam.field), int(shifts[0]), 0)
        beam.field = np.roll(np.array(beam.field), int(shifts[1]), 1)

        # f_angle = np.random.random() * 2 * np.pi
        # f_size =  min(fringe_size) + np.random.random()*np.diff(fringe_size)[0]
        # x_fringe = 1/f_size*100*um*np.cos(f_angle)
        # y_fringe = 1/f_size*100*um*np.sin(f_angle)
        # beam1 = PointSource(beam1, x=x_fringe, y=y_fringe)
        # beam2 = PointSource(beam2, x=-x_fringe, y=-y_fringe)

        # intbeam = BeamMix(beam1,beam2)
        # intbeam = Fresnel(intbeam, z=1*cm)

        # beam = RandomIntensity(beam, np.random.randint(0, 1000),
        #                        noise=noise*100*np.log(np.max(Intensity(beam))))

        # warp_interference = noise_shift(Intensity(intbeam), (dim/500)**2*np.random.randint(5,20))
        # beam = MultIntensity(beam, warp_interference)

        # beam = Normal(beam)
        # beam = Fresnel(beam, z=0.2*cm)

        # for j in range(np.random.randint(min(spec_num), max(spec_num))):
        #     beam = CircScreen(beam, R = min(spec_rad) + np.random.random()*np.diff(spec_rad)[0],
        #                         x_shift=np.random.random()* 4 * w - 2 * w,
        #                         y_shift=np.random.random()* 4 * w - 2* w)

        beam = Forvard(beam, z=0.03*cm)

        aperture_radius = w + np.random.random()*size
        aperture_pos = np.random.random(2)*aperture_radius - aperture_radius/2
        #beam = CircAperture(beam, R = aperture_radius, x_shift=aperture_pos[0], y_shift=aperture_pos[1])
        # im = rotate(Intensity(beam)/np.max(Intensity(beam)), angle = np.random.randint(0,360), reshape=False)

        im = quick_norm(Intensity(beam))
        im += im * np.random.random(im.shape)/10
        if sigm is not None:
            im = sigmoid(im, sigm[0], sigm[1])
    

        im = quick_norm(im)
        # im = noise_shift(im, (im.shape[0]/500)**2*np.random.randint(1,20))
        im_max = np.max(im)
        im += im * np.random.random(im.shape)/10# + np.random.random()*0.5*np.random.normal(im_max/100, np.std(im), im.shape)

        im = 255 * quick_norm(im)

        im_mid = int(im.shape[0]/2)
        im_crop = int(im.shape[0]/4)
        crop_im = im[im_mid - im_crop:im_mid + im_crop, im_mid - im_crop:im_mid + im_crop]

        im = zoom(crop_im, 224/(im.shape[0]/2))
        im = np.round(im, decimals=1) / 255


        if save:
            with open(save_dir + r'\training_image' + '@' +
                      str(time.time()) + '@' + ''.join(
                ['1' if torch.all(i.eq(torch.tensor([1.,0.]))) else '0' for i in outputs]
                               ) + '.pkl', 'wb') as f:
                pkl.dump((im, outputs), f)
            f.close()
        else:    
            images.append((im,outputs))
    return images




# modelist = [
#     [0,0], [0,1], [0,2], [0,3], [1,1], [1,0]
# ]
# ims = gererate_data(30000, 2000*um, 300, modelist, 100*um, fringe_size=[0.5, 1.5], save = save, LG = True, mult_las_split=0)
#
#
# for i, (img, k) in enumerate(ims):
#     plt.imshow(img)
#     plt.title(str(k))
#     plt.show()

import concurrent.futures

def generate_data_worker(args):
    index, num, size, dim, modes, w0, noise, fringe_size, wavelen, spec_num, mult_las_split, spec_rad, save, save_dir, mode_func, sigm = args

    return generate_data(num, size, dim, modes, w0, noise, fringe_size, wavelen, spec_num, mult_las_split, spec_rad, save, save_dir, mode_func, sigm)

def generate_data_multithreaded(
        num_threads, num, size, dim, modes, w0, noise=1, fringe_size=[0.2,0.5],
        wavelen=950*nm, spec_num=[0, 20], mult_las_split=0.5, spec_rad=[1*um, 7*um], save=True, 
        save_dir = None, mode_func = mode_func, sigm = None):
    save = True
    if save:
        for f in glob(save_dir + r'\*'):
            os.remove(f)
    args_list = [(
        i, num, size, dim, modes, w0, noise, fringe_size, 
        wavelen, spec_num, mult_las_split, spec_rad, save, 
        save_dir, mode_func, sigm) for i in range(num_threads)]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(generate_data_worker, args_list))
    return results



# if save:
#     for f in glob(r'Training_images\*'):
#         os.remove(f)
#
# modelist = [
#     [0,0], [0,1], [0,2], [0,3], [0,4], [0,5], [1,1], [1,2], [1,3], [2,2], [1,0]
# ]




# ims = gererate_data(10, 2000*um, 300, [0,2], 100*um, [0.5, 1.5], save = False, LG=False)
# for i, (img, k) in enumerate(ims):
#     plt.imshow(img)
#     plt.title(str(k))
#     plt.show()
# modelist = [
#     ([0,0], False), ([0,1], False), ([0,2], False), ([0,3], False), ([0,4], False),  ([0,5], False),  ([0,6], False), ([0,7], False),  ([0,8], False),  ([0,9], False), 
#     ([1,1], False), ([1,2], False), ([1,3], False), ([1,4], False)
# 

# 2001 - 22
# modelist = [
#     ([0,0], False, 0, False), ([0,1], False, 155 - 90, [50*um, 200*um]), 
#     ([0,4], False, 70 + 90,False), ([0,6], False, 70 + 90, False), ([0,9], False, 70 + 90, False),# ([0,10], False, 70 + 90), ([0,8], False, 70 + 90)
# ]

# 2201
modelist = [
    ([0,0], False, 0, [150*um, 250*um]), ([0,1], False, 155 - 90, False), 
    ([0,4], False, 70 + 90,False), ([0,6], False, 70 + 90, False), ([0,9], False, 70 + 90, False), # ([0,10], False, 70 + 90, False), #  ([0,8], False, 70 + 90)
]

# modelist = [
#     ([0,9], False, False), ([0,1], False, True) 
# ]

#gererate_data(1, 2000*um, 300, [0,2], 100*um, [0.5, 1.5], save = True, LG=False)

#thread

if __name__ == '__main__':
    t = time.localtime()
    save = True
    save_dir = r'C:\Users\Pouis\Documents\Uni Shit\Masters\Test Images'
    num_threads = 2
    ims = generate_data_multithreaded(num_threads, 10 // num_threads, 2500*um, 500, modelist, [150*um, 200*um], fringe_size=[0.3, 0.6], save=save, mult_las_split=0, save_dir=save_dir)
