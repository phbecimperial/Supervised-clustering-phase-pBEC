"""
Generates Training data
"""

import random
import lzma

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from scipy.ndimage import rotate, gaussian_filter, zoom
from scipy.fft import fft2, fftshift
import cv2 as cv
from modes import mode_func, modelist
from LightPipes import * 
from tqdm import tqdm

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


def gererate_data(num, size, dim, modes, w0, noise=1, fringe_size=[0.2,0.5], 
                  wavelen=950*nm, spec_num = [0, 20], mult_las_split = 0.5, spec_rad = [1*um, 7*um], 
                  save = True, LG = True):
    images = []
    for i in tqdm(range(num)):
        beam = Begin(size=size, labda=wavelen, N=dim)
        beam1 = beam2 = beam
        comb, outputs = mode_func(mult_las_split)
        #comb = modes[np.random.randint(0, len(modes)-1)]

        amps = 0.05 + np.random.random(len(comb))*0.95
        amps = amps/max(amps)
        for j, (mode, amp) in enumerate(zip(comb, amps)):

            addbeam = GaussBeam(beam, w0=w0, n=mode[0], m=mode[1], LG=LG)
            addbeam.field = rotate(np.absolute(addbeam.field), angle = np.random.randint(0,360), reshape=False)
            addbeam = Normal(addbeam)
            addbeam = IntAttenuator(addbeam, amp)

            beam = BeamMix(beam,addbeam)
        beam = Normal(beam)


        f_angle = np.random.random() * 2 * np.pi
        f_size =  min(fringe_size) + np.random.random()*np.diff(fringe_size)[0]
        x_fringe = 1/f_size*100*um*np.cos(f_angle)
        y_fringe = 1/f_size*100*um*np.sin(f_angle)
        beam1 = PointSource(beam1, x=x_fringe, y=y_fringe)
        beam2 = PointSource(beam2, x=-x_fringe, y=-y_fringe)

        intbeam = BeamMix(beam1,beam2)
        intbeam = Fresnel(intbeam, z=1*cm)
        # beam = RandomIntensity(beam, np.random.randint(0, 1000), 
        #                        noise=noise*100*np.log(np.max(Intensity(beam))))
        warp_interference = noise_shift(Intensity(intbeam), (dim/500)**2*np.random.randint(5,20))
        beam = MultIntensity(beam, warp_interference)

        beam = Normal(beam)
        beam = Fresnel(beam, z=0.2*cm)

        for j in range(np.random.randint(min(spec_num), max(spec_num))):
            beam = CircScreen(beam, R = min(spec_rad) + np.random.random()*np.diff(spec_rad)[0],
                                x_shift=np.random.random()* 4 * w0 - 2 * w0,
                                y_shift=np.random.random()* 4 * w0 - 2* w0)

        beam = Forvard(beam, z=0.01*cm)

        aperture_radius = w0 + np.random.random()*size
        aperture_pos = np.random.random(2)*aperture_radius - aperture_radius/2
        beam = CircAperture(beam, R = aperture_radius, x_shift=aperture_pos[0], y_shift=aperture_pos[1])
        im = rotate(Intensity(beam)/np.max(Intensity(beam)), angle = np.random.randint(0,360), reshape=False)

        im = noise_shift(im, (im.shape[0]/500)**2*np.random.randint(5,20))

        im_avg = np.mean(im)
        im += im * np.random.random(im.shape) + np.random.random()*np.random.normal(im_avg/2, np.std(im), im.shape)

        im = (im + np.min(im)) / (np.max(im) + np.min(im))

        im_mid = int(im.shape[0]/2)
        im_crop = int(im.shape[0]/2.5)
        crop_im = im[im_mid - im_crop:im_mid + im_crop, im_mid - im_crop:im_mid + im_crop]

        im = zoom(crop_im, 224/im.shape[0])

        if save:
            # Using mgzip to compress pickles
            with lzma.open(r'MultiMode Analysis\Training_images\training_image' + '@' +
                           str(i) + '@' + ''.join(['1' if i else '0' for i in outputs]) + '.pkl.xz', 'wb') as f:
                pkl.dump((im, outputs), f)
            f.close()
        else:    
            images.append((im,outputs))
    return images


ims = gererate_data(5000, 2000*um, 300, modelist, 100*um, fringe_size=[0.5, 1.5], save = False, LG = False)


for i, (img, k) in enumerate(ims):
    plt.imshow(img)
    plt.title(str(k))
    plt.show()

