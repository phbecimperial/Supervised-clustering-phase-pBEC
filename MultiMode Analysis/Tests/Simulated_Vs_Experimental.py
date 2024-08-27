import matplotlib.pyplot as plt
from LightPipes import *
from scipy.ndimage import rotate, zoom
from scipy.special import erf
from astropy.convolution import convolve, Gaussian2DKernel, TrapezoidDisk2DKernel
from tqdm import tqdm
import cv2
from os.path import sep
from generate_training import noise_shift,quick_norm

def sigmoid(x, k, c):
    return 1/(1+np.exp(-(x - c) / k))

def skew_gauss(x, a, c, sig):
    gaus = 1/2 * np.exp(-((x+c)/(2*sig)**2))
    ef = 1 + erf((a*x + c)/np.sqrt(2))

    return gaus*ef

def gererate_data(num, size, dim, modes, w0, noise=1, fringe_size=[0.2,0.5], 
                  wavelen=950*nm, spec_num = [0, 20], mult_las_split = 0.5, spec_rad = [1*um, 7*um], 
                  save = True, save_dir = None):
    images = []

    for i in tqdm(range(num)):
        beam = Begin(size=size, labda=wavelen, N=dim)
        beam1 = beam2 = beam
        # comb = modes[np.random.randint(0, len(modes)-1)]
        # comb = [1]
        # outputs = []
        # amps = 0.3 + np.random.random(len(comb))*0.7
        # amps = amps/max(amps)
        # shifts = np.random.random(2)*dim/32 - dim/64

        
        # w = np.random.random(1)*(max(w0) - min(w0)) + min(w0)
        for j, mode in enumerate(modes):
            w = np.random.random(1)*(max(w0) - min(w0)) + min(w0)
            if mode[3] is not False:
                w = mode[3]

            addbeam = GaussBeam(beam, w0=w, n=mode[0][0], m=mode[0][1], LG=mode[1])
            to_squash = Intensity(addbeam)/np.max(Intensity(addbeam))

            to_squash = zoom(to_squash, [mode[5],1])
            to_squash = to_squash[:,int((to_squash.shape[1] - to_squash.shape[0])/2):int((to_squash.shape[1] + to_squash.shape[0])/2)]
            # to_squash = gaussian_filter(to_squash, [8,8])
            to_squash = convolve(to_squash, TrapezoidDisk2DKernel(bl[0], bl[1]), boundary= None)

            # to_squash = to_squash/(np.sum(to_squash))
            # plt.imshow(to_squash)
            # plt.show()
            addbeam.field = zoom(to_squash, [dim/to_squash.shape[0],dim/to_squash.shape[1]])

            if mode[2] is not False:
                addbeam.field = rotate(np.absolute(addbeam.field), angle = mode[2], reshape=False)
            else:
                addbeam.field = rotate(np.absolute(addbeam.field), angle = np.random.randint(0,360), reshape=False)

            addbeam.field = quick_norm(addbeam.field)
            addbeam = IntAttenuator(addbeam, mode[4])

            addbeam.field = Intensity(addbeam)
            beam.field += addbeam.field
        beam = Normal(beam)

        # beam.field = np.roll(np.array(beam.field), int(shifts[0]), 0)
        # beam.field = np.roll(np.array(beam.field), int(shifts[1]), 1)

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
        # beam = Fresnel(beam, z=2*cm)

        # for j in range(np.random.randint(min(spec_num), max(spec_num))):
        #     beam = CircScreen(beam, R = min(spec_rad) + np.random.random()*np.diff(spec_rad)[0],
        #                         x_shift=np.random.random()* 4 * w - 2 * w,
        #                         y_shift=np.random.random()* 4 * w - 2* w)

        # beam = Forvard(beam, z=0.03*cm)

        # aperture_radius = w + np.random.random()*size
        # aperture_pos = np.random.random(2)*aperture_radius - aperture_radius/2
        #beam = CircAperture(beam, R = aperture_radius, x_shift=aperture_pos[0], y_shift=aperture_pos[1])
        # im = rotate(Intensity(beam)/np.max(Intensity(beam)), angle = np.random.randint(0,360), reshape=False)

        im = quick_norm(Intensity(beam))

        gaus = Gaussian2DKernel(sz, sz,x_size = im.shape[0], y_size = im.shape[1])._array
        gaus /= np.max(gaus)
        im *= gaus
        # im = im/np.max(im)
        # im = skew_gauss(im, 100, 0.001, 5)
        # im = (np.sin(im*np.pi/2 - np.pi/2))**2
        # im = noise_shift(im, (im.shape[0]/500)*10)
        # im = gaussian_filter(im, 6)
        # im = np.exp(10*(im+1))
        im = noise_shift(im, (im.shape[0]/500)*10)
        # im_max = np.max(im)
        # im += im * np.random.random(im.shape)/10 + np.random.random()*0.5*np.random.normal(im_max/100, np.std(im), im.shape)

        im = 255 * quick_norm(im)

        im_mid = int(im.shape[0]/2)
        im_crop = int(im.shape[0]/4)
        crop_im = im[im_mid - im_crop:im_mid + im_crop, im_mid - im_crop:im_mid + im_crop]

        
        im = zoom(crop_im, 224/(im.shape[0]/2))
        print(im.shape)
        im = np.round(im, decimals=1) / 255
        images.append(im)
    return images

if __name__ == '__main__':

    plt.style.use(['science', 'ieee', 'no-latex'])

    plt.rcParams.update({
        'figure.figsize': [6.3, 6.3],
        'font.size': 12,
        'figure.dpi': 100,
        'savefig.dpi': 300
    })

    root_dir = r"C:\Users\Pouis\OneDrive - Imperial College London\202403_link - Photon BEC's files\Cropped_Images\20240321"
    file_list = [
        r"pbec_20240321_121859_500000.0_0.15617534482758622_955.1035766601562_1.105263157894737_.png",
        r"pbec_20240320_213939_250000.0_0.12069137931034483_950.73681640625_-1.0_.png",
        r"pbec_20240320_214633_3907.0_0.1612444827586207_950.5794067382812_-0.6551724137931034_.png",
        r"pbec_20240320_223811_62499.0_0.09534568965517243_947.4107055664062_2.4482758620689657_.png",
    ]

    # plt.imshow(sigmoid((im - np.min(im)) / (np.max(im) - np.min(im)), 0.01,0.1))
    # plt.show()

    bright_list = [
        40, 35, 47, 100
    ]
    blur_list = [
        [9,0.7], [10,10], [15,3], [1,100]
    ]
    modes_list = [
        [([0,1], False ,160 - 90, 200*um, 0.05, 1), ([0,9], False ,70 + 90, 200*um, 1, 0.6)],
        [([0,1], False , 30 , 200*um, 0.2, 1), ([0,4], False ,50 + 90, 200*um, 1, 0.7)],
        [([0,4], False , -27, 200*um, 1, 0.4), ([0,1], False , 60, 200*um, 0.05, 1), ([0,9], False , -23, 200*um, 1, 0.7), ([0,6], False , -26, 200*um, 1, 0.4)],
        [([0,0], False, False, 280*um, 1, 1)]
    ]

    sz = bright_list[1]
    bl = blur_list[1]


    im = cv2.imread(root_dir + sep + 'Crop' + file_list[1], 0)
    im2 = gererate_data(1, 2500*um, 500, modes_list[1], [100*um, 210*um], fringe_size=[0.3, 0.6],  mult_las_split=0)[0]


    im = sigmoid(quick_norm(im), 0.05, 0.1)
    im2 = sigmoid(quick_norm(im2), 0.05, 0.1)
    im = quick_norm(im)
    im2 = quick_norm(im2)
    plt.imshow(im2)
    plt.show()
    plt.plot(im2[112])
    plt.plot(im[112])
    plt.show()




    from power_length_analysis import grid_plot

    fig, axes, gs = grid_plot(8, 4, 2, 0.2, 0.1)

    for i, file in enumerate(file_list):
        im = cv2.imread(root_dir + sep + 'Crop' + file, 0)
        axes[i].imshow(im, aspect = 'auto', cmap = 'inferno')
        axes[i].set_yticklabels([])
        axes[i].set_xticklabels([])
        if i == 0:
            axes[i].set_ylabel('Experimental')
    
    for i, mode in enumerate(modes_list):
        sz = bright_list[i]
        bl = blur_list[i]
        im = gererate_data(1, 2500*um, 500, mode, [100*um, 210*um], fringe_size=[0.3, 0.6],  mult_las_split=0)[0]
        axes[i+4].imshow(im, aspect = 'auto')
        axes[i+4].set_yticklabels([])
        axes[i+4].set_xticklabels([])
        if i == 0:
            axes[i+4].set_ylabel('Training')
    
    # plt.savefig(r'C:\Users\Pouis\OneDrive - Imperial College London\Masters\Thesis\Thesis_Plots\CNN plots\TrainingVSExperimental.pdf', format = 'pdf')
    plt.show()

