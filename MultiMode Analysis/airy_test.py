from LightPipes import *
import matplotlib.pyplot as plt


beam = Begin(30*mm, 784, 500)

beam = AiryBeam2D(beam,x0=0.1*mm, y0=0.1*mm)

I = Intensity(beam)

plt.imshow(I)

plt.show()
