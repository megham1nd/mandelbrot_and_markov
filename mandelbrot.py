import numpy as np
np.warnings.filterwarnings("ignore")
import scipy as sci
import matplotlib.pyplot as plt

def mandelbrot_plot(n, N, thresh):
    x = np.linspace(-2, 1, n)
    y = np.linspace(-1.5, 1.5, n)

    grid_x, grid_y = np.meshgrid(x, y)

    mask = mandelbrot_func(grid_x, grid_y, N)
    mb_set = np.zeros([n, n])
    mb_set[np.abs(mask) < thresh] = 1   

    plt.imshow(np.abs(mb_set), extent = [-2, 1, -1.5, 1.5])
    plt.gray()
    plt.savefig('mandelbrot.png')

def mandelbrot_func(x, y, N):
    z = 0
    c = x + 1j*y

    for i in range(N):
        z = z**2 + c
    
    return z

mandelbrot_plot(1000, 50, 50)