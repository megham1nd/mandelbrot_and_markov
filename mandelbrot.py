import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
np.warnings.filterwarnings("ignore")  # type: ignore

def mandelbrot_plot(n, N, thresh):
    '''
    Returns computed Mandelbrot set within an nxn grid in the range
    [-2, 1] x [-1.5i, 1.5i]. 
    The divergence test |z| < thresh is run on each point in the grid
    following N iterations of the Mandelbrot function.
    '''
    x = np.linspace(-2, 1, n)
    y = np.linspace(-1.5, 1.5, n)

    # creating the grid of points on which to run the Mandelbrot
    # function
    grid_x, grid_y = np.meshgrid(x, y)

    # running the Mandelbrot function on each point in the grid
    mb_set = mandelbrot_func(grid_x, grid_y, N)
    # applying a binary mask on the resulting set depending on 
    # the threshold condition |z| < thresh:
    # 1 if the point belongs in the set
    # 0 if the point does not belong in the set
    mask = np.zeros([n, n])
    mask[np.abs(mb_set) < thresh] = 1   

    # plotting the resulting set
    plt.imshow(np.abs(mb_set), extent = [-2, 1, -1.5, 1.5])
    plt.gray()
    plt.savefig('mandelbrot.png')

def mandelbrot_func(x, y, N):
    '''
    Returns product (z) of Mandelbrot function according to input 
    variables x and y after N iterations:

    z = z^2 + (x + yi)
    '''
    z = 0
    c = x + 1j*y

    for i in range(N):
        z = z**2 + c
    
    return z

mandelbrot_plot(1000, 50, 50)