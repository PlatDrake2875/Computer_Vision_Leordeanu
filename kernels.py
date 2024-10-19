import numpy as np
import plotly.graph_objects as go
from numpy import ndarray


def compute_gauss(X: ndarray, Y: ndarray, sigma: float) -> ndarray:
    return np.exp(- (X ** 2 + Y ** 2) / (2 * sigma ** 2))


def gaussian_kernel(size: int, sigma: float) -> ndarray:
    if size % 2 == 0:
        raise ValueError("Gaussian kernel size can't be even!")
    if size < 0:
        raise ValueError("Gaussian kernel size can't be negative!")

    r = size // 2
    x = np.arange(-r, r + 1)
    y = np.arange(-r, r + 1)
    X, Y = np.meshgrid(x, y)

    kernel = compute_gauss(X, Y, sigma)
    kernel = kernel / np.sum(kernel)
    # show_gaussian_kernel(kernel)
    return kernel


def show_gaussian_kernel(kernel: ndarray):
    r = len(kernel[0]) // 2
    x = np.arange(-r, r + 1)
    y = np.arange(-r, r + 1)
    X, Y = np.meshgrid(x, y)

    fig = go.Figure(data=[go.Surface(z=kernel, x=X, y=Y)])
    fig.update_layout(title='Interactive 2D Gaussian Filter',
                      scene=dict(
                          xaxis_title='X-axis',
                          yaxis_title='Y-axis',
                          zaxis_title='Filter Value'),
                      autosize=False,
                      width=700,
                      height=700,
                      margin=dict(l=65, r=50, b=65, t=90))
    fig.show()


def box_kernel(size: int) -> ndarray:
    kernel = np.ones((size, size)) / size
    return kernel
