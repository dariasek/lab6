import random

import numpy as np


def separate_dots(dots, c_x, c_y, r, epsilon):
    inner_points = dots[((dots[:, 0] - c_x)**2 + (dots[:, 1] - c_y)**2 < r**2 - epsilon).flatten(), :]
    outer_points = dots[((dots[:, 0] - c_x)**2 + (dots[:, 1] - c_y)**2 > r**2 + epsilon).flatten()]
    outer_points = outer_points[:inner_points.shape[0], :]
    return inner_points, outer_points


def generate_dots(x_min, y_min, x_max, y_max, n_samples, seed=None):
    """Generate n_samples dots inside the bounding box with coordinates (x_min,y_min), (x_max, y_max)
    
    >>> generate_dots(0, 0, 1, 1, 1, 0)
    array([[0.549 , 0.7153]], dtype=float16)
    >>> generate_dots(0, 0, 1, 1, 5, 0)
    array([[0.549 , 0.646 ],
           [0.7153, 0.4375],
           [0.6025, 0.8916],
           [0.545 , 0.964 ],
           [0.4236, 0.3835]], dtype=float16)
    >>> generate_dots(-10, -10, 0, 0, 1, 0)
    array([[-4.51 , -2.848]], dtype=float16)
    >>> generate_dots(-10, -10, 0, 0, 5, 0)
    array([[-4.51  , -3.541 ],
           [-2.848 , -5.625 ],
           [-3.973 , -1.082 ],
           [-4.55  , -0.3633],
           [-5.76  , -6.164 ]], dtype=float16)
    >>> generate_dots(5, 5, 5, 5, 1, 0)
    array([[5., 5.]], dtype=float16)
    >>> generate_dots(0, 0, 1, 1, 3, 0)
    array([[0.549 , 0.545 ],
           [0.7153, 0.4236],
           [0.6025, 0.646 ]], dtype=float16)
    """
    np.random.seed(0)
    random_xs = np.array([(np.random.rand()*(x_max - x_min) + x_min) for i in range(n_samples)]).reshape(
                        (-1, 1)).astype(np.float16)
    random_ys = np.array([(np.random.rand()*(y_max - y_min) + y_min) for i in range(n_samples)]).reshape(
                            (-1,1)).astype(np.float16)
    random_dots = np.concatenate([random_xs, random_ys], axis=1)
    return random_dots
