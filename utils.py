import random

import numpy as np


def separate_dots(dots, c_x, c_y, r, epsilon):
    inner_points = dots[((dots[:, 0] - c_x)**2 + (dots[:, 1] - c_y)**2 < r**2 - epsilon).flatten(), :]
    outer_points = dots[((dots[:, 0] - c_x)**2 + (dots[:, 1] - c_y)**2 > r**2 + epsilon).flatten()]
    outer_points = outer_points[:inner_points.shape[0], :]
    return inner_points, outer_points


def shuffle_dataset(x, y):
    indices = np.arange((x.shape[0]))
    random.shuffle(indices)
    x = x[indices, :]
    y = y[indices]
    return x, y


def generate_dots(x_min, y_min, x_max, y_max, n_samples):
    """Generate n_samples dots inside the bounding box with coordinates (x_min,y_min), (x_max, y_max)
    """
    random_xs = np.array([(np.random.rand()*(x_max - x_min) + x_min) for i in range(n_samples)]).reshape(
                        (-1, 1)).astype(np.float16)
    random_ys = np.array([(np.random.rand()*(y_max - y_min) + y_min) for i in range(n_samples)]).reshape(
                            (-1,1)).astype(np.float16)
    random_dots = np.concatenate([random_xs, random_ys], axis=1)
    return random_dots
