import numpy as np


def gaussian_kernel10(distances):
    weights = np.exp((-10 * (distances ** 2)))
    return weights / np.sum(weights)


def gaussian_kernel30(distances):
    weights = np.exp((-30 * (distances ** 2)))
    return weights / np.sum(weights)


def gaussian_kernel50(distances):
    weights = np.exp((-50 * (distances ** 2)))
    return weights / np.sum(weights)


def gaussian_kernel100(distances):
    weights = np.exp((-100 * (distances ** 2)))
    return weights / np.sum(weights)


def gaussian_kernel1000(distances):
    weights = np.exp((-1000 * (distances ** 2)))
    return weights / np.sum(weights)