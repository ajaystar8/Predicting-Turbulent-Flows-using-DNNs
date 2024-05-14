from config import *


# Function defining nine ODEs
def MoehlisCoefficientsGenerator(y0):
    # setting up of initial coefficient values
    a1, a2, a3, a4, a5, a6, a7, a8, a9 = y0

    # constants' definition
    k_ag = np.sqrt(np.square(a) + np.square(g))
    k_bg = np.sqrt(np.square(b) + np.square(g))
    k_ab = np.sqrt(np.square(a) + np.square(b))
    k_abg = np.sqrt(np.square(a) + np.square(b) + np.square(g))

    # calculation of differential coefficients - Equations as it is from Srinivasan's paper
    da1dt = (b ** 2 / Re) * (1 - a1) - np.sqrt(3 / 2) * \
            ((b * g / k_abg) * a6 * a8 - (b * g / k_bg) * a2 * a3)

    da2dt = -(4 * b ** 2 / 3 + g ** 2) * (a2 / Re) + ((5 * np.sqrt(2) * g ** 2) / (3 * np.sqrt(3) * k_ag)) * a4 * a6 - (
            g ** 2 / (np.sqrt(6) * k_ag)) * a5 * a7 - \
            ((a * b * g) / (np.sqrt(6) * k_ag * k_abg)) * a5 * a8 - np.sqrt(3 / 2) * ((b * g) / k_bg) * a1 * a3 - \
            np.sqrt(3 / 2) * ((b * g) / k_bg) * a3 * a9

    da3dt = -(k_bg ** 2 / Re) * a3 + ((2 * a * b * g) / (np.sqrt(6) * k_ag * k_bg)) * (a4 * a7 + a5 * a6) + \
            ((b ** 2 * (3 * a ** 2 + g ** 2) - 3 * g ** 2 * k_ag ** 2) / (np.sqrt(6) * k_ag * k_bg * k_abg)) * a4 * a8

    da4dt = -((3 * a ** 2 + 4 * b ** 2) / (3 * Re)) * a4 - (a / np.sqrt(6)) * a1 * a5 - (
            (10 * a ** 2) / (3 * np.sqrt(6) * k_ag)) * a2 * a6 - \
            np.sqrt(3 / 2) * ((a * b * g) / (k_ag * k_bg)) * a3 * a7 - np.sqrt(3 / 2) * (
                    (a ** 2 * b ** 2) / (k_ag * k_bg * k_abg)) * a3 * a8 - \
            (a / np.sqrt(6)) * a5 * a9

    da5dt = -(k_ab ** 2 / Re) * a5 + (a / np.sqrt(6)) * a1 * a4 + (a ** 2 / (np.sqrt(6) * k_ag)) * a2 * a7 - \
            (a * b * g / (np.sqrt(6) * k_ag * k_abg)) * a2 * a8 + (a / np.sqrt(6)) * a4 * a9 + \
            ((2 * a * b * g) / (np.sqrt(6) * k_ag * k_bg)) * a3 * a6

    da6dt = -((3 * a ** 2 + 4 * b ** 2 + 3 * g ** 2) / (3 * Re)) * a6 + (a / np.sqrt(6)) * a1 * a7 + (
            np.sqrt(3 / 2) * b * g / k_abg) * a1 * a8 + \
            (10 * (a ** 2 - g ** 2) / (3 * np.sqrt(6) * k_ag)) * a2 * a4 - (
                    2 * np.sqrt(2 / 3) * a * b * g / (k_ag * k_bg)) * a3 * a5 + \
            (a / np.sqrt(6)) * a7 * a9 + (np.sqrt(3 / 2) * b * g / k_abg) * a8 * a9

    da7dt = -(k_abg ** 2 / Re) * a7 - a * (a1 * a6 + a6 * a9) / np.sqrt(6) + (
            (g ** 2 - a ** 2) / (np.sqrt(6) * k_ag)) * a2 * a5 + \
            ((a * b * g) / (np.sqrt(6) * k_ag * k_bg)) * a3 * a4

    da8dt = -(k_abg ** 2 / Re) * a8 + ((2 * a * b * g) / (np.sqrt(6) * k_ag * k_abg)) * a2 * a5 + \
            (g ** 2 * (3 * a ** 2 - b ** 2 + 3 * g ** 2) / (np.sqrt(6) * k_ag * k_bg * k_abg)) * a3 * a4

    da9dt = -(9 * b ** 2 / Re) * a9 + (np.sqrt(3 / 2) * b * g / k_bg) * \
            a2 * a3 - (np.sqrt(3 / 2) * b * g / k_abg) * a6 * a8

    # returning the differential coefficients in a list format
    func = [da1dt, da2dt, da3dt, da4dt, da5dt, da6dt, da7dt, da8dt, da9dt]
    return func
