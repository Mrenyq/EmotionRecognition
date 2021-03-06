from scipy.signal import butter, lfilter, convolve
import numpy as np
from datetime import datetime
from scipy import signal


def valArToLabels(y, soft=False):
    if soft is False:
        if (y < 3):
            return 0
        elif (y == 3):
            return 1
        else:
            return 2

    else:
        if y >= 2 & y <= 4:
            return np.array([0.1, 0.8, 0.1])
        elif (y < 2):
            return np.array([0.8, 0.2, 0])
        else:
            return np.array([0.0, 0.2, 0.8])


def arToLabels(y):
    if (y < 3):
        return 0
    else:
        return 1


def valToLabels(y):
    if (y < 3):
        return 0

    else:
        return 1


def arValMulLabels(ar, val):
    if ar == 0 and val == 0:
        return 0
    elif ar == 0 and val == 1:
        return 1
    elif ar == 1 and val == 0:
        return 2
    else:
        return 3


def convertLabels(ar, val):
    labels = np.ones_like(ar)
    labels[(ar == 0) & (val == 1)] = 1
    labels[(ar == 1) & (val == 0)] = 2
    labels[(ar == 1) & (val == 1)] = 3
    return labels


def convertContrastiveLabels(time1, time2, sub1, sub2):
    if sub1 == sub2:
        if (time1 + 45 <= time2) or (time1 - 45 >= time2):
            return 0
        else:
            return 1
    else:
        return 1


def windowFilter(x, numtaps=120, cutoff=2.0, fs=256.):
    b = signal.firwin(numtaps, cutoff, fs=fs, window='hamming', pass_zero='lowpass')
    y = lfilter(b, [1.0], x)
    return y


def utcToTimeStamp(x):
    utc = datetime.fromtimestamp(x / 1000).strftime('%Y-%m-%d %H:%M:%S.%f')
    return timeToInt(utc)


def timeToInt(time):
    hours = time.split(" ")[-1]
    h, m, s = hours.split(":")
    inttime = 3600 * float(h) + 60 * float(m) + float(s)

    return inttime


def butterBandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butterBandpassFilter(data, lowcut, highcut, fs, order=5):
    b, a = butterBandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def avgSlidingWindow(x, n):
    '''
    Smooth a signal using a sliding window with n columns
    :param x: a signal
    :param n: number of columns
    :return: the smoothed signal
    '''

    window = np.ones(n) / n
    filtered = convolve(x, window, mode="same")

    return filtered


def rollingWindow(a, size=50):
    slides = []
    for i in range(len(a) // size):
        slides.append(a[(i * size):((i + 1) * size)])

    return np.array(slides)
