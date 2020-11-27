from scipy.signal import butter, lfilter, convolve
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import signal


def valArLevelToLabels(y):
    if (y < 3):
        return 0
    else:
        return 1


def windowFilter(x, numtaps=120, cutoff=2.0, fs=256.):
    b = signal.firwin(numtaps, cutoff, fs=fs, window='hamming', pass_zero='lowpass')
    y = lfilter(b, [1.0], x)
    return y


def strTimeToUnixTime(time, form='%Y/%m/%d %H:%M:%S'):
    dt = datetime.strptime(time, form)
    unixtime = dt.timestamp()
    return unixtime


def unixTimeToStrTime(unixtime):
    dt = datetime.fromtimestamp(unixtime)
    time = dt.strftime('%Y/%m/%d %H:%M:%S')
    return time


def utcToTimeStamp(x):
    utc = datetime.fromtimestamp(x/1000).strftime('%Y-%m-%d %H:%M:%S.%f')
    return timeToInt(utc)


def timeToInt(time):
    date, hours = time.split(" ")
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

def splitEEGPerSubject(eeg_data: pd.DataFrame, timedata: pd.DataFrame, subjects: list, fs=1000):
    eeg_split = []
    idx = 0
    for s in subjects:
        time_start = strTimeToUnixTime(timedata.at[s, 'Time_Start'])
        time_end = strTimeToUnixTime(timedata.at[s, 'Time_End'])
        noise_start = strTimeToUnixTime(timedata.at[s, 'Noise_Start'])
        noise_end = strTimeToUnixTime(timedata.at[s, 'Noise_End'])
        timestamp = np.arange(time_start, time_end, 1 / fs)
        timestamp = list(map(unixTimeToStrTime, timestamp))
        ms = np.arange(0, len(timestamp), 1000 / fs, dtype=int)
        eeg = eeg_data.iloc[idx:idx+len(timestamp), 2:21]
        eeg = eeg.reset_index(drop=True)

        eeg_split_df = pd.DataFrame(timestamp)
        ms = pd.DataFrame(ms)
        eeg_split_df = pd.concat([eeg_split_df, ms, eeg], axis=1)
        eeg_split.append(eeg_split_df)
        idx += len(timestamp) + int(((noise_end - noise_start) * fs))

    return eeg_split


def splitEEGPerVideo(eeg_data: pd.DataFrame, gameresults: pd.DataFrame, fs=1000):
    form_eegtime = '%Y/%m/%d %H:%M:%S'
    form_gameresulttime = '%Y-%m-%d %H:%M:%S'
    eeg_timedata = [strTimeToUnixTime(t, form_eegtime) for t in eeg_data.iloc[:, 0]]
    time_start = [strTimeToUnixTime(t, form_gameresulttime) for t in gameresults['Time_Start'].values.tolist()]
    time_end = [strTimeToUnixTime(t, form_gameresulttime) for t in gameresults['TestTime_End'].values.tolist()]
    eeg_split = []
    eeg_timedata = np.array(eeg_timedata)
    for start, end in zip(time_start, time_end):
        split_time = (eeg_timedata >= start) & (eeg_timedata <= end)
        eeg_split_df = eeg_data.iloc[split_time]
        ms = np.arange(0, len(eeg_split_df), 1000 / fs, dtype=int)
        eeg_split_df['ms'] = ms
        eeg_split.append(eeg_split_df)

    return eeg_split

