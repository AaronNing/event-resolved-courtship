import numpy as np
from dv import AedatFile
import librosa
import matplotlib.pyplot as plt
import sys

sys.path.append("../utils")
from utils import *
from dv_utils import *



def show_spec(x, sr, n_fft = 2048, yscale = 'log', time_range = None):
    if time_range != None:
        idx_range = np.floor(np.array(time_range) * sr).astype(int)
        x = x[idx_range[0] : idx_range[1]]
    X = librosa.stft(x, n_fft=n_fft)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5), dpi=300)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis=yscale)
    plt.colorbar()
    plt.show()

def getPerc(x, sr, n_fft = 2048, time_range = None):
    if time_range != None:
        idx_range = np.floor(np.array(time_range) * sr).astype(int)
        x = x[idx_range[0] : idx_range[1]]
    X = librosa.stft(x, n_fft=n_fft)
    X_harmonic, X_percussive = librosa.decompose.hpss(X)
    x_perc = librosa.istft(X_percussive)
    return x_perc

def getPulseOnsets(x, pre_max, post_max, show=False):
    x_pow = x**2
    peaks = librosa.util.peak_pick(x_pow, pre_max=pre_max, post_max=post_max, pre_avg=pre_max, post_avg=post_max, delta=0, wait=0)
    # peaks = signal.find_peaks(x_pow, height=max(x_pow)/5, distance=1*sr)[0]
    thresh = x_pow.std()*5
    peaks = peaks[x_pow[peaks] > thresh]
    
    first_peaks = peaks.copy()
    for i, peak in enumerate(peaks):
        window = np.arange(max(peak - int(pre_max), 0), min(peak + int(post_max), len(x_pow)-1), 1)
        first_peaks[i] = window[np.where(x_pow[window] > x_pow[peak] / 3)[0][0]]

    if show:
        plt.figure(figsize=[14,4], dpi=300)
        plt.plot(x_pow, linewidth=0.5)
        plt.vlines(first_peaks, max(x_pow)*1.1, plt.ylim()[1], colors='r', linewidth=1)
        plt.show()
    
    return first_peaks


def decodeBcode(bc: np.ndarray, head_interval=0.11, width=0.06, tol=0.02, show=False):
    '''
    tol: tolerance of time drift (s)
    '''
    bc = bc - bc[0]
    times = np.arange(8) * width + head_interval
    n_2 = (np.array([np.sum((bc < time + tol) & (bc > time - tol)) for time in times]) > 0).astype(int)[::-1]
    n = int(np.array2string(n_2, separator='')[1:-1], 2)

    if show:
        plt.figure()
        plt.vlines(bc, 0, 1, colors='k')
        plt.vlines(times, 1, 2, colors='r')
        plt.title(str(n))
        plt.show()

    return n


def decodeBitcodeTrain(peaks: np.ndarray, head_interval=0.11, width=0.06, sig_interval=1, show=False) -> list:
    dif = np.diff(peaks)
    idx_heads = np.where(dif > sig_interval*0.8)[0] + 1

    codes = [] # (time, n)
    for ih, it in zip(idx_heads[:-1], idx_heads[1:]):
        bcode = peaks[ih:it]
        codes.append((peaks[ih], decodeBcode(bcode, head_interval, width, show=show)))

    return codes



def wv_decode(wv_path, tw):

    x, sr = librosa.load(wv_path, sr=None)

    x_perc = getPerc(x, sr, time_range=tw)
    pulses = getPulseOnsets(x_perc, pre_max=0.03*sr, post_max=0.03*sr)
    codes = decodeBitcodeTrain(pulses/sr)

    codes = pd.DataFrame(codes, columns=['time', 'code'])

    return codes



def dv_decode(dv_path, tw, roi, fps=5000):

    x, sr = accumulate(dv_path, fps=fps, roi=roi, tw_mode=True, tw=tw).astype(float), fps

    x_perc = getPerc(x, sr, n_fft=2048)
    pulses = getPulseOnsets(x_perc, 0.03*sr, 0.03*sr)
    codes = decodeBitcodeTrain(pulses/sr)

    codes = pd.DataFrame(codes, columns=['time', 'code'])

    return codes
    


