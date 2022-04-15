import cv2
from dv import AedatFile
import librosa
import numpy as np
import os
import pandas as pd
import sys

sys.path.append("../utils")
from dv_utils import *
from bitcode import *


def getAlignment(codes1: pd.DataFrame, codes2: pd.DataFrame, tol: float=0.1) -> pd.DataFrame:
    
    overlap = np.intersect1d(codes1['code'].values, codes2['code'].values)
    align = [codes1[codes1['code']==code]['time'].values[0] - codes2[codes2['code']==code]['time'].values[0] for code in overlap]

    alignment = pd.DataFrame({'code': overlap, 'align': align})

    mean_align = np.mean(align)

    alignment['error'] = alignment['align'] - mean_align

    # bad = (np.abs(alignment['error']) > tol).sum()
    # if (bad > 0):
    #     print('[WARNING] ' + str(bad) + ' bad align(s) found. ')
    #     alignment = alignment[np.abs(alignment['error']) <= tol]

    return alignment, np.mean(align), np.mean(alignment['error']**2)



def getWvSlice(wv_path: str, tw):

    x, sr = librosa.load(wv_path, sr=None)

    tw_bool = (np.arange(x.shape[0]) >= np.floor(tw[0] * sr).astype(int)) & (np.arange(x.shape[0]) < np.floor(tw[1] * sr).astype(int))
    x = x[tw_bool]

    return {'x': x, 'sr': sr}



def getDvSlice(dv_path, tw, fps, roi):

    x, sr = accumulate(dv_path, fps=fps, roi=roi, tw_mode=True, tw=tw).astype(float), fps

    return {'x': x, 'sr': sr}




class AlignedDvSound():

    @property
    def align(self):
        return self._align


    def __init__(self, dv_path: str, wv_path: str):

        self._dv_path = dv_path
        self._wv_path = wv_path


    def alignFromCodes(self, dv_codes: pd.DataFrame, wv_codes: pd.DataFrame, tol: float = 0.01*2) -> None:

        self._dv_codes = dv_codes
        self._wv_codes = wv_codes
        
        alignment, align, mse = getAlignment(self._dv_codes, self._wv_codes)

        if mse > tol:
            print('[WARNING] bad alignment with mse = ' + str(mse))

        self._align = align


    def autoAlign(self, time_window, roi, fps=5000, tol: float = 0.01*2):

        self._dv_codes = dv_decode(self._dv_path, tw=time_window, roi=roi, fps=fps)
        self._wv_codes = wv_decode(self._wv_path, tw=time_window)

        alignment, align, mse = getAlignment(self._dv_codes, self._wv_codes)

        if mse > tol:
            print('[WARNING] bad alignment with mse = ' + str(mse))

        self._align = align


    def getSlice(self, time_window, coord: str, roi, fps=5000):
        '''
        time_window: [start, end] in seconds
        coord = 'dv' or 'wv'
        '''

        if coord == 'dv':
            dv_tw = time_window
            wv_tw = time_window - self._align
        elif coord == 'wv':
            dv_tw = time_window + self._align
            wv_tw = time_window
        else:
            raise(ValueError(coord + ' is not valid. "dv" or "wv" only. '))

        print('\nslicing dv...')
        dv_slice = getDvSlice(self._dv_path, dv_tw, fps=fps, roi=roi)
        print('\nslicing wv...')
        wv_slice = getWvSlice(self._wv_path, wv_tw)
        print('\ndone')

        return {'dv': dv_slice, 'wv': wv_slice}



    

