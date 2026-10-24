import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

PRE, POST = 0.25, 0.65   # seconds kept before / after each peak

def bandpass(x, fs=100, lo=0.5, hi=8.0):
    """Remove slow drift (below lo Hz) and fast noise (above hi Hz)."""
    b, a = butter(3, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)

def cut_beats(x, fs=100):
    """Find pulse peaks and cut a fixed window around each one."""
    peaks, _ = find_peaks(x, distance=int(0.4 * fs), prominence=0.5 * np.std(x))
    before, after = int(PRE * fs), int(POST * fs)
    keep_peaks, beats = [], []
    for p in peaks:
        if p - before >= 0 and p + after <= len(x):   # skip beats cut off at the edges
            beats.append(x[p - before:p + after])
            keep_peaks.append(p)
    return np.array(keep_peaks), np.array(beats)

def quality_gate(beats, min_corr=0.9):
    """True for beats that resemble the typical (median) beat."""
    template = np.median(beats, axis=0)
    return np.array([np.corrcoef(b, template)[0, 1] > min_corr for b in beats])